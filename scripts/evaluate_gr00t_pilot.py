#!/usr/bin/env python3
"""Evaluate fixed pilot windows in a fresh process; never send robot commands."""
import argparse
import dataclasses
from enum import Enum
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from project_config import PROJECT_ROOT, load_config, resolve_path, guard_derived, guard_path
from convert_single_camera import ensure_python
from run_gr00t_smoke import GIB, available_memory, existing_ancestor, prepare_offline_workdir

GROUPS = [('left_arm', 0, 7, 'radians'), ('right_arm', 7, 14, 'radians'),
          ('left_hand', 14, 20, 'normalized hand command'), ('right_hand', 20, 26, 'normalized hand command')]
SEEDS = [42, 43, 44]
HORIZON = 16


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def semantic_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sample_windows(episodes):
    if len(episodes) != 5:
        raise ValueError('This bounded pilot requires exactly five episodes')
    result = []
    for expected_index, episode in enumerate(episodes):
        length = int(episode['length'])
        if episode['episode_index'] != expected_index or length-HORIZON < 19:
            raise ValueError('Episode indices or full-horizon sample count are invalid')
        for i in range(20):
            start = i*(length-HORIZON)//19
            result.append({'episode_index':expected_index, 'start':start, 'length':length,
                           'progress':start/(length-HORIZON)})
    return result


def dataset_identity(dataset, pilot_manifest):
    files = {}
    for path in sorted(dataset.rglob('*')):
        if path.is_symlink():
            raise ValueError('Evaluation dataset must not contain symlinks')
        if path.is_file():
            files[str(path.relative_to(dataset))] = digest(path)
    if not files:
        raise ValueError('Empty evaluation dataset')
    episodes = [json.loads(line) for line in (dataset/'meta/episodes.jsonl').read_text().splitlines() if line]
    pilot = json.loads(pilot_manifest.read_text())
    if [x['rows'] for x in pilot['episodes']] != [x['length'] for x in episodes]:
        raise ValueError('Frozen pilot and converted episode lengths differ')
    return {'dataset_files_sha256':files, 'pilot_manifest_sha256':digest(pilot_manifest),
            'samples':sample_windows(episodes), 'horizon':HORIZON, 'seeds':SEEDS,
            'joint_order':[name for name,_,_,_ in GROUPS], 'denoising_steps':4}


def freeze_samples(path, identity):
    if path.exists():
        if path.is_symlink() or json.loads(path.read_text()) != identity:
            raise ValueError('Evaluation manifest differs from frozen dataset or sampling recipe')
    else:
        write_new_json(path, identity)


def summarize(prediction, target, state, lower, upper):
    """All errors use decoded absolute actions; normalized errors are not clipped."""
    import numpy as np
    prediction, target, state = map(np.asarray, (prediction, target, state))
    lower, upper = np.broadcast_to(lower, (HORIZON,26)), np.broadcast_to(upper, (HORIZON,26))
    if prediction.shape != (len(target),len(SEEDS),HORIZON,26) or target.shape[1:] != (HORIZON,26) or state.shape != (len(target),26):
        raise ValueError('Evaluation action/state shapes differ from the fixed recipe')
    if not all(np.isfinite(value).all() for value in (prediction,target,state,lower,upper)):
        raise ValueError('Nonfinite predictions, targets, states, or normalization bounds')
    if (upper < lower).any():
        raise ValueError('Reversed normalization bounds')
    span = upper-lower
    active = ~np.isclose(upper,lower)
    safe = np.where(active,span,1.0)
    motion = np.where(active[None],np.abs(target-state[:,None,:])/safe[None],0.0)
    moving = np.stack([np.max(motion[:,:,start:end],axis=(1,2)) > 0.05 for _,start,end,_ in GROUPS],axis=1)
    persistence = np.broadcast_to(state[:,None,None,:],prediction.shape)
    result = {}
    for group_index,(name,start,end,unit) in enumerate(GROUPS):
        valid = active[:,start:end]
        error = prediction[:,:,:,start:end]-target[:,None,:,start:end]
        baseline = persistence[:,:,:,start:end]-target[:,None,:,start:end]
        def metrics(values, sample_mask):
            selected = values[sample_mask]
            if not selected.size:
                return {'sample_count':0,'mae':None,'rmse':None,'normalized_mae':None,'normalized_rmse':None,'per_seed_mae':[]}
            normalized = (2*selected/safe[None,None,:,start:end])[:,:,valid]
            return {'sample_count':int(sample_mask.sum()),'mae':float(np.abs(selected).mean()),
                    'rmse':float(np.sqrt(np.square(selected).mean())),
                    'normalized_mae':float(np.abs(normalized).mean()) if normalized.size else None,
                    'normalized_rmse':float(np.sqrt(np.square(normalized).mean())) if normalized.size else None,
                    'per_seed_mae':[float(np.abs(selected[:,i]).mean()) for i in range(len(SEEDS))]}
        result[name] = {'unit':unit,'nonconstant_horizon_joint_count':int(valid.sum()),
            'constant_horizon_joint_count':int(valid.size-valid.sum()),
            'all':{'model':metrics(error,np.ones(len(target),dtype=bool)), 'persistence':metrics(baseline,np.ones(len(target),dtype=bool))},
            'commanded_motion':{'model':metrics(error,moving[:,group_index]),'persistence':metrics(baseline,moving[:,group_index])},
            'small_commanded_motion':{'model':metrics(error,~moving[:,group_index]),'persistence':metrics(baseline,~moving[:,group_index])}}
    return result, moving, active


def verify_reference(reference, manifest_sha, semantics_sha, target, state, moving):
    import numpy as np
    report = json.loads(reference.read_text())
    if not report.get('passed') or report.get('sample_manifest_sha256') != manifest_sha or report.get('processor_semantics_sha256') != semantics_sha:
        raise ValueError('Reference evaluation manifest or saved processor semantics differ')
    array_path = reference.with_suffix('.npz')
    if report.get('artifacts', {}).get(array_path.name) != digest(array_path):
        raise ValueError('Reference prediction artifact hash differs')
    with np.load(array_path,allow_pickle=False) as data:
        for key,value in [('targets',target),('states',state),('commanded_motion_mask',moving)]:
            if not np.array_equal(data[key],value):
                raise ValueError('Reference '+key+' differ')
        prediction = data['absolute_predictions'].copy()
    return report,prediction


def plot_trajectories(path, samples, prediction, target, state, reference=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5,4,figsize=(18,14),squeeze=False)
    for episode in range(5):
        indices = np.array([i for i,s in enumerate(samples) if s['episode_index']==episode])
        progress = [samples[i]['progress'] for i in indices]
        for column,(name,start,end,unit) in enumerate(GROUPS):
            # Choose the joint by target variation only, identically for both models.
            dim = start+int(np.argmax(np.ptp(target[indices,:,start:end],axis=(0,1))))
            axis = axes[episode,column]
            mean = prediction[indices,:,-1,dim].mean(axis=1)
            std = prediction[indices,:,-1,dim].std(axis=1)
            axis.plot(progress,target[indices,-1,dim],label='target at +15',color='black')
            axis.plot(progress,state[indices,dim],label='current state',color='gray',linestyle=':')
            if reference is not None:
                axis.plot(progress,reference[indices,:,-1,dim].mean(axis=1),label='two-step reference',color='tab:orange')
            axis.plot(progress,mean,label='evaluated model',color='tab:blue')
            axis.fill_between(progress,mean-std,mean+std,color='tab:blue',alpha=.15)
            axis.set_title(f'Episode {episode}, {name}[{dim-start}]')
            axis.set_xlabel('Episode progress');axis.set_ylabel(unit)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Pilot training-set trajectories: horizon +15, mean and standard deviation of three paired seeds')
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--pilot-manifest',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='Completed two-step evaluation JSON for paired candidate comparison')
    args=parser.parse_args();config=load_config(args.config)
    checkpoint=resolve_path(args.checkpoint);dataset=guard_derived(args.dataset,config)
    report_path=guard_path(args.report,config,'reports');manifest_path=guard_path(args.manifest,config,'reports')
    pilot_manifest=guard_derived(args.pilot_manifest or config['paths']['derived']/'pilot-v1/manifest.json',config)
    reference=guard_path(args.reference,config,'reports') if args.reference else None
    for output in [report_path,report_path.with_suffix('.npz'),report_path.with_suffix('.png')]:
        if output.exists() or output.is_symlink():raise FileExistsError(output)
    if not checkpoint.is_dir():raise FileNotFoundError(checkpoint)
    trainer_state_path=checkpoint/'trainer_state.json'
    trainer_state=json.loads(trainer_state_path.read_text())
    if not reference and trainer_state.get('global_step')!=2:
        raise ValueError('Baseline must be the saved two-step reference checkpoint')
    source_files=['scripts/evaluate_gr00t_pilot.py','scripts/project_config.py','scripts/convert_single_camera.py','scripts/run_gr00t_smoke.py']
    source_hashes={name:digest(PROJECT_ROOT/name) for name in source_files}
    repository_commit=subprocess.check_output(['git','-C',str(PROJECT_ROOT),'rev-parse','HEAD'],text=True).strip()
    reserve=config['reserve_gib']*GIB
    if shutil.disk_usage(PROJECT_ROOT).free < reserve or shutil.disk_usage(existing_ancestor(report_path)).free < reserve or available_memory()<40*GIB:
        raise RuntimeError('Insufficient evaluation disk or host-memory reserve')
    runtime=ensure_python(config,'gr00t')
    os.chdir(prepare_offline_workdir(config,runtime))
    identity=dataset_identity(dataset,pilot_manifest);freeze_samples(manifest_path,identity)
    import numpy as np
    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 16*GIB:
        raise RuntimeError('Insufficient free VRAM on selected CUDA device 0')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();torch.manual_seed(42)
    policy=Gr00tPolicy(embodiment_tag='NEW_EMBODIMENT',model_path=str(checkpoint),device='cuda:0')
    policy.model.eval();policy.processor.eval();policy.model.action_head.num_inference_timesteps=4
    modalities=policy.modality_configs
    keys=[name for name,_,_,_ in GROUPS]
    if modalities['action'].modality_keys!=keys or modalities['state'].modality_keys!=keys or list(modalities['action'].delta_indices)!=list(range(HORIZON)):
        raise ValueError('Checkpoint joint order or horizon differs from pilot')
    processor=policy.processor.state_action_processor;tag=EmbodimentTag.NEW_EMBODIMENT.value
    params=processor.norm_params[tag]['action']
    lower=np.concatenate([np.broadcast_to(params[key]['min'],(HORIZON,end-begin)) for key,begin,end,_ in GROUPS],axis=-1)
    upper=np.concatenate([np.broadcast_to(params[key]['max'],(HORIZON,end-begin)) for key,begin,end,_ in GROUPS],axis=-1)
    def jsonable(value):
        if dataclasses.is_dataclass(value):return jsonable(dataclasses.asdict(value))
        if isinstance(value,Enum):return value.value
        if hasattr(value,'model_dump'):return jsonable(value.model_dump(mode='json'))
        if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
        if hasattr(value,'tolist'):return value.tolist()
        return value
    processor_root=checkpoint if (checkpoint/'processor_config.json').exists() else checkpoint/'processor'
    semantics={'processor_config':json.loads((processor_root/'processor_config.json').read_text()),'modalities':jsonable(modalities),'statistics':jsonable(processor.statistics[tag]),
        'use_relative_action':processor.use_relative_action,'use_percentiles':processor.use_percentiles,
        'clip_outliers':processor.clip_outliers,'apply_sincos_state_encoding':processor.apply_sincos_state_encoding,'lower':lower.tolist(),'upper':upper.tolist()}
    semantics_sha=semantic_digest(semantics)
    loader=LeRobotEpisodeLoader(dataset_path=dataset,modality_configs=modalities)
    predictions=[];targets=[];states=[];instructions=[];cached_episode=None;trajectory=None
    for index,sample in enumerate(identity['samples']):
        if available_memory()<16*GIB or shutil.disk_usage(report_path.parent).free<reserve:
            raise RuntimeError('Evaluation resource reserve crossed')
        if sample['episode_index']!=cached_episode:
            trajectory=loader[sample['episode_index']];cached_episode=sample['episode_index']
        step=extract_step_data(trajectory,sample['start'],modalities,EmbodimentTag.NEW_EMBODIMENT)
        target=np.concatenate([step.actions[key] for key in keys],axis=-1).astype(np.float32)
        state=np.concatenate([step.states[key][-1] for key in keys],axis=-1).astype(np.float32)
        if target.shape!=(HORIZON,26):raise ValueError('Incomplete target horizon')
        obs={'video':{k:np.asarray(v,dtype=np.uint8)[None] for k,v in step.images.items()},
            'state':{k:np.asarray(v,dtype=np.float32)[None] for k,v in step.states.items()},
            'language':{policy.language_key:[[step.text]]}}
        draws=[]
        for seed in SEEDS:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():action,_=policy.get_action(obs)
            joined=np.concatenate([action[key][0] for key in keys],axis=-1)
            if joined.shape!=(HORIZON,26) or not np.isfinite(joined).all():raise ValueError('Invalid model action chunk')
            draws.append(joined.astype(np.float32))
        predictions.append(draws);targets.append(target);states.append(state);instructions.append(step.text)
        if (index+1)%20==0:print(json.dumps({'phase':'evaluation','completed_samples':index+1,'total_samples':100}),flush=True)
    prediction,target,state=map(np.asarray,(predictions,targets,states))
    groups,moving,active=summarize(prediction,target,state,lower,upper)
    reference_report=None;reference_prediction=None;comparison={}
    if reference:
        reference_report,reference_prediction=verify_reference(reference,digest(manifest_path),semantics_sha,target,state,moving)
        reference_groups,_,_=summarize(reference_prediction,target,state,lower,upper)
        for name,_,_,_ in GROUPS:
            comparison[name]={}
            for subset in ['all','commanded_motion','small_commanded_motion']:
                before=reference_groups[name][subset]['model']['mae'];after=groups[name][subset]['model']['mae']
                comparison[name][subset]={'reference_mae':before,'candidate_mae':after,
                    'mae_change':after-before if before is not None else None,
                    'relative_mae_change_percent':100*(after-before)/before if before else None}
    if any(digest(PROJECT_ROOT/name)!=sha for name,sha in source_hashes.items()):
        raise ValueError('Evaluation helper source changed while the evaluation was running')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(report_path.with_suffix('.npz'),absolute_predictions=prediction,targets=target,states=state,
        lower=lower,upper=upper,normalization_active_mask=active,commanded_motion_mask=moving,seeds=np.array(SEEDS),
        episode_indices=np.array([s['episode_index'] for s in identity['samples']]),starts=np.array([s['start'] for s in identity['samples']]))
    plot_trajectories(report_path.with_suffix('.png'),identity['samples'],prediction,target,state,reference_prediction)
    processor_root=checkpoint if (checkpoint/'processor_config.json').exists() else checkpoint/'processor'
    record={'generated_at':datetime.now(timezone.utc).isoformat(),'passed':True,'candidate_checkpoint':str(checkpoint),
        'repository_commit':repository_commit,'script_sha256':source_hashes,'checkpoint_global_step':trainer_state.get('global_step'),
        'checkpoint_trainer_state_sha256':digest(trainer_state_path),
        'checkpoint':str(checkpoint),'reference_evaluation':str(reference) if reference else None,
        'baseline_report_sha256':digest(reference) if reference else None,
        'role':'candidate' if reference else 'two_step_reference','fresh_process':True,'offline':True,'training_set_only':True,
        'dataset':str(dataset),'sample_manifest':str(manifest_path),'sample_manifest_sha256':digest(manifest_path),
        'processor_semantics_sha256':semantics_sha,'processor_semantics':semantics,
        'statistics_sha256':digest(processor_root/'statistics.json'),'sample_count':100,'inferences':300,
        'seeds':SEEDS,'horizon':HORIZON,'denoising_steps':4,'joint_order':keys,'action_shape':list(prediction.shape),
        'absolute_decoded_actions':True,'groups':groups,'comparison':comparison,'instructions':sorted(set(instructions)),
        'commanded_motion_definition':'Per group, max over full target horizon and nonconstant joints of abs(target-current observed state)/(saved action max-min) > 0.05. Fixed from ground truth, independent of predictions.',
        'normalized_error_definition':'2*(decoded absolute prediction-absolute target)/(saved action max-min), without additional clipping. np.isclose(max,min) dimensions excluded only from normalized metrics; retained in native-unit errors.',
        'limitation':'Five-episode training-set fitting comparison only. Passing means complete finite comparable evidence, irrespective of improvement; it does not establish held-out performance, instruction-dependent target selection, or robot success.',
        'elapsed_seconds':time.monotonic()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'artifacts':{path.name:digest(path) for path in [report_path.with_suffix('.npz'),report_path.with_suffix('.png')]}}
    write_new_json(report_path,record)
    print(json.dumps({'passed':True,'report':str(report_path),'groups':groups,'comparison':comparison},indent=2),flush=True)


if __name__=='__main__':main()
