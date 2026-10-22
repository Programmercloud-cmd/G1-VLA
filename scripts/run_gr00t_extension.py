"""Warm-start 4096 bounded updates from certified cumulative8192 learned weights.

Trainer steps and checkpoint names remain local. Cumulative steps describe weight
lineage only. Full native checkpoints preserve optimizer/scheduler/RNG, but the
pinned iterable loader does not serialize its cursor: exact sample replay is not
supported. Existing experiment sources and evidence are never rewritten.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from dataclasses import asdict
import copy
import hashlib
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import shlex
import signal
import subprocess
import sys
import time

from project_config import (PROJECT_ROOT as ROOT, load_config, load_runtime,
                            resolve_path, guard_path, runtime_environment)
from prepare_pilot import digest
from convert_single_camera import ensure_python
from run_gr00t_smoke import GIB, available_memory, existing_ancestor, prepare_offline_workdir
from run_gr00t_pilot import finite_summary, read_audit, stop_process_group, require_finite, json_finite
from run_gr00t_dataset import (CoverageAudit, PROCESSOR_FILES, SOURCE_FILES, dataset_command,
                               gpu_memory, validate_split, verify_bindings, verify_dataset_inventory)
from archive_gr00t_dataset import (checkpoint_files, prefix_integrity, read_json,
                                  create_checkpoint_inventory,
                                  verify_inventory_records)

BASE_STEP = 8192
STEPS = 4096
ACCUMULATION = 8
CHECKPOINT_STEPS = [2048, 4096]
LEARNING_RATE = 2.5e-5
WARMUP_RATIO = 0.025
SEGMENT_SEED = 42 + BASE_STEP
FULL_STATE_FILES = ('optimizer.pt', 'scheduler.pt', 'rng_state.pth')
SAMPLING_LIMIT = ('Native optimizer/scheduler/RNG state is saved. The pinned iterable '
                  'loader has no state_dict/load_state_dict cursor and reseeds on resume '
                  'with ignore_data_skip=True; exact sample-position replay is not supported. '
                  'This fresh segment starts a new native seed8234 traversal (42+8192), '
                  'not an exact restoration of either previous iterable cursor. Frozen split/evaluation seeds are unchanged.')


# Reuse proven stateless safety helpers without changing their frozen source.
import run_gr00t_continuation as previous
from run_gr00t_continuation import (require, write_new_json, binding, remote_python_bounded,
    bounded_inventory_transport, check_resources, require_watchdog_alive,
    check_armed_watchdog, fsync_checkpoint_directories)


def validate_full_state(path, step, require_cuda=True):
    proof = previous.validate_full_state(path, step, require_cuda=require_cuda)
    proof['sampling_limit'] = SAMPLING_LIMIT
    return proof


def verify_full_source_inventory(path, checkpoint):
    # The source has a 13GB optimizer: signatures + bounded small metadata only.
    import continuation_artifacts as prior
    return prior.full_inventory(path, checkpoint)


def cumulative_step(local_step):
    require(type(local_step) is int and 0 <= local_step <= STEPS, 'Invalid local step')
    return BASE_STEP + local_step


def configure_segment(config):
    """Change only the native data/training RNG seed before upstream set_seed."""
    before = asdict(config)
    require(config.data.seed == 42, 'Unexpected upstream base seed before segment override')
    config.data.seed = SEGMENT_SEED
    after = asdict(config)
    expected = copy.deepcopy(before); expected['data']['seed'] = SEGMENT_SEED
    require(after == expected, 'Segment adapter changed fields other than data.seed')
    frozen = copy.deepcopy(after); frozen['data']['seed'] = 42
    encoded = lambda value: json.dumps(value, sort_keys=True, default=str).encode()
    return {'event':'configuration_override','passed':True,'changed_fields':['data.seed'],
            'base_seed':42,'segment_seed':SEGMENT_SEED,'derivation':'42+source_checkpoint8192',
            'effective_config_sha256':hashlib.sha256(encoded(after)).hexdigest(),
            'config_with_base_seed_sha256':hashlib.sha256(encoded(frozen)).hexdigest(),
            'upstream_config_before_override_sha256':hashlib.sha256(encoded(before)).hexdigest(),
            'split_seed':42,'evaluation_seeds':[42,43,44]}


def extension_command(config, runtime, dataset, output, shards, initial_checkpoint):
    command = dataset_command(config, runtime, dataset, output, shards)
    for flag, value in [('--base-model-path', initial_checkpoint), ('--max-steps', STEPS),
                        ('--save-total-limit', 2), ('--learning-rate', LEARNING_RATE),
                        ('--warmup-ratio', WARMUP_RATIO)]:
        command[command.index(flag) + 1] = str(value)
    command[command.index('--save-only-model')] = '--no-save-only-model'
    return command


def source_prefix(run, report, run_path, report_path):
    """Two native4096 finite segments, never a fabricated native8192 prefix."""
    require(run.get('cumulative_base_step') == 4096 and run.get('cumulative_target_step') == BASE_STEP,
            'Source must have the completed prior4096-to8192 lineage')
    original_run_path = resolve_path(run['source_run_manifest'])
    original_report_path = resolve_path(run['source_train_report'])
    original = read_json(original_run_path); original_report = read_json(original_report_path)
    require(run['episodes'] == original['episodes'], 'Prior segments have different episode ordering or goals')
    require(report.get('passed') is True and report.get('local_optimizer_updates') == 4096
            and report.get('cumulative_lineage_updates') == BASE_STEP, 'Source8192 segment is not complete')
    original_prefix, original_rows = previous.source_prefix(original, original_report)
    parent_prefix, parent_rows = previous.source_prefix(run, report)
    require(run['source_prefix'] == original_prefix, 'Prior4096 certified ancestry changed')
    require(original_prefix['finite_summary']['passed'] and parent_prefix['finite_summary']['passed'],
            'Both native4096 source prefixes must be finite')
    segments = []
    for base, segment_run, segment_report, manifest_path, train_report_path, summary in [
            (0, original, original_report, original_run_path, original_report_path, original_prefix),
            (4096, run, report, resolve_path(run_path), resolve_path(report_path), parent_prefix)]:
        segments.append({'cumulative_base_step':base,'local_updates':4096,'cumulative_end_step':base+4096,
                         'run_manifest':binding(manifest_path),'train_report':binding(train_report_path),
                         'audit':binding(resolve_path(segment_run['audit'])),'finite_prefix':summary})
    previous_rows = original_rows + parent_rows
    counts = Counter((row['episode_index'],row['row']) for row in previous_rows if row['event'] == 'micro_loss')
    require(sum(counts.values()) == BASE_STEP*ACCUMULATION, 'Prior lineage must contain65536 loss-computed examples')
    coverage = CoverageAudit(run['episodes']);coverage.counts = counts
    saved_coverage = read_json(Path(parent_prefix['coverage_path']))
    require(saved_coverage['lineage_coverage'] == coverage.summary(), 'Prior8192 lineage coverage differs from both finite prefixes')
    return {'schema_version':2,'artifact_type':'certified_multisegment_weight_lineage',
            'cumulative_source_updates':BASE_STEP,'native_segments':segments,
            'lineage_coverage':coverage.summary(),'source_unsaved1135_updates_excluded':True,
            'source_run_manifest':binding(resolve_path(run_path)),
            'source_train_report':binding(resolve_path(report_path)),
            'original_run_manifest':binding(original_run_path),
            'original_train_report':binding(original_report_path)}, previous_rows


def bind_source(paths, protocol_path, protocol):
    """Validate selected full8192 source, its view/pair/archive and both finite ancestors."""
    import continuation_artifacts as prior
    from evaluate_gr00t_dataset import load_completed
    from archive_gr00t_dataset import validate_reload_proof
    run = read_json(paths['source_run_manifest']);report = read_json(paths['source_train_report'])
    prior_declaration, _, prior_original = prior.load_protocol(paths['previous_protocol'])
    require(resolve_path(prior_original) == resolve_path(protocol_path), 'Prior original protocol differs')
    prior.verify_ancestry(run,prior_declaration,protocol_path)
    require(run.get('artifact_type') == 'larger_dataset_continuation_run'
            and run.get('run') == 'larger-dataset-v1-continuation-v1-retry1'
            and report.get('run') == run['run'] and report.get('passed') is True
            and report.get('run_manifest_sha256') == digest(paths['source_run_manifest']), 'Expected completed certified8192 source run')
    require(run['processor'] == protocol['processor'], 'Source processor changed')
    raw = read_json(paths['source_inventory']);checkpoint = resolve_path(raw['checkpoint'])
    with bounded_inventory_transport(time.monotonic()+120):
        inventory = verify_full_source_inventory(paths['source_inventory'],checkpoint)
        view_manifest, view_full, view = prior.load_view(paths['source_view_manifest'],paths['source_view_inventory'])
    require(inventory['step'] == 4096 and checkpoint.parent == resolve_path(run['output'])
            and view_manifest['cumulative_step'] == BASE_STEP
            and resolve_path(view_full['checkpoint']) == checkpoint
            and view_full['files'] == inventory['files'], 'Full8192 checkpoint/view identity differs')
    require(prior.verify_binding(view_manifest['training_run']) == resolve_path(paths['source_run_manifest'])
            and prior.verify_binding(view_manifest['training_report']) == resolve_path(paths['source_train_report']),
            'Source view is not bound to actual source training')
    origin_paths = {name:resolve_path(run[name]) for name in ['source_run_manifest','source_train_report',
        'source_inventory','source_archive_report','source_evaluation_train','source_evaluation_validation']}
    previous.bind_source(origin_paths,protocol_path,protocol)  # Original certified4096, excluding unsaved1135.
    prefix, _ = source_prefix(run,report,paths['source_run_manifest'],paths['source_train_report'])
    saved = [x for x in report['checkpoints'] if x['step'] == 4096 and x.get('passed') is True]
    require(len(saved) == 1 and saved[0]['cumulative_step'] == BASE_STEP
            and resolve_path(saved[0]['checkpoint']) == checkpoint
            and saved[0]['coverage']['checkpoint_inventory_sha256'] == digest(paths['source_inventory']),
            'Source8192 checkpoint lacks full save certification')
    require(all(len(report['runtime_audits'][name]) == 1 and report['runtime_audits'][name][0]['passed'] is True
                for name in ['configuration_override','initial_checkpoint_load','runtime','cap_protection','fresh_optimizer']),
            'Source8192 startup audits incomplete')
    comparison = read_json(paths['source_comparison']);selected = read_json(paths['selected_policy'])
    require(comparison.get('passed') is True and comparison.get('local_step') == 4096
            and comparison.get('cumulative_step') == BASE_STEP
            and comparison['continuation_protocol_sha256'] == digest(paths['previous_protocol'])
            and comparison['original_protocol_sha256'] == digest(protocol_path)
            and resolve_path(comparison['canonical_full_resume_reference']) == checkpoint
            and prior.verify_binding(comparison['view_manifest']) == resolve_path(paths['source_view_manifest'])
            and prior.verify_binding(comparison['view_inventory']) == resolve_path(paths['source_view_inventory']),
            'Source comparison ancestry changed')
    require(selected.get('artifact_type') == 'validation_selected_offline_policy'
            and selected.get('cumulative_weight_lineage') == BASE_STEP
            and selected.get('native_checkpoint_global_step') == 4096
            and selected.get('canonical_full_state_resume_ref') == inventory['checkpoint_ref']
            and selected['canonical_full_state_inventory']['sha256'] == digest(paths['source_inventory'])
            and selected['model_only_inference_ref'] == view['checkpoint_ref']
            and selected['model_only_inference_inventory']['sha256'] == digest(paths['source_view_inventory']),
            'Source must be the verified formally selected8192 policy')
    processes = []
    for split in ['train','validation']:
        path = paths['source_evaluation_'+split]
        require(prior.verify_binding(comparison['evaluations'][split]['report']) == resolve_path(path),
                'Source paired comparison report differs')
        evaluation, _ = load_completed(path,protocol,digest(protocol_path),split,False)
        validate_reload_proof(evaluation);processes.append(evaluation['process_id'])
        require(evaluation['checkpoint_global_step'] == 4096
                and resolve_path(evaluation['candidate_checkpoint']) == resolve_path(view['checkpoint'])
                and evaluation['checkpoint_inventory_sha256'] == digest(paths['source_view_inventory'])
                and evaluation['checkpoint_files_sha256'] == {x['path']:x['sha256'] for x in view['files']},
                'Selected8192 paired model-load identity differs')
    require(processes[0] == processes[1] == comparison['same_model_process_id'], 'Source pair did not share one strict model load')
    archive = read_json(paths['source_archive_report'])
    require(archive.get('passed') is True and archive.get('full_state_sha256_and_fsync_verified') is True
            and archive.get('source_preserved') is True and archive.get('weights_or_optimizer_copied') is False
            and archive['canonical_full_checkpoint_ref'] == inventory['checkpoint_ref'], 'Full8192 archive is incomplete')
    archive_manifest = resolve_path(archive['record_directory'])/'archive_manifest.json'
    require(digest(archive_manifest) == archive['manifest_sha256'], 'Full8192 archive manifest changed')
    archived = read_json(archive_manifest)
    require({x['path']:(x['size'],x['sha256']) for x in archived['full_checkpoint_files']} ==
            {x['path']:(x['size'],x['sha256']) for x in inventory['files']}, 'Archive/full inventory differ')
    for name,wanted in protocol['processor']['files_sha256'].items():
        require(digest(checkpoint/name) == wanted, 'Source train-only processor/statistics changed')
    source_bindings = {str(p):digest(p) for p in [*paths.values(),archive_manifest,resolve_path(run['audit'])]}
    for segment in prefix['native_segments']:
        path = resolve_path(segment['finite_prefix']['coverage_path']);source_bindings[str(path)] = digest(path)
    # Preserve inherited original4096 inputs and both actual certified audit sources.
    origin = read_json(resolve_path(run['source_run_manifest']))
    for p in [resolve_path(run['source_run_manifest']),resolve_path(run['source_train_report']),resolve_path(origin['audit'])]:
        source_bindings[str(p)] = digest(p)
    for split in ['train','validation']:
        path=paths['source_evaluation_'+split]
        source_bindings.update({str(path.parent/relative):sha for relative,sha in read_json(path)['artifacts'].items()})
    initial = {'path':str(checkpoint),'source_updates':BASE_STEP,'source_native_step':4096,
        'source_full_state_preserved':True,'optimizer_scheduler_rng_restored':False,
        'files':{x['path']:{'sha256':x['sha256'],'size':x['size'],'signature':x['local_view_signature']} for x in inventory['files']},
        'source_artifacts_sha256':source_bindings,
        'verification_policy':'Certified complete remote SHA/fsync inventory; remote/mounted signatures and bounded small metadata rechecked before/after strict model load. Source optimizer/scheduler/RNG are preserved, not loaded or locally rehashed.'}
    return initial,prefix,run,source_bindings


def validate_loading_info(path, loading_info, initial_checkpoint):
    require(resolve_path(path) == resolve_path(initial_checkpoint), 'Model loader did not use cumulative8192 checkpoint')
    require(not any(loading_info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),
            'Warm-start requires every source tensor to load strictly')
    return {'event':'initial_checkpoint_load','passed':True,'checkpoint':str(resolve_path(path)),
            'source_updates':BASE_STEP,'strict_tensor_loading':True,
            **{k:loading_info.get(k,[]) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}}


def check_runtime(pipeline, manifest):
    config = pipeline.config
    expected = {'max_steps':STEPS,'global_batch_size':1,'gradient_accumulation_steps':ACCUMULATION,
                'save_steps':2048,'save_total_limit':2,'save_only_model':False,
                'learning_rate':LEARNING_RATE,'warmup_ratio':WARMUP_RATIO,'warmup_steps':0,
                'lr_scheduler_type':'cosine','optim':'adamw_torch','num_gpus':1,
                'dataloader_num_workers':0,'remove_unused_columns':False,
                'resume_from_checkpoint':False,'skip_weight_loading':False}
    require(all(getattr(config.training,k) == v for k,v in expected.items()), 'Effective continuation recipe differs')
    require(resolve_path(config.training.start_from_checkpoint) == resolve_path(manifest['initial_checkpoint']['path']),
            'Runtime warm start did not use cumulative8192 checkpoint')
    require(config.data.seed == SEGMENT_SEED and config.data.shard_size == 256
            and config.data.episode_sampling_rate == 1.0 and config.data.override_pretraining_statistics is True
            and config.data.num_shards_per_epoch == manifest['num_shards_per_epoch'], 'Sampling/statistics override differs')
    tuning = {'tune_llm':False,'tune_visual':False,'tune_projector':True,'tune_diffusion_model':True,'tune_vlln':True}
    require(all(getattr(config.model,k) == v for k,v in tuning.items()), 'Tuning flags differ')
    dropout = {'training_config':config.model.state_dropout_prob,'model_config':pipeline.model.config.state_dropout_prob,
               'action_head':pipeline.model.action_head.state_dropout_prob,'processor':pipeline.processor.state_dropout_prob}
    require(all(v == 0.0 for v in dropout.values()), 'State dropout is not zero throughout')
    mixture = pipeline.train_dataset
    require(pipeline.eval_dataset is None and len(mixture.datasets) == 1, 'Only train split may enter training')
    leaf = mixture.datasets[0]
    require(resolve_path(leaf.dataset_path) == resolve_path(manifest['train_dataset'])
            and len(leaf) == manifest['num_shards_per_epoch']
            and list(map(int,leaf.episode_loader.episode_lengths)) == [e['rows'] for e in manifest['episodes']],
            'Runtime train dataset geometry differs')
    processor_dir = Path(manifest['output']) / 'audited-processor'
    pipeline.processor.save_pretrained(processor_dir)
    hashes = {name:digest(processor_dir/name) for name in PROCESSOR_FILES}
    require(hashes == manifest['processor']['files_sha256'], 'Frozen train-only processor/statistics changed')
    return {'event':'runtime','passed':True,'state_dropout_prob':dropout,'processor_files_sha256':hashes,
            'local_initial_step':0,'cumulative_initial_step':BASE_STEP,'resume_from_checkpoint':False,
            'save_only_model':False,'sampling_limit':SAMPLING_LIMIT}


def certify_checkpoint(config, manifest, checkpoint, step, coverage, prior_counts):
    full_state = validate_full_state(checkpoint, step)
    hashes = {name:digest(checkpoint/name) for name in PROCESSOR_FILES}
    require(hashes == manifest['processor']['files_sha256'], 'Saved processor/statistics changed')
    lineage = CoverageAudit(manifest['episodes']); lineage.counts = prior_counts + coverage.counts
    metadata = {'schema_version':1,'artifact_type':'larger_dataset_extension_checkpoint',
                'run':manifest['run'],'local_step':step,'cumulative_step':cumulative_step(step),
                'cumulative_base_step':BASE_STEP,'source_checkpoint':manifest['initial_checkpoint']['path'],
                'source_checkpoint_inventory_sha256':manifest['source_checkpoint_inventory_sha256'],
                'source_prefix':manifest['source_prefix'],'full_state':full_state,'sampling_limit':SAMPLING_LIMIT}
    write_new_json(checkpoint/'continuation.json',metadata)
    result = {'step':step,'local_step':step,'cumulative_step':cumulative_step(step),'checkpoint':str(checkpoint),
              **coverage.summary(),'lineage_coverage':lineage.summary()}
    require(result['consumed_examples'] == step * ACCUMULATION, 'Segment coverage differs from local progress')
    coverage_path = Path(manifest['coverage_prefix'] + f'-{step}.json')
    write_new_json(coverage_path,result)
    inventory_path = Path(manifest['coverage_prefix'] + f'-checkpoint-{step}-inventory.json')
    with bounded_inventory_transport(manifest['wall_time_deadline_monotonic']):
        inventory = create_checkpoint_inventory(config,checkpoint,inventory_path,hash_via_ssh=config['storage']['enabled'])
    fsync_checkpoint_directories(config,checkpoint,manifest['wall_time_deadline_monotonic'])
    require(set(FULL_STATE_FILES).issubset({r['path'] for r in inventory['files']}), 'Durable full state inventory missing')
    return {'event':'checkpoint','step':step,'local_step':step,'cumulative_step':cumulative_step(step),'passed':True,
            'processor_files_sha256':hashes,'coverage_path':str(coverage_path),'coverage_sha256':digest(coverage_path),
            'checkpoint_inventory_path':str(inventory_path),'checkpoint_inventory_sha256':digest(inventory_path),
            'full_state_required':True,'full_state':full_state,'per_file_fsync_completed':True,
            'directories_fsync_completed':True,'continuation_metadata_sha256':digest(checkpoint/'continuation.json')}


def install_audits(config, manifest, stream):
    import torch
    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.model.gr00t_n1d7 import setup
    from gr00t.experiment import experiment
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    coverage = CoverageAudit(manifest['episodes']); counter = 0; started = time.monotonic(); loads = []
    _, previous_rows = source_prefix(read_json(Path(manifest['source_run_manifest'])), read_json(Path(manifest['source_train_report'])),
                                     Path(manifest['source_run_manifest']),Path(manifest['source_train_report']))
    prior_counts = Counter((r['episode_index'],r['row']) for r in previous_rows if r['event'] == 'micro_loss')
    def emit(record):
        record['elapsed_seconds'] = time.monotonic() - started
        stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
    def finite(value,label):
        try: return require_finite(value,label)
        except FloatingPointError:
            emit({'event':'nonfinite','kind':label,'value':repr(float(value))});raise
    original_run = experiment.run
    def configured_run(effective_config):
        emit(configure_segment(effective_config))
        return original_run(effective_config)
    experiment.run = configured_run
    original_load = setup.AutoModel.from_pretrained
    def strict_load(*args,**kwargs):
        require(len(loads) == 0 and kwargs.get('output_loading_info') is True, 'Expected exactly one audited model load')
        require(resolve_path(args[0]) == resolve_path(manifest['initial_checkpoint']['path']), 'Incorrect initialization path')
        model,info = original_load(*args,**kwargs)
        proof = validate_loading_info(args[0],info,manifest['initial_checkpoint']['path'])
        with bounded_inventory_transport(manifest['wall_time_deadline_monotonic']):
            verify_full_source_inventory(Path(manifest['source_checkpoint_inventory']),Path(manifest['initial_checkpoint']['path']))
        loads.append(proof);emit(proof);return model,info
    setup.AutoModel.from_pretrained = staticmethod(strict_load)
    original_shard = ShardedSingleStepDataset.get_shard
    def audited_shard(self,index):
        points = original_shard(self,index)
        coverage.register(points,[(int(ep),int(row)) for ep,rows in self.sharded_episodes[index] for row in rows])
        return points
    ShardedSingleStepDataset.get_shard = audited_shard
    original_setup = setup.Gr00tN1d7Pipeline.setup
    def audited_setup(self):
        original_setup(self)
        require(len(loads) == 1, 'No verified cumulative8192 checkpoint model load')
        emit(check_runtime(self,manifest));original_collator = self.data_collator
        def collator(features):
            result = original_collator(features);coverage.collated(features);return result
        self.data_collator = collator
    setup.Gr00tN1d7Pipeline.setup = audited_setup
    class Audits(TrainerCallback):
        def on_train_begin(self,args,state,control,optimizer=None,lr_scheduler=None,**kwargs):
            emit(check_armed_watchdog(manifest))
            scheduler = getattr(lr_scheduler,'scheduler',lr_scheduler)
            require(state.global_step == 0 and optimizer is not None and not optimizer.state
                    and scheduler is not None and scheduler.last_epoch in (-1,0), 'Optimizer/scheduler must start fresh')
            require(args.save_only_model is False, 'Native optimizer/scheduler/RNG saving disabled')
            emit({'event':'fresh_optimizer','passed':True,'initial_step':0,'local_step':0,
                  'cumulative_step':BASE_STEP,'optimizer_state_entries':0,'scheduler_epoch':scheduler.last_epoch,
                  'source_optimizer_scheduler_rng_restored':False})
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            require(bool(gradients),'No parameter gradients')
            norm = torch.linalg.vector_norm(torch.stack(list(torch._foreach_norm(gradients))).float()).item()
            emit({'event':'gradient','step':state.global_step+1,'local_step':state.global_step+1,
                  'cumulative_step':cumulative_step(state.global_step+1),
                  'post_clip_norm':finite(norm,'post-clip gradient norm'),'finite':True})
        def on_step_end(self,args,state,control,optimizer=None,**kwargs):
            emit({'event':'step_end','step':state.global_step,'local_step':state.global_step,
                  'cumulative_step':cumulative_step(state.global_step),
                  'learning_rate':finite(optimizer.param_groups[0]['lr'],'learning rate')})
        def on_save(self,args,state,control,**kwargs):
            require(state.global_step in CHECKPOINT_STEPS,'Unexpected local checkpoint step')
            # Synchronous: no subsequent training update until native full-state
            # structure, remote hashes and file/directory fsync all pass.
            emit(certify_checkpoint(config,manifest,Path(args.output_dir)/f'checkpoint-{state.global_step}',
                                    state.global_step,coverage,prior_counts))
    original_init,original_loss,original_train = Gr00tTrainer.__init__,Gr00tTrainer.compute_loss,Gr00tTrainer.train
    def audited_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs);self.add_callback(Audits());self.args.logging_nan_inf_filter=False
    def audited_loss(self,*args,**kwargs):
        nonlocal counter
        result = original_loss(self,*args,**kwargs);loss = result[0] if isinstance(result,tuple) else result
        value = finite(loss.detach().float().item(),'microbatch loss');sample = coverage.consume();counter += 1
        emit({'event':'micro_loss','microbatch':counter,'step_before':self.state.global_step,
              'cumulative_step_before':cumulative_step(self.state.global_step),'loss':value,'finite':True,**sample})
        return result
    def audited_train(self,*args,**kwargs):
        callbacks = self.callback_handler.callbacks
        self.callback_handler.callbacks = [c for c in callbacks if not isinstance(c,Audits)]+[c for c in callbacks if isinstance(c,Audits)]
        return original_train(self,*args,**kwargs)
    Gr00tTrainer.__init__,Gr00tTrainer.compute_loss,Gr00tTrainer.train = audited_init,audited_loss,audited_train
    return coverage


def worker(config, path):
    ensure_python(config,'gr00t');manifest = read_json(path)
    require(os.environ.get('PYTHONHASHSEED') == '1','Set PYTHONHASHSEED=1 before launching Python')
    verify_bindings(manifest)
    from evaluate_gr00t_dataset import load_protocol
    protocol = load_protocol(Path(manifest['evaluation_protocol']))
    import extension_artifacts
    declaration, _, original_path = extension_artifacts.load_protocol(Path(manifest['extension_protocol']))
    extension_artifacts.verify_ancestry(manifest,declaration,original_path)
    require(protocol['processor'] == manifest['processor'],'Frozen processor differs')
    for record in protocol['datasets'].values():
        verify_dataset_inventory(Path(record['path']),record['files_sha256'])
    with bounded_inventory_transport(manifest['wall_time_deadline_monotonic']):
        verify_full_source_inventory(Path(manifest['source_checkpoint_inventory']),Path(manifest['initial_checkpoint']['path']))
    runtime = load_runtime(config)
    expected = extension_command(config,runtime,Path(manifest['train_dataset']),Path(manifest['output']),
                                    manifest['num_shards_per_epoch'],Path(manifest['initial_checkpoint']['path']))
    require(expected == manifest['upstream_command'],'Bound continuation command differs')
    os.chdir(prepare_offline_workdir(config,runtime))
    with guard_path(manifest['audit'],config,'reports').open('x',buffering=1) as stream:
        install_audits(config,manifest,stream);sys.argv=expected[1:];runpy.run_path(expected[1],run_name='__main__')


def checkpoint_results(manifest, records):
    result = []
    for step in CHECKPOINT_STEPS:
        checkpoint = Path(manifest['output'])/f'checkpoint-{step}'
        events = [r for r in records if r['event'] == 'checkpoint' and r['step'] == step]
        passed = len(events) == 1 and events[0].get('passed') is True
        if passed:
            checkpoint_files(checkpoint,step)
            require(digest(Path(events[0]['checkpoint_inventory_path'])) == events[0]['checkpoint_inventory_sha256'],
                    'Certified inventory changed')
        result.append({'step':step,'local_step':step,'cumulative_step':cumulative_step(step),
                       'checkpoint':str(checkpoint),'passed':passed,
                       'bytes':sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),
                       'coverage':events[0] if events else None})
    return result


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    for name in ['source-run-manifest','source-train-report','source-inventory','source-archive-report',
                 'source-evaluation-train','source-evaluation-validation','source-comparison','source-view-manifest',
                 'source-view-inventory','selected-policy','previous-protocol','evaluation-protocol','extension-protocol']:
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--run');parser.add_argument('--timeout-seconds',type=int,default=21600)
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args();config=load_config(args.config)
    if args.worker: worker(config,resolve_path(args.worker));return
    required=['source_run_manifest','source_train_report','source_inventory','source_archive_report',
              'source_evaluation_train','source_evaluation_validation','source_comparison','source_view_manifest',
              'source_view_inventory','selected_policy','previous_protocol','evaluation_protocol','extension_protocol','run']
    for name in required:
        if not getattr(args,name):parser.error('--'+name.replace('_','-')+' is required')
    require(re.fullmatch('[A-Za-z0-9_-]+',args.run) is not None and 1 <= args.timeout_seconds <= 21600,
            'Use fresh simple run name and cap within six hours')
    require(os.environ.get('PYTHONHASHSEED') == '1','Set PYTHONHASHSEED=1 before launching Python')
    from evaluate_gr00t_dataset import load_protocol
    protocol_path=resolve_path(args.evaluation_protocol);protocol=load_protocol(protocol_path)
    paths={name:resolve_path(getattr(args,name)) for name in required if name not in ['evaluation_protocol','extension_protocol','run']}
    initial,prefix,source,source_bindings=bind_source(paths,protocol_path,protocol)
    train=read_json(Path(source['train_manifest']));validation=read_json(Path(source['validation_manifest']))
    lengths,shards=validate_split(read_json(Path(source['split_manifest'])),train,validation)
    require(source['episodes'] == [{k:e[k] for k in ['episode','rows','derived_episode_index','source_data_json_sha256','effective_goal']} for e in train['episodes']],
            'Source episode identity changed')
    runtime=load_runtime(config)
    output=guard_path(config['paths']['runs']/args.run,config,'runs')
    require(output.is_relative_to(config['paths']['archive']) and not output.is_relative_to(Path('/dev/shm')),
            'Full checkpoints require durable output storage')
    evidence={suffix:guard_path(config['paths']['reports']/(args.run+suffix),config,'reports')
              for suffix in ['-run.json','-train.json','-train.log','-finite.jsonl']}
    coverage_prefix=str(guard_path(config['paths']['reports']/(args.run+'-coverage'),config,'reports'))
    occupied=[output,*evidence.values(),*[Path(coverage_prefix+f'-{step}.json') for step in CHECKPOINT_STEPS],
              *[Path(coverage_prefix+f'-checkpoint-{step}-inventory.json') for step in CHECKPOINT_STEPS]]
    if any(p.exists() or p.is_symlink() for p in occupied):raise FileExistsError('Continuation outputs already exist')
    import extension_artifacts
    source_hashes={name:digest(ROOT/name) for name in
                   [*SOURCE_FILES,*read_json(paths['previous_protocol'])['new_source_sha256'],*extension_artifacts.NEW_SOURCES]}
    inputs={**source['input_files_sha256'],**source_bindings,str(protocol_path):digest(protocol_path)}
    extension_protocol = resolve_path(args.extension_protocol)
    inputs[str(extension_protocol)] = digest(extension_protocol)
    # Hash all frozen source/data bindings at preflight; large local pretrained
    # files were already verified and are rechecked in the worker, not a dry run.
    pretrained={str(Path(source['pretrained']['path'])/name) for name in source['pretrained']['files_sha256']}
    for path,wanted in inputs.items():
        if path not in pretrained:require(digest(Path(path)) == wanted,'Bound source/data input changed: '+path)
    command=extension_command(config,runtime,Path(source['train_dataset']),output,shards,Path(initial['path']))
    manifest={'schema_version':1,'artifact_type':'larger_dataset_extension_run',
              'created_at':datetime.now(timezone.utc).isoformat(),'run':args.run,
              'repository_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
              **{k:source[k] for k in ['train_dataset','train_manifest','validation_manifest','split_manifest','episodes','pretrained']},
              **{k:str(v) for k,v in paths.items()},'source_checkpoint_inventory':str(paths['source_inventory']),
              'source_checkpoint_inventory_sha256':digest(paths['source_inventory']),
              'source_prefix':prefix,'initial_checkpoint':initial,
              'evaluation_protocol':str(protocol_path),'evaluation_protocol_sha256':digest(protocol_path),
              'extension_protocol':str(extension_protocol),'extension_protocol_sha256':digest(extension_protocol),
              'processor':protocol['processor'],'input_files_sha256':inputs,'script_sha256':source_hashes,
              'output':str(output),'audit':str(evidence['-finite.jsonl']),'coverage_prefix':coverage_prefix,
              'checkpoint_steps':CHECKPOINT_STEPS,'cumulative_checkpoint_steps':[10240,12288],
              'local_to_cumulative':{str(step):cumulative_step(step) for step in CHECKPOINT_STEPS},
              'max_steps':STEPS,'cumulative_base_step':BASE_STEP,'cumulative_target_step':12288,
              'microbatch':1,'accumulation':ACCUMULATION,'state_dropout_prob':0.0,'seed':SEGMENT_SEED,
              'base_seed':42,'frozen_split_seed':42,'evaluation_seeds':[42,43,44],
              'configuration_overrides':{'data.seed':SEGMENT_SEED},
              'fresh_optimizer_and_scheduler':True,'save_only_model':False,
              'source_full_state_preserved':True,'source_optimizer_scheduler_rng_restored':False,
              'cap_watchdog_engine':binding(ROOT/'scripts/watch_training_cap.py'),
              'cap_watchdog_adapter':binding(ROOT/'scripts/watch_extension_cap.py'),
              'learning_rate':LEARNING_RATE,'warmup_ratio':WARMUP_RATIO,'warmup_steps':math.ceil(STEPS*WARMUP_RATIO),
              'lr_scheduler_type':'cosine','sampling_limit':SAMPLING_LIMIT,
              'full_state_files':list(FULL_STATE_FILES),'eligible_training_windows':sum(lengths),
              'num_shards_per_epoch':shards,'timeout_seconds':args.timeout_seconds,'upstream_command':command,
              'wall_time_deadline_monotonic':started+args.timeout_seconds,
              'resource_requirements':{'startup_output_gib':120,'startup_host_available_gib':60,
                                       'startup_gpu_free_mib':40960,'project_reserve_gib':max(8,config['reserve_gib'])},
              'temporary_directory':os.environ.get('TMPDIR'),
              'scope':'Fresh optimizer continuation from certified learned8192 weights; local progress is not fabricated cumulative TrainerState. No automatic resume or robot-success claim.'}
    safety_directory = ROOT/'reports'/(args.run+'-safety')
    manifest['cap_watchdog_report'] = str(safety_directory/'cap-watchdog.json')
    manifest['cap_watchdog_startup_report'] = str(safety_directory/'cap-watchdog-armed.json')
    manifest['cap_watchdog_log'] = str(safety_directory/'cap-watchdog.log')
    require(not safety_directory.exists(),'Local safety outputs already exist')
    require(not config['paths']['reports'].is_relative_to(config['paths']['archive']),
            'Training audit/reports must use local storage independent of the external mount')
    import extension_artifacts
    declaration, _, original_path = extension_artifacts.load_protocol(extension_protocol)
    extension_artifacts.verify_ancestry(manifest,declaration,original_path)
    if args.dry_run:print(json.dumps(manifest,indent=2));return
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch',*source_hashes],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--','scripts','configs'],check=True)
    total,free,_=gpu_memory();reserve=max(8,config['reserve_gib'])
    def resources(startup):
        return check_resources(shutil.disk_usage(ROOT).free,shutil.disk_usage(existing_ancestor(evidence['-run.json'])).free,
                               shutil.disk_usage(existing_ancestor(output)).free,available_memory(),total,free,reserve,startup)
    resources(True);output.parent.mkdir(parents=True,exist_ok=True);evidence['-run.json'].parent.mkdir(parents=True,exist_ok=True)
    write_new_json(evidence['-run.json'],manifest)
    process=None;watchdog=None;watchdog_log=None;aborted=None;failure=None;peak=0;minimum_host=available_memory()
    minimum_output=shutil.disk_usage(output.parent).free;minimum_project=shutil.disk_usage(ROOT).free;last_progress=0
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with evidence['-train.log'].open('x') as log:
            # Match ensure_python's canonical exec argv exactly. Bytecode remains
            # disabled by runtime_environment(PYTHONDONTWRITEBYTECODE=1).
            process=subprocess.Popen([str(runtime['interpreters']['gr00t']),str(Path(__file__).resolve()),
                                      '--config',str(config['config_path']),'--worker',str(evidence['-run.json'])],
                                     cwd=ROOT,env=runtime_environment(config,offline=True),stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            safety_directory.mkdir()
            watchdog_log = Path(manifest['cap_watchdog_log']).open('x')
            watchdog_command = [str(runtime['interpreters']['gr00t']),'-B',str(ROOT/'scripts/watch_extension_cap.py'),
                                '--launcher-pid',str(os.getpid()),'--worker-pid',str(process.pid),
                                '--manifest',str(evidence['-run.json']),'--report',manifest['cap_watchdog_report'],
                                '--run',args.run,'--cap-seconds',str(args.timeout_seconds)]
            if config['storage']['enabled']:
                watchdog_command += ['--sshfs-source',config['storage']['host']+':'+config['storage']['remote_path'],
                                     '--sshfs-mount',str(config['paths']['archive'])]
            watchdog = subprocess.Popen(watchdog_command,cwd=ROOT,stdout=watchdog_log,stderr=subprocess.STDOUT,
                                        start_new_session=True)
            armed_until = time.monotonic()+10
            while not Path(manifest['cap_watchdog_report']).exists() and time.monotonic()<armed_until:
                if watchdog.poll() is not None:raise RuntimeError('Independent local cap watchdog failed to arm')
                time.sleep(.1)
            require(Path(manifest['cap_watchdog_report']).is_file(),'Independent local cap watchdog did not arm')
            while process.poll() is None:
                # Local process check comes before every possible SSHFS stat.
                require_watchdog_alive(watchdog)
                minimum_host=min(minimum_host,available_memory());minimum_output=min(minimum_output,shutil.disk_usage(output.parent).free)
                minimum_project=min(minimum_project,shutil.disk_usage(ROOT).free)
                try:resources(False)
                except RuntimeError as error:aborted='resource_reserve';failure=str(error)
                if time.monotonic()-started > args.timeout_seconds:aborted='wall_time_limit'
                try:peak=max(peak,gpu_memory()[2])
                except (subprocess.CalledProcessError,ValueError):pass
                if aborted:stop_process_group(process);break
                if time.monotonic()-last_progress >= 30:
                    steps=[r['step'] for r in read_audit(evidence['-finite.jsonl']) if r['event']=='step_end']
                    step=steps[-1] if steps else 0
                    print(json.dumps({'phase':'training','local_step':step,'cumulative_step':cumulative_step(step),
                                      'elapsed_seconds':round(time.monotonic()-started,1),'peak_gpu_mib':peak,
                                      'project_free_bytes':shutil.disk_usage(ROOT).free}),flush=True)
                    last_progress=time.monotonic()
                time.sleep(2)
    except KeyboardInterrupt:aborted='interrupted'
    except Exception as error:aborted='execution_error';failure=repr(error)
    finally:
        if process is not None:stop_process_group(process)
        if watchdog_log is not None:watchdog_log.close()
        records=read_audit(evidence['-finite.jsonl']);finite=finite_summary(records,STEPS,ACCUMULATION)
        try:checkpoints=checkpoint_results(manifest,records)
        except Exception as error:
            checkpoints=[{'step':step,'local_step':step,'cumulative_step':cumulative_step(step),
                          'passed':False,'inspection_error':repr(error)} for step in CHECKPOINT_STEPS]
        audits={name:[r for r in records if r['event']==name] for name in
                ['configuration_override','initial_checkpoint_load','runtime','cap_protection','fresh_optimizer']}
        runtime_passed=all(len(rows)==1 and rows[0].get('passed') is True for rows in audits.values())
        report={'schema_version':1,'artifact_type':'larger_dataset_extension_training_report',
                'passed':process is not None and process.returncode==0 and not aborted and finite['passed']
                         and runtime_passed and all(c['passed'] for c in checkpoints),
                'run':args.run,'run_manifest':str(evidence['-run.json']),'run_manifest_sha256':digest(evidence['-run.json']),
                'evaluation_protocol_sha256':digest(protocol_path),'exit_code':process.returncode if process else None,
                'aborted':aborted,'resource_failure_detail':failure,'elapsed_seconds':time.monotonic()-started,
                'sampled_peak_gpu_mib':peak,'min_available_host_gib':minimum_host/GIB,
                'min_output_free_gib':minimum_output/GIB,'min_project_free_bytes':minimum_project,
                'finite_audit':finite,'finite_audit_sha256':digest(evidence['-finite.jsonl']) if evidence['-finite.jsonl'].exists() else None,
                'local_optimizer_updates':finite['optimizer_updates_observed'],
                'cumulative_lineage_updates':cumulative_step(finite['optimizer_updates_observed']),
                'source_prefix':prefix,'runtime_audits':audits,'checkpoints':checkpoints,
                'sampling_limit':SAMPLING_LIMIT,'exact_sample_position_resume_supported':False}
        write_new_json(evidence['-train.json'],json_finite(report));print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
