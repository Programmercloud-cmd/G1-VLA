#!/usr/bin/env python3
"""Check configured source pins, assets and runtime; never optimize a model."""
import argparse
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

from project_config import PROJECT_ROOT, guard_derived, resolve_path, guard_path, load_config, load_runtime, runtime_environment
from setup_pipeline import atomic_json, check_host, environment_matches, model_status, read_lock, reserve, sha256


def resources(config, training=False):
    memory = dict((line.split(':',1)[0], int(line.split()[1])*1024) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:')))
    from setup_pipeline import nearest_existing
    free = shutil.disk_usage(nearest_existing(config['paths']['runs'])).free
    if training and (memory['MemAvailable'] < 60*1024**3 or free < 35*1024**3):
        raise ValueError('Training readiness requires at least 60GiB available RAM and 35GiB free output space')
    return {**memory,'output_free_bytes':free,'output_path':str(config['paths']['runs'])}


def dataset_metadata(path, config):
    dataset = guard_derived(path,config)
    required = ['info.json','modality.json','stats.json','relative_stats.json','episodes.jsonl','tasks.jsonl']
    for name in required:
        if not (dataset/'meta'/name).is_file():raise ValueError('Missing dataset metadata: '+name)
    info=json.loads((dataset/'meta/info.json').read_text()); features=info['features']
    if info.get('codebase_version')!='v2.1' or info.get('total_episodes',0)<1:raise ValueError('Expected nonempty LeRobot v2.1 dataset')
    if features['observation.state']['shape']!=[26] or features['action']['shape']!=[26]:raise ValueError('Expected26-D state/action')
    if len([x for x in features.values() if x.get('dtype')=='video'])!=1:raise ValueError('Expected one video camera')
    modality=json.loads((dataset/'meta/modality.json').read_text())
    expected={'left_arm':(0,7),'right_arm':(7,14),'left_hand':(14,20),'right_hand':(20,26)}
    for group in ['state','action']:
        for name,(start,end) in expected.items():
            entry=modality[group][name]
            if (entry['start'],entry['end'])!=(start,end):raise ValueError('Joint ordering mismatch')
    relative=json.loads((dataset/'meta/relative_stats.json').read_text())
    if not all(k in relative for k in ['left_arm','right_arm']):raise ValueError('Missing relative arm statistics')
    if len(modality.get('video',{}))!=1 or not modality.get('annotation'):raise ValueError('Camera/language modality metadata missing')
    return {'dataset':str(dataset),'episodes':info['total_episodes'],'frames':info['total_frames'],'state_action_dim':26,'cameras':1}


def validate_gpu_capacity(record, training=False):
    """Validate CUDA device 0, after CUDA_VISIBLE_DEVICES remapping."""
    if training and (record['total_bytes'] < 40*1024**3 or record['free_bytes'] < 40*1024**3):
        raise ValueError('Training readiness requires selected CUDA device 0 to have at least 40GiB total and currently free VRAM')
    return record


def verify(config, level='structure', hashes=False, profile='setup', dataset=None):
    checks=[]
    def check(name, function):
        try:
            detail=function()
            checks.append({'name':name,'passed':True,'detail':detail})
        except Exception as exc:
            checks.append({'name':name,'passed':False,'error':str(exc),'error_type':type(exc).__name__})
    check('supported_host_and_ffmpeg',lambda:(check_host(config),'Ubuntu22.04x86_64; FFmpeg4–7 executable/shared runtime')[1])
    check('disk_reserve',lambda:{key:reserve(config,config['paths'][key]) for key in ('envs','cache')})
    check('host_memory_and_output_capacity',lambda:resources(config,training=profile=='training'))
    if profile=='training' and dataset is None:
        checks.append({'name':'dataset','passed':False,'error':'--profile training requires --dataset'})
    elif dataset is not None:
        check('dataset_metadata',lambda:dataset_metadata(dataset,config))
    locks=read_lock('runtime-sources.json')
    roles={'unitree':'unitree_lerobot','conversion':'lerobot-conversion','gr00t':'Isaac-GR00T'}
    for role,key in roles.items():
        def source_check(role=role,key=key):
            path=config['sources'][role]
            result=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True,check=True)
            if result.stdout.strip()!=locks[key]['commit']:raise ValueError('Source commit differs')
            state=subprocess.check_output(['git','-C',str(path),'submodule','status','--recursive'],text=True)
            if any(line and line[0]!=' ' for line in state.splitlines()):raise ValueError('Submodule revision/init mismatch')
            dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True)
            if dirty:raise ValueError('Tracked upstream source modifications detected')
            if role=='gr00t' and sha256(path/'uv.lock')!=locks[key]['uv_lock_sha256']:raise ValueError('Upstream uv.lock SHA256 mismatch')
            return {'path':str(path),'commit':result.stdout.strip()}
        check('source_'+role,source_check)
    for role,python in {**config['interpreters'],'tools':config['tools']['python']}.items():
        def env_check(role=role,python=python):
            ok,mismatch,installed=environment_matches(role,python,config)
            if not ok:raise ValueError('Pinned package mismatch: '+json.dumps(mismatch))
            return {'python':str(python),'python_version':installed['python'],'packages':len(installed['packages'])}
        check('environment_'+role,env_check)
    def assets():
        missing=model_status(config,hashes=hashes)
        if missing:raise ValueError('Missing '+str(len(missing))+' required model files')
        for repo,record in load_runtime(config)['models'].items():
            ref=config['paths']['hf_cache']/('models--'+repo.replace('/','--'))/'refs/main'
            if not ref.exists() or ref.read_text().strip()!=record['revision']:raise ValueError('Pinned offline ref missing or different')
        return {'all_required_files_present':True,'sha256_checked':hashes}
    check('model_assets',assets)
    env=runtime_environment(config)
    def run_python(role,code):
        result=subprocess.run([str(config['interpreters'][role]),'-c',code],env=env,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr[-3500:])
        return result.stdout.strip()
    if level in ('imports','gpu'):
        for role in ('unitree','conversion','gr00t'):
            def imports(role=role):
                if role=='gr00t':
                    code='import torch,torchvision,torchcodec,flash_attn,gr00t; print(torch.__version__,torchvision.__version__,torchcodec.__version__,flash_attn.__version__)'
                else:
                    code='import torch,torchvision,torchcodec,lerobot; print(torch.__version__,torchvision.__version__,torchcodec.__version__,lerobot.__file__); assert torch.version.cuda is None'
                return run_python(role,code)
            check('imports_'+role,imports)
        def video():
            config['paths']['cache'].mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='vla-readiness-',dir=config['paths']['cache']) as folder:
                target=Path(folder)/'test.mp4'
                subprocess.run(['ffmpeg','-v','error','-nostdin','-f','lavfi','-i','testsrc2=size=64x48:rate=2','-frames:v','2','-c:v','libx264','-pix_fmt','yuv420p',str(target)],env=env,check=True,capture_output=True)
                return run_python('gr00t','from torchcodec.decoders import VideoDecoder; d=VideoDecoder('+repr(str(target))+'); assert len(d)==2; f=d[0]; assert tuple(f.shape)==(3,48,64); print("TorchCodec H264 decode passed")')
        check('encoded_video_decode',video)
    if level=='gpu' or profile=='training':
        def gpu_capacity():
            code = 'import json,os,torch; assert torch.cuda.is_available(), "No CUDA-visible GPU"; free,total=torch.cuda.mem_get_info(0); print(json.dumps({"device_index":0,"name":torch.cuda.get_device_name(0),"total_bytes":total,"free_bytes":free,"visible_device_count":torch.cuda.device_count(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}))'
            return validate_gpu_capacity(json.loads(run_python('gr00t',code)),training=profile=='training')
        check('gpu_capacity',gpu_capacity)
    if level in ('imports','gpu') and dataset is not None:
        code = 'import sys,numpy as np; sys.path.insert(0,'+repr(str(PROJECT_ROOT/'configs'))+'); from g1_inspire_config import g1_inspire_config; from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader; from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data; from gr00t.data.embodiment_tags import EmbodimentTag; loader=LeRobotEpisodeLoader('+repr(str(resolve_path(dataset)))+',g1_inspire_config); step=extract_step_data(loader[0],0,g1_inspire_config,EmbodimentTag.NEW_EMBODIMENT); action=np.concatenate([step.actions[k] for k in ["left_arm","right_arm","left_hand","right_hand"]],axis=-1); assert action.shape==(16,26) and np.isfinite(action).all(); print("Actual GR00T loader16x26 sample passed")'
        check('actual_dataset_loader',lambda:run_python('gr00t',code))
    if level=='gpu':
        code='''import json,torch
from flash_attn import flash_attn_func
assert torch.cuda.is_available()
a=torch.randn(256,256,device="cuda",requires_grad=True);(a@a.T).square().mean().backward()
q=torch.randn(1,32,4,64,device="cuda",dtype=torch.bfloat16,requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
y=flash_attn_func(q,k,v);y.float().square().mean().backward();torch.cuda.synchronize()
assert all(torch.isfinite(t).all() for t in [a.grad,y,q.grad,k.grad,v.grad])
print(json.dumps({"gpu":torch.cuda.get_device_name(),"cuda":torch.version.cuda,"matmul_backward":True,"flash_attention_backward":True}))'''
        check('gpu_kernels',lambda:run_python('gr00t',code))
    return {'generated_at':datetime.now(timezone.utc).isoformat(),'level':level,'profile':profile,'passed':all(c['passed'] for c in checks),
        'checks':checks,'scope':'Readiness and optional tiny kernels only; no training or full checkpoint inference.',
        'clean_host_install_proven':False,'config_path':str(config['config_path'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--level',choices=['structure','imports','gpu'],default='structure')
    parser.add_argument('--hash-models',action='store_true')
    parser.add_argument('--profile',choices=['setup','training'],default='setup')
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();config=load_config(args.config)
    report=verify(config,args.level,args.hash_models,args.profile,args.dataset)
    path=guard_path(args.report or config['paths']['reports']/'machine-readiness.json',config,'reports')
    atomic_json(path,report)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    main()
