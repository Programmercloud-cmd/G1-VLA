#!/usr/bin/env python3
"""Inspect or recreate the pinned pipeline. Default: plan only; never train."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

from project_config import PROJECT_ROOT, guard_path, load_config, load_runtime, runtime_environment

GIB = 1024 ** 3
STAGES = ('tools', 'sources', 'environments', 'models', 'record')
ROLE_LOCKS = {'unitree':'unitree_lerobot', 'conversion':'lerobot-conversion', 'gr00t':'Isaac-GR00T'}


def read_lock(name):
    return json.loads((PROJECT_ROOT / 'locks' / name).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_existing(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def reserve(config, path, extra=0):
    free = shutil.disk_usage(nearest_existing(path)).free
    if free - extra < config['reserve_gib'] * GIB:
        raise RuntimeError('Insufficient free space at ' + str(path) + '; required reserve would be crossed')
    return free


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    try:
        with temporary.open('x') as out:
            json.dump(data, out, indent=2, default=str)
            out.write('\n')
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def command(args, config, cwd=PROJECT_ROOT, model_credentials=False):
    """No secrets in argv; stop mutating subprocesses before exhausting disk."""
    env = runtime_environment(config, offline=False)
    if model_credentials and os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    reserve(config, config['paths']['envs'])
    process = subprocess.Popen([str(x) for x in args], cwd=cwd, env=env)
    try:
        while process.poll() is None:
            reserve(config, config['paths']['envs'])
            reserve(config, config['paths']['cache'])
            time.sleep(0.5)
        if process.returncode:
            raise RuntimeError('Command failed: ' + shlex.join(str(x) for x in args))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


def installed_versions(python):
    code = "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version.split()[0],'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}}))"
    result = subprocess.run([str(python), '-c', code], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Cannot inspect configured interpreter: ' + str(python))
    return json.loads(result.stdout)


def expected_packages(role):
    result = {}
    for line in (PROJECT_ROOT / 'locks' / (role + '-requirements.txt')).read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            key, value = line.split('==', 1); result[key.lower().replace('_','-')] = value
    return result


def environment_matches(role, python, config=None):
    installed = installed_versions(python)
    expected = expected_packages(role)
    mismatch = {k:{'expected':v,'installed':installed['packages'].get(k)} for k,v in expected.items() if installed['packages'].get(k) != v}
    if config is not None and role != 'tools':
        expected_python = config['python_versions']['gr00t' if role == 'gr00t' else 'conversion']
        if installed['python'] != expected_python:
            mismatch['python'] = {'expected':expected_python,'installed':installed['python']}
        modules = {'gr00t':{'gr00t':config['sources']['gr00t']},
            'conversion':{'lerobot':config['sources']['conversion']},
            'unitree':{'unitree_lerobot':config['sources']['unitree'], 'lerobot':config['sources']['unitree']/'unitree_lerobot/lerobot'}}[role]
        code = 'import importlib.util,json; specs={k:importlib.util.find_spec(k) for k in '+repr(list(modules))+'}; print(json.dumps({k:(v.origin or next(iter(v.submodule_search_locations or []),None)) if v else None for k,v in specs.items()}))'
        found = json.loads(subprocess.check_output([str(python),'-c',code],text=True))
        for module,root in modules.items():
            if not found[module] or not Path(found[module]).resolve().is_relative_to(root.resolve()):
                mismatch['source:'+module] = {'expected':str(root),'installed':found[module]}
    return not mismatch, mismatch, installed


def render_requirements(role, config):
    replacements = {'vendor/unitree_lerobot/unitree_lerobot/lerobot':config['sources']['unitree'] / 'unitree_lerobot/lerobot',
                    'vendor/unitree_lerobot':config['sources']['unitree'], 'vendor/lerobot-conversion':config['sources']['conversion'],
                    'vendor/Isaac-GR00T':config['sources']['gr00t']}
    lines = []
    for line in (PROJECT_ROOT / 'locks' / (role + '-requirements.txt')).read_text().splitlines():
        if line.startswith('-e '):
            relative = line[3:]
            if relative not in replacements:
                raise ValueError('Unexpected editable source in lock')
            line = '-e ' + str(replacements[relative])
        lines.append(line)
    return '\n'.join(lines) + '\n'


def bootstrap_tools(config):
    lock = read_lock('bootstrap-tools.json')['uv']
    uv = config['tools']['uv']
    if uv.exists():
        actual = subprocess.check_output([str(uv), '--version'], text=True).strip()
        if actual.split()[:2] != ['uv',lock['version']]:
            raise RuntimeError('Existing uv differs from pinned version; choose a fresh tools path')
    else:
        reserve(config, config['paths']['tools'], 180 * 1024**2)
        bootstrap = config['paths']['tools'] / 'bootstrap'
        bootstrap.mkdir(parents=True, exist_ok=True)
        wheel = bootstrap / 'uv-pinned.whl'
        if not wheel.exists():
            urllib.request.urlretrieve(lock['url'], wheel)
        if sha256(wheel) != lock['sha256']:
            raise RuntimeError('Bootstrap uv wheel SHA256 mismatch')
        binary = bootstrap / 'uv'
        if not binary.exists():
            with zipfile.ZipFile(wheel) as archive:
                member = next(x for x in archive.namelist() if x.endswith('/scripts/uv') or x.endswith('/bin/uv'))
                with binary.open('xb') as out:
                    out.write(archive.read(member))
            binary.chmod(0o755)
        uv = binary
    python = config['tools']['python']
    if not python.exists():
        target = python.parent.parent
        if target.exists() and any(target.iterdir()):
            raise RuntimeError('Refusing nonempty tools environment without an interpreter')
        command([uv, 'venv', '--python', sys.executable, target], config)
        command([uv, 'pip', 'install', '--python', python, '--no-deps', '-r', PROJECT_ROOT/'locks/tools-requirements.txt'], config)
    else:
        ok, mismatch, _ = environment_matches('tools', python)
        conflicting = {k:v for k,v in mismatch.items() if v['installed'] is not None}
        if conflicting:
            raise RuntimeError('Existing tool versions differ; no resync performed: ' + json.dumps(conflicting))
        if mismatch:
            command([uv,'pip','install','--python',python,'--no-deps',*[k+'=='+v['expected'] for k,v in mismatch.items()]],config)
    return config['tools']['uv']


def sources(config):
    locks = read_lock('runtime-sources.json')
    for role, key in ROLE_LOCKS.items():
        path = config['sources'][role]; record = locks[key]
        if not path.exists():
            command(['git','clone','--filter=blob:none','--no-checkout',record['origin'],path], config)
            command(['git','-C',path,'checkout','--detach',record['commit']], config)
            command(['git','-C',path,'submodule','update','--init','--recursive'], config)
        actual = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual != record['commit']:
            raise RuntimeError('Source revision mismatch; checkout left unchanged: '+str(path))
        state = subprocess.check_output(['git','-C',str(path),'submodule','status','--recursive'],text=True)
        if any(line and line[0] != ' ' for line in state.splitlines()):
            raise RuntimeError('Source has uninitialized or mismatched submodules: '+str(path))
        dirty = subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True)
        if dirty:
            raise RuntimeError('Tracked upstream source modifications detected: '+str(path))
        if role == 'gr00t' and sha256(path/'uv.lock') != record['uv_lock_sha256']:
            raise RuntimeError('Upstream uv.lock differs from recorded lock')


def check_host(config):
    info = dict(line.split('=',1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    if info.get('ID','').strip('"') != 'ubuntu' or info.get('VERSION_ID','').strip('"') != '22.04' or platform.machine() != 'x86_64':
        raise RuntimeError('Validated setup target is Ubuntu 22.04 x86_64; other hosts require separate validation')
    env = runtime_environment(config)
    required = ['git','gcc','make','ffmpeg','ffprobe']
    missing = [x for x in required if not shutil.which(x,path=env['PATH'])]
    if missing:
        raise RuntimeError('Missing host prerequisites: '+','.join(missing)+'; see locks/bootstrap-tools.json prerequisites_command')
    result = subprocess.run(['ffmpeg','-version'],env=env,capture_output=True,text=True)
    if result.returncode or not result.stdout.startswith(('ffmpeg version 4.','ffmpeg version 5.','ffmpeg version 6.','ffmpeg version 7.')):
        raise RuntimeError('FFmpeg 4–7 executable and shared libraries are required; install host ffmpeg dependency closure')


def environments(config):
    check_host(config); sources(config)
    uv = config['tools']['uv']
    if not uv.exists():
        raise RuntimeError('Run tools stage first')
    for role in ('unitree','conversion','gr00t'):
        python = config['interpreters'][role]
        if python.exists():
            ok,mismatch,_ = environment_matches(role,python,config)
            if not ok:
                raise RuntimeError('Existing '+role+' environment differs; no resync performed: '+json.dumps(mismatch))
            continue
        target = python.parent.parent
        if target.exists() and any(target.iterdir()):
            raise RuntimeError('Refusing nonempty environment without interpreter: '+str(target))
        version = config['python_versions']['gr00t' if role == 'gr00t' else 'conversion']
        reserve(config,target,256*1024**2)
        command([uv,'venv','--python',version,target],config)
        if role == 'gr00t':
            env = runtime_environment(config,offline=False); env['UV_PROJECT_ENVIRONMENT']=str(target)
            args = [str(uv),'sync','--project',str(config['sources']['gr00t']),'--frozen','--no-dev','--python',str(python),
                '--no-install-package','tensorrt-cu12','--no-install-package','tensorrt-cu12-bindings','--no-install-package','tensorrt-cu12-libs']
            old = os.environ.get('UV_PROJECT_ENVIRONMENT'); os.environ['UV_PROJECT_ENVIRONMENT']=str(target)
            try: command(args,config)
            finally:
                if old is None: os.environ.pop('UV_PROJECT_ENVIRONMENT',None)
                else: os.environ['UV_PROJECT_ENVIRONMENT']=old
        else:
            config['paths']['cache'].mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w',suffix='.txt',dir=config['paths']['cache'],delete=False) as temp:
                temp.write(render_requirements(role,config)); requirements = Path(temp.name)
            try:
                command([uv,'pip','install','--python',python,'--no-deps','--index','https://download.pytorch.org/whl/cpu',
                    '--index-strategy','unsafe-best-match','-r',requirements],config)
            finally: requirements.unlink()
        ok,mismatch,_ = environment_matches(role,python,config)
        if not ok: raise RuntimeError('Installed environment failed pinned inventory verification: '+str(role))


def model_status(config, hashes=False):
    runtime = load_runtime(config); assets = read_lock('model-assets.json'); missing=[]
    for repo,record in assets.items():
        snapshot = runtime['models'][repo]['snapshot']
        for entry in record['files']:
            path = snapshot / entry['name']
            if not path.exists(): missing.append((repo,entry));continue
            if path.stat().st_size != entry['size'] or (hashes and sha256(path)!=entry['sha256']):
                raise RuntimeError('Pinned model asset mismatch: '+repo+'/'+entry['name'])
    return missing


def download_models(config):
    missing = model_status(config,hashes=True)
    if missing:
        from huggingface_hub import hf_hub_download
        token = os.environ.get('HF_TOKEN')
        credential_file = PROJECT_ROOT / '.env.local'
        if not token and credential_file.exists():
            for line in credential_file.read_text().splitlines():
                if line.strip().removeprefix('export ').startswith('HF_TOKEN=') or line.strip().startswith('HF_TOKEN ='):
                    values = shlex.split(line.split('=',1)[1],comments=True)
                    if len(values)==1: token=values[0]
        if not token: raise RuntimeError('Model download requires HF_TOKEN in local environment or .env.local; never send a token in chat')
        reserve(config,config['paths']['hf_cache'],sum(item['size'] for _,item in missing))
        locks=read_lock('model-assets.json')
        for repo,entry in missing:
            try:
                path=hf_hub_download(repo,entry['name'],revision=locks[repo]['revision'],cache_dir=str(config['paths']['hf_cache']),token=token)
            except Exception as exc:
                raise RuntimeError('Model download failed for '+repo+' ('+type(exc).__name__+'); check account access/token locally') from None
            if sha256(path)!=entry['sha256']: raise RuntimeError('Downloaded model SHA256 mismatch')
    runtime=load_runtime(config)
    for repo,record in runtime['models'].items():
        ref=config['paths']['hf_cache']/('models--'+repo.replace('/','--'))/'refs/main';ref.parent.mkdir(parents=True,exist_ok=True)
        if ref.exists() and ref.read_text().strip()!=record['revision']: raise RuntimeError('Existing cache ref differs; refusing replacement')
        if not ref.exists():ref.write_text(record['revision'])
    print('Pinned model assets verified; no credential recorded')


def machine_record(config):
    record={'generated_at':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'machine':platform.machine(),
        'paths':config['paths'],'sources':{},'environments':{},'tools':config['tools'],'models':load_runtime(config)['models'],
        'host_prerequisites':read_lock('bootstrap-tools.json')['prerequisites_command'],'fresh_host_install_tested':False}
    for role,path in config['sources'].items():
        if path.exists():
            result=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True)
            record['sources'][role]={'path':path,'commit':result.stdout.strip() if result.returncode==0 else None}
    for role,python in config['interpreters'].items():
        if python.exists():record['environments'][role]={'python':python,**installed_versions(python)}
    record['free_bytes']={key:shutil.disk_usage(nearest_existing(config['paths'][key])).free for key in ('cache','envs','runs')}
    atomic_json(guard_path(config['paths']['runtime_record'],config,'reports'),record)
    return record


def plan(config, stages):
    missing=model_status(config)
    return {'mode':'plan','stages':stages,'paths':config['paths'],'interpreters':config['interpreters'],
        'existing_environments':{k:p.exists() for k,p in config['interpreters'].items()},'missing_model_bytes':sum(x['size'] for _,x in missing),
        'reserve_gib':config['reserve_gib'],'prerequisites':read_lock('bootstrap-tools.json'),
        'policy':'Existing pinned environments are verified and reused; conflicting environments are never resynced. No training. Model files reused by hash.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--plan',action='store_true');mode.add_argument('--apply',action='store_true')
    parser.add_argument('--stages',default=','.join(STAGES))
    parser.add_argument('--download-models-internal',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();config=load_config(args.config)
    if args.download_models_internal:
        download_models(config);return
    stages=args.stages.split(',')
    if any(x not in STAGES for x in stages):raise ValueError('Unknown setup stage')
    if not args.apply:
        print(json.dumps(plan(config,stages),indent=2,default=str));return
    for stage in stages:
        print('Setup stage: '+stage,flush=True)
        reserve(config,config['paths']['cache'])
        if stage=='tools':bootstrap_tools(config)
        elif stage=='sources':sources(config)
        elif stage=='environments':environments(config)
        elif stage=='models':
            command([config['tools']['python'],Path(__file__).resolve(),'--config',config['config_path'],'--download-models-internal'] if config['config_path'].exists() else [config['tools']['python'],Path(__file__).resolve(),'--download-models-internal'],config,model_credentials=True)
        elif stage=='record':machine_record(config)
    print('Requested setup stages complete; no training launched')


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
        print('Setup failed: '+str(exc),file=sys.stderr);raise SystemExit(1)
