"""Publish evidence for a larger-run checkpoint already on durable storage, in place."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import uuid

from project_config import PROJECT_ROOT as ROOT, load_config, resolve_path, guard_path, require_storage
from prepare_pilot import digest
from run_gr00t_dataset import CHECKPOINT_STEPS, PROCESSOR_FILES, ACCUMULATION
from run_gr00t_pilot import read_audit, finite_summary
from archive_smoke_checkpoint import remote_python, repository_provenance


def read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError('Expected ordinary metadata JSON of at most16MiB: ' + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise ValueError('Metadata must be a JSON object')
    return value


def checkpoint_files(source, step):
    if source.is_symlink() or not source.is_dir() or source.name != f'checkpoint-{step}':
        raise ValueError('Expected the declared complete checkpoint directory')
    paths = sorted(source.rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('Checkpoint symlinks are not accepted')
    files = {p.relative_to(source).as_posix(): p for p in paths if p.is_file()}
    required = {'config.json', 'trainer_state.json', 'model.safetensors.index.json', *PROCESSOR_FILES}
    if not required.issubset(files): raise ValueError('Checkpoint metadata is incomplete')
    if read_json(source/'trainer_state.json').get('global_step') != step:
        raise ValueError('Saved optimizer step differs from checkpoint name')
    index = read_json(source/'model.safetensors.index.json')
    shards = set(index.get('weight_map', {}).values())
    actual = {name for name in files if name.endswith('.safetensors')}
    if not shards or shards != actual or any(Path(name).name != name for name in shards):
        raise ValueError('Checkpoint weight index/shard inventory differs')
    return files


def prefix_integrity(records, step):
    events = [(i,r) for i,r in enumerate(records) if r['event'] == 'checkpoint' and r['step'] == step]
    if len(events)!=1 or events[0][1].get('passed') is not True:
        raise ValueError('Checkpoint lacks a complete finite training prefix and save audit')
    result = finite_summary(records[:events[0][0]+1], step, ACCUMULATION)
    if not result['passed']:
        raise ValueError('Checkpoint lacks a complete finite training prefix and save audit')
    return result, events[0][1]


def validate_reload_proof(report):
    proof=report.get('reload_proof',{})
    if (report.get('fresh_process') is not True or proof.get('strict_tensor_loading') is not True
            or proof.get('model_load_count_in_process')!=1 or proof.get('finite_predictions') is not True
            or proof.get('single_prediction_shape')!=[16,26]
            or proof.get('saved_prediction_shape')!=[100,3,16,26]
            or any(proof.get(key)!=[] for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])):
        raise ValueError('Paired evaluation must prove a strict fresh-process reload and finite action chunks')


def collect_evidence(source, run_path, training_path, reload_path, train_eval, validation_eval, protocol_path):
    from evaluate_gr00t_dataset import load_protocol
    run = read_json(run_path); training = read_json(training_path); reload = read_json(reload_path)
    protocol = load_protocol(protocol_path)
    match = re.fullmatch(r'checkpoint-(\d+)', source.name)
    step = int(match.group(1)) if match else 0
    if (step not in CHECKPOINT_STEPS or run.get('artifact_type') != 'larger_dataset_training_run'
            or run.get('max_steps') != 8192 or run.get('checkpoint_steps') != CHECKPOINT_STEPS
            or source.parent != resolve_path(run['output'])):
        raise ValueError('Only declared larger-dataset checkpoints can use in-place archival')
    files = checkpoint_files(source, step)
    if (reload.get('passed') is not True or reload.get('fresh_process') is not True
            or resolve_path(reload.get('checkpoint', '')) != source):
        raise ValueError('Matching successful fresh-process reload is required')
    validate_reload_proof(reload)
    if (training.get('artifact_type') != 'larger_dataset_training_report'
            or training.get('run_manifest_sha256') != digest(run_path)
            or run.get('evaluation_protocol_sha256') != digest(protocol_path)
            or training.get('evaluation_protocol_sha256') != digest(protocol_path)):
        raise ValueError('Training/run/protocol provenance differs')
    saved = [x for x in training['checkpoints'] if x['step'] == step]
    if len(saved) != 1 or saved[0].get('passed') is not True or resolve_path(saved[0]['checkpoint']) != source:
        raise ValueError('Training report does not certify this complete checkpoint')
    audit = resolve_path(run['audit'])
    if audit.is_symlink() or audit.stat().st_size > 64 * 1024**2 or digest(audit) != training['finite_audit_sha256']:
        raise ValueError('Finite audit is missing, changed, or exceeds64MiB')
    prefix, event = prefix_integrity(read_audit(audit), step)
    coverage = resolve_path(event['coverage_path'])
    if digest(coverage) != event['coverage_sha256'] or read_json(coverage)['consumed_examples'] != step*ACCUMULATION:
        raise ValueError('Consumed-sample coverage binding differs')
    if any(digest(source/name) != protocol['processor']['files_sha256'][name] for name in PROCESSOR_FILES):
        raise ValueError('Checkpoint processor/statistics differ from frozen training normalization')
    sources = [(run_path, 'reports/run.json'), (training_path, 'reports/training.json'),
               (reload_path, 'reports/reload.json'), (protocol_path, 'evaluation/protocol.json'),
               (audit, 'reports/finite.jsonl'), (coverage, 'reports/coverage.json')]
    process_ids=[]
    for role, report_path in [('train',train_eval), ('validation',validation_eval)]:
        report = read_json(report_path)
        validate_reload_proof(report);process_ids.append(report.get('process_id'))
        if (report.get('passed') is not True or report.get('split') != role
                or report.get('checkpoint_global_step') != step
                or resolve_path(report.get('candidate_checkpoint', '')) != source
                or report.get('evaluation_protocol_sha256') != digest(protocol_path)
                or report.get('processor_semantics_sha256') != protocol['processor']['semantics_sha256']
                or report.get('statistics_sha256') != protocol['processor']['files_sha256']['statistics.json']
                or report.get('sample_manifest_sha256') != protocol['samples'][role]['sha256']):
            raise ValueError('Evaluation does not match this checkpoint/protocol: ' + role)
        sources.append((report_path, 'evaluation/' + role + '.json'))
        for suffix in ['.npz', '.png']:
            path = report_path.with_suffix(suffix)
            if suffix == '.npz' or path.exists():
                if report.get('artifacts', {}).get(path.name) != digest(path):
                    raise ValueError('Changed evaluation companion: ' + str(path))
                sources.append((path, 'evaluation/' + role + suffix))
        references = []
        for path, expected in run['input_files_sha256'].items():
            if expected == report.get('baseline_report_sha256') and Path(path).suffix == '.json':
                references.append(Path(path))
        if len(references) != 1: raise ValueError('Evaluation baseline is not uniquely bound by run')
        baseline = references[0]; baseline_data = read_json(baseline); arrays = baseline.with_suffix('.npz')
        if (digest(baseline) != report['baseline_report_sha256'] or baseline_data.get('role') != 'step_zero_reference'
                or baseline_data.get('split') != role or baseline_data.get('passed') is not True
                or baseline_data.get('evaluation_protocol_sha256') != digest(protocol_path)
                or baseline_data.get('artifacts', {}).get(arrays.name) != digest(arrays)):
            raise ValueError('Step-zero reference evidence changed')
        sources.extend([(baseline, 'evaluation/' + role + '-step-zero.json'),
                        (arrays, 'evaluation/' + role + '-step-zero.npz')])
        sample = protocol['samples'][role]
        sources.extend([(resolve_path(sample['path']), 'evaluation/' + role + '-samples.json'),
                        (resolve_path(sample['arrays']['path']), 'evaluation/' + role + '-samples.npz')])
    if process_ids[0]!=process_ids[1] or not isinstance(process_ids[0],int) or process_ids[0]<=0:
        raise ValueError('Training and validation evaluations must share one fresh paired process')
    for name, record in protocol['bindings'].items():
        sources.append((resolve_path(record['path']), 'dataset/' + name + '.json'))
    for name in PROCESSOR_FILES:
        sources.append((resolve_path(protocol['processor']['path'])/name, 'evaluation/processor/' + name))
    if protocol['processor'].get('semantics_file'):
        sources.append((resolve_path(protocol['processor']['semantics_file']['path']), 'evaluation/processor/semantics.json'))
    for relative, expected in {**run['script_sha256'], **protocol['source_sha256']}.items():
        path = ROOT/relative
        if digest(path) != expected: raise ValueError('Executed source changed before archival: ' + relative)
        sources.append((path, relative))
    sources.append((Path(__file__).resolve(), 'scripts/archive_gr00t_dataset.py'))
    for path in sorted((ROOT/'locks').iterdir()):
        if path.is_file() and path.suffix in ['.json','.txt']: sources.append((path, 'locks/' + path.name))
    unique = {}
    for path, relative in sources:
        if relative in unique and unique[relative] != path: raise ValueError('Evidence destination collision')
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64*1024**2:
            raise ValueError('Evidence must be ordinary bounded files: ' + str(path))
        unique[relative] = path
    return files, [(p,n) for n,p in unique.items()], {'step':step,'finite_prefix':prefix,
            'whole_run_completed':training.get('passed') is True,'training_aborted':training.get('aborted'),
            'persistence_is_descriptive':True,'evaluation_integrity_passed':True}


def hash_checkpoint_in_place(source):
    """Hash/fsync without copying or rewriting any checkpoint content."""
    records=[]
    paths=sorted(p for p in source.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in source.rglob('*')): raise ValueError('Symlink in checkpoint')
    before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    for path in paths:
        sha=digest(path)
        with path.open('rb') as stream: os.fsync(stream.fileno())
        if (path.stat().st_size,path.stat().st_mtime_ns)!=before[path]: raise ValueError('Checkpoint changed during hashing')
        stat=path.stat()
        records.append({'path':path.relative_to(source).as_posix(),'size':before[path][0],
                        'mtime_ns':before[path][1],'device':stat.st_dev,'inode':stat.st_ino,'sha256':sha})
    if {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()} != {r['path'] for r in records}:
        raise ValueError('Checkpoint inventory changed during hashing')
    return records


REMOTE_HASH = r'''from pathlib import Path
import hashlib,json,os,sys
root=Path(sys.argv[1]);assert root.is_dir() and not root.is_symlink()
paths=sorted(root.rglob('*'));assert not any(p.is_symlink() for p in paths)
files=[p for p in paths if p.is_file()];records=[]
for p in files:
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
  os.fsync(stream.fileno())
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Checkpoint changed'
 records.append({'path':p.relative_to(root).as_posix(),'size':after.st_size,'mtime_ns':after.st_mtime_ns,'device':after.st_dev,'inode':after.st_ino,'sha256':h.hexdigest()})
assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}=={r['path'] for r in records}
print(json.dumps(records))
'''

REMOTE_STAT = r'''from pathlib import Path
import json,sys
root=Path(sys.argv[1]);paths=sorted(root.rglob('*'))
assert root.is_dir() and not root.is_symlink() and not any(p.is_symlink() for p in paths)
result=[]
for p in paths:
 if p.is_file():
  s=p.stat();result.append({'path':p.relative_to(root).as_posix(),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino})
print(json.dumps(result))
'''


def local_signature(path):
    stat=path.stat()
    return {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'device':stat.st_dev,'inode':stat.st_ino}


def bind_local_view(files, records):
    """Retain both full remote stat and the potentially second-granularity SSHFS view."""
    if len(records)!=len(files) or {r['path'] for r in records}!=set(files):
        raise ValueError('Remote and mounted checkpoint inventories differ')
    for record in records:
        signature=local_signature(files[record['path']])
        if (signature['size']!=record['size']
                or signature['mtime_ns']//10**9!=record['mtime_ns']//10**9):
            raise ValueError('Mounted file does not match the remote size/time view')
        record['local_view_signature']=signature
    return records


def verify_remote_signatures(identity, records):
    host=identity['host'];source=PurePosixPath(identity['path'])
    if (not re.fullmatch(r'[A-Za-z0-9_.@:\[\]-]+',host) or host.startswith('-')
            or not source.is_absolute() or '..' in source.parts):
        raise ValueError('Invalid remote checkpoint identity')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',host]
    actual=json.loads(remote_python(ssh,REMOTE_STAT,source).stdout)
    expected=[{key:r[key] for key in ['path','size','mtime_ns','device','inode']} for r in records]
    if actual!=expected: raise ValueError('Remote checkpoint signature or inventory changed')


def verify_inventory_records(files, records):
    if len(records)!=len(files) or {r['path'] for r in records}!=set(files):
        raise ValueError('Durable checkpoint inventory differs')
    for record in records:
        path=files[record['path']]
        expected=record['local_view_signature'] if 'local_view_signature' in record else {
            key:record[key] for key in ['size','mtime_ns','device','inode']}
        if local_signature(path)!=expected:
            raise ValueError('Durable checkpoint signature changed: '+record['path'])
        if not re.fullmatch('[0-9a-f]{64}',record['sha256']): raise ValueError('Invalid checkpoint digest')


def load_checkpoint_inventory(path, checkpoint):
    """Verify a bound complete inventory using signatures, without rereading weights.

    Call before and after model loading/inference. The inventory is initially
    created by hash+fsync of the immutable durable checkpoint on its storage host.
    """
    value=read_json(Path(path));checkpoint=resolve_path(checkpoint)
    if (value.get('schema_version')!=1 or value.get('artifact_type')!='durable_checkpoint_inventory'
            or value.get('passed') is not True or value.get('per_file_fsync_completed') is not True
            or resolve_path(value.get('checkpoint',''))!=checkpoint
            or value.get('step') not in CHECKPOINT_STEPS):
        raise ValueError('Checkpoint inventory identity differs')
    files=checkpoint_files(checkpoint,value['step'])
    if value.get('remote_identity'): verify_remote_signatures(value['remote_identity'],value['files'])
    verify_inventory_records(files,value['files'])
    # Bind small checkpoint metadata by content as well as size/mtime; only the
    # large immutable shards use the freshly established inventory signatures.
    for record in value['files']:
        if not record['path'].endswith('.safetensors') and digest(files[record['path']])!=record['sha256']:
            raise ValueError('Checkpoint metadata digest changed')
    return value


def create_checkpoint_inventory(config, source, path, hash_via_ssh=False):
    archive=config['paths']['archive'];source=guard_path(source,config,'runs')
    path=guard_path(path,config,'reports')
    if not source.is_relative_to(archive) or source.is_relative_to(Path('/dev/shm')):
        raise ValueError('Inventory requires the new run checkpoint on durable archive storage')
    if config['storage']['enabled']: require_storage(config)
    match=re.fullmatch(r'checkpoint-(\d+)',source.name)
    step=int(match.group(1)) if match else 0
    if step not in CHECKPOINT_STEPS: raise ValueError('Not a declared larger-dataset checkpoint')
    files=checkpoint_files(source,step)
    if path.exists() or path.is_symlink(): raise FileExistsError(path)
    remote_identity=None
    if hash_via_ssh:
        host=config['storage']['host'];remote=config['storage']['remote_path']
        if (not config['storage']['enabled'] or not host or not remote or host.startswith('-')
                or not re.fullmatch(r'[A-Za-z0-9_.@:\[\]-]+',host)):
            raise ValueError('Configured SSHFS storage is required')
        remote_source=PurePosixPath(remote)/source.relative_to(archive).as_posix()
        ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',host]
        records=json.loads(remote_python(ssh,REMOTE_HASH,remote_source).stdout)
        remote_identity={'host':host,'path':str(remote_source)}
        bind_local_view(files,records)
        verify_remote_signatures(remote_identity,records)
    else:
        records=hash_checkpoint_in_place(source)
    verify_inventory_records(files,records)
    value={'schema_version':1,'artifact_type':'durable_checkpoint_inventory','passed':True,
           'created_at_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':str(source),'step':step,
           'checkpoint_ref':'archive:'+source.relative_to(archive).as_posix(),'files':records,
           'remote_identity':remote_identity,
           'per_file_fsync_completed':True,'hash_location':'configured_storage_host' if hash_via_ssh else 'local',
           'verification_policy':'Full SHA256/fsync once; retain full remote signatures and separately bound mounted-view signatures. Before/after consumers, verify complete remote and local inventories/signatures plus small metadata hashes. Keep checkpoint immutable.'}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--source',type=Path,required=True)
    for name in ['run-config','train-report','reload-report','train-evaluation','validation-evaluation','protocol']:
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--record-name')
    parser.add_argument('--inventory-output',type=Path,help='Only create a fresh hash/fsync inventory for evaluation and subsequent archive publication')
    parser.add_argument('--checkpoint-inventory',type=Path,help='Reuse the verified immutable checkpoint inventory from evaluation; no weight reread')
    parser.add_argument('--hash-via-ssh',action='store_true',help='Hash/fsync on the configured SSHFS host, avoiding network reads of weights')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();config=load_config(args.config)
    if args.inventory_output:
        if args.dry_run: raise ValueError('Inventory output hashes/fsyncs the checkpoint; omit --dry-run')
        result=create_checkpoint_inventory(config,resolve_path(args.source),args.inventory_output,args.hash_via_ssh)
        print(json.dumps({'inventory':str(resolve_path(args.inventory_output)),'files':len(result['files']),'passed':True}));return
    for name in ['record_name','run_config','train_report','train_evaluation','validation_evaluation','protocol','checkpoint_inventory']:
        if not getattr(args,name): parser.error('--'+name.replace('_','-')+' is required for archive publication')
    # A paired evaluation starts a fresh process and proves reload plus finite
    # actions. Reuse that bound proof instead of loading12GB a second time.
    if args.reload_report is None: args.reload_report=args.train_evaluation
    if not re.fullmatch('[A-Za-z0-9_-]+',args.record_name): raise ValueError('Use a simple new record name')
    source=guard_path(args.source,config,'runs');archive=config['paths']['archive']
    if not source.is_relative_to(archive) or source.is_relative_to(Path('/dev/shm')):
        raise ValueError('In-place archival requires a checkpoint already within configured durable archive storage')
    if config['storage']['enabled']: require_storage(config)
    files,sources,checks=collect_evidence(source,*(resolve_path(x) for x in
        [args.run_config,args.train_report,args.reload_report,args.train_evaluation,args.validation_evaluation,args.protocol]))
    final=guard_path(archive/'checkpoints'/args.record_name,config,'archive')
    report_path=guard_path(config['paths']['reports']/(args.record_name+'-archive.json'),config,'reports')
    if final.exists() or final.is_symlink() or report_path.exists(): raise FileExistsError('Archive record already exists')
    inventory_path=resolve_path(args.checkpoint_inventory)
    inventory=load_checkpoint_inventory(inventory_path,source);records=inventory['files']
    for report_path_to_check in [args.train_evaluation,args.validation_evaluation]:
        if read_json(resolve_path(report_path_to_check)).get('checkpoint_inventory_sha256')!=digest(inventory_path):
            raise ValueError('Evaluation must bind the exact checkpoint hash inventory')
    sources.append((inventory_path,'reports/checkpoint-inventory.json'))
    if args.dry_run:
        print(json.dumps({'checkpoint':str(source),'record_directory':str(final),'checkpoint_files':len(files),
                          'evidence_files':len(sources),'weights_copied':False,'checks':checks},indent=2));return
    verify_inventory_records(files,records)
    stage=final.with_name('.'+final.name+'.incomplete-'+uuid.uuid4().hex);stage.mkdir(parents=True)
    evidence=[]
    for path,relative in sources:
        before=(path.stat().st_size,path.stat().st_mtime_ns);sha=digest(path)
        target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with path.open('rb') as incoming,target.open('xb') as outgoing:
            shutil.copyfileobj(incoming,outgoing,1024**2);outgoing.flush();os.fsync(outgoing.fileno())
        if digest(target)!=sha or (path.stat().st_size,path.stat().st_mtime_ns)!=before:
            raise ValueError('Evidence changed during publication: '+relative)
        evidence.append({'path':relative,'size':before[0],'sha256':sha})
    verify_inventory_records(files,records)
    load_checkpoint_inventory(inventory_path,source)
    manifest={'schema_version':1,'artifact_type':'durable_in_place_checkpoint_archive',
              'created_at_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_ref':'archive:'+source.relative_to(archive).as_posix(),
              'checkpoint_files':records,'evidence_files':evidence,'checks':checks,'repository':repository_provenance(),
              'checkpoint_bytes':sum(r['size'] for r in records),'checkpoint_inventory_sha256':digest(inventory_path),
              'weights_copied':False,'source_preserved':True,
              'retention':'The referenced durable run checkpoint IS the archived weight artifact. Keep it immutable; do not delete it after publishing this evidence record.'}
    manifest_path=stage/'archive_manifest.json'
    with manifest_path.open('x') as stream:
        json.dump(manifest,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    report={'passed':True,'mode':'durable_in_place','checkpoint_ref':manifest['checkpoint_ref'],
            'record_directory':str(final),'manifest_sha256':digest(manifest_path),'checkpoint_file_count':len(records),
            'evidence_file_count':len(evidence),'checkpoint_sha256_verified':True,'complete_inventory_verified':True,
            'per_file_fsync_completed':True,'source_preserved':True,'weights_copied':False,
            'checkpoint_bytes':manifest['checkpoint_bytes'],'checks':checks}
    with (stage/'archive_verification.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    if final.exists(): raise FileExistsError(final)
    stage.rename(final)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open('x') as stream: json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
