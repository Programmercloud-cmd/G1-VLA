"""Create explicit model-only inference views and archive canonical continuation state."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import os
import uuid

from project_config import load_config, resolve_path, guard_path, require_storage
from archive_smoke_checkpoint import remote_python, repository_provenance
import archive_gr00t_dataset as old
import evaluate_gr00t_dataset as original
import continuation_artifacts as common


VIEW_PROGRAM = r'''from pathlib import Path
import json,os,shutil,sys
source,destination=map(Path,sys.argv[1:3]);rows=json.loads(sys.argv[3]);method=sys.argv[4]
assert source.is_dir() and not source.is_symlink() and not destination.exists()
destination.mkdir(parents=True)
for row in rows:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
 src=source/relative;dst=destination/relative;s=src.stat()
 assert not src.is_symlink() and (s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino)==tuple(row[k] for k in ['size','mtime_ns','device','inode'])
 dst.parent.mkdir(parents=True,exist_ok=True)
 if method=='hardlink':os.link(src,dst)
 elif method=='verified_copy':
  with src.open('rb') as inp,dst.open('xb') as out:shutil.copyfileobj(inp,out,1024**2);out.flush();os.fsync(out.fileno())
 else:raise ValueError('Unsupported view method')
 if method=='hardlink':assert dst.stat().st_dev==s.st_dev and dst.stat().st_ino==s.st_ino
 with dst.open('rb') as f:os.fsync(f.fileno())
for p in sorted([destination,*[p for p in destination.rglob('*') if p.is_dir()]],key=lambda p:len(p.parts),reverse=True):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'passed':True,'method':method,'files':len(rows),'destination':str(destination)}))
'''


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def fsync_directories(path):
    for directory in sorted([path,*[p for p in path.rglob('*') if p.is_dir()]], key=lambda p:len(p.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def create_view(args, config):
    checkpoint = guard_path(args.checkpoint, config, 'runs')
    destination = guard_path(args.view, config, 'runs')
    receipt = guard_path(args.view_manifest, config, 'reports')
    inventory_path = guard_path(args.view_inventory, config, 'reports')
    common.require(destination != checkpoint and not destination.is_relative_to(checkpoint)
                   and destination.name == checkpoint.name, 'Use a separate native-local-step inference directory')
    for path in [destination, receipt, inventory_path]:
        common.require(not path.exists() and not path.is_symlink(), 'View output already occupied: ' + str(path))
    common.require(config['storage']['enabled'], 'This continuation view requires configured SSH storage')
    archive = require_storage(config)
    common.require(checkpoint.is_relative_to(archive) and destination.is_relative_to(archive), 'Views must share durable storage')
    run, report, prefix, event, full = common.certify_training(resolve_path(args.run_manifest), resolve_path(args.train_report), checkpoint,
                                                             int(checkpoint.name.split('-')[-1]))
    common.require(resolve_path(args.full_inventory) == resolve_path(event['checkpoint_inventory_path']), 'Wrong certified full inventory')
    names = common.model_view_names(full)
    rows = [row for row in full['files'] if row['path'] in names]
    remote = PurePosixPath(config['storage']['remote_path'])
    source_remote = remote / checkpoint.relative_to(archive).as_posix()
    view_remote = remote / destination.relative_to(archive).as_posix()
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', config['storage']['host']]
    remote_python(ssh, VIEW_PROGRAM, source_remote, view_remote, json.dumps(rows), args.method)
    # Hash/fsync the model view once on the storage host; no optimizer bytes cross SSHFS.
    view = old.create_checkpoint_inventory(config, destination, inventory_path, True)
    manifest = {'schema_version': 1, 'artifact_type': 'continuation_model_only_inference_view',
                'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'method': args.method,
                'local_step': full['step'], 'cumulative_step': common.LOCAL_TO_CUMULATIVE[full['step']],
                'view_path': str(destination), 'canonical_full_checkpoint': str(checkpoint),
                'canonical_full_inventory': common.binding(resolve_path(args.full_inventory)),
                'view_inventory': common.binding(inventory_path), 'files': sorted(names),
                'resumable_training_checkpoint': False,
                'canonical_full_resume_reference': str(checkpoint),
                'sampling_limit': common.read(checkpoint/'continuation.json').get('sampling_limit'),
                'training_run': common.binding(resolve_path(args.run_manifest)),
                'training_report': common.binding(resolve_path(args.train_report))}
    common.validate_view_mapping(manifest, full, view)
    common.full_inventory(resolve_path(args.full_inventory), checkpoint)
    write_new(receipt, manifest)
    print(json.dumps({'passed': True, 'view_manifest': str(receipt), 'method': args.method,
                      'local_step': full['step'], 'cumulative_step': manifest['cumulative_step']}))


def collect(args, config):
    declaration, protocol, original_path = common.load_protocol(resolve_path(args.continuation_protocol))
    comparison_path = resolve_path(args.comparison)
    comparison = common.read(comparison_path)
    common.require(comparison.get('artifact_type') == 'g1_continuation_checkpoint_comparison'
                   and comparison.get('passed') is True
                   and comparison['continuation_protocol_sha256'] == original.digest(resolve_path(args.continuation_protocol)),
                   'Continuation paired comparison does not match declaration')
    view_manifest_path = common.verify_binding(comparison['view_manifest'])
    view_inventory_path = common.verify_binding(comparison['view_inventory'])
    view_manifest, full, view = common.load_view(view_manifest_path, view_inventory_path)
    checkpoint = guard_path(view_manifest['canonical_full_checkpoint'], config, 'runs')
    run_path = common.verify_binding(view_manifest['training_run']); train_path = common.verify_binding(view_manifest['training_report'])
    run, training, prefix, event, full = common.certify_training(run_path, train_path, checkpoint, full['step'])
    common.verify_ancestry(run, declaration, original_path)
    common.require(comparison['local_step'] == full['step']
                   and comparison['cumulative_step'] == common.LOCAL_TO_CUMULATIVE[full['step']], 'Comparison step ancestry differs')
    sources = [(run_path, 'reports/run.json'), (train_path, 'reports/training.json'),
               (resolve_path(run['audit']), 'reports/finite.jsonl'), (resolve_path(event['coverage_path']), 'reports/coverage.json'),
               (resolve_path(run['cap_watchdog_report']), 'reports/cap-watchdog.json'),
               (resolve_path(run['cap_watchdog_log']), 'reports/cap-watchdog.log'),
               (resolve_path(run['cap_watchdog_startup_report']), 'reports/cap-watchdog-armed.json'),
               (resolve_path(event['checkpoint_inventory_path']), 'reports/full-checkpoint-inventory.json'),
               (view_manifest_path, 'evaluation/view-manifest.json'), (view_inventory_path, 'evaluation/view-inventory.json'),
               (comparison_path, 'evaluation/comparison.json'),
               (resolve_path(args.continuation_protocol), 'evaluation/continuation-protocol.json'),
               (original_path, 'evaluation/original-protocol.json')]
    paired_ids = []
    for split in ['train', 'validation']:
        report_path = common.verify_binding(comparison['evaluations'][split]['report'])
        report, arrays = original.load_completed(report_path, protocol, original.digest(original_path), split, False)
        old.validate_reload_proof(report); paired_ids.append(report['process_id'])
        common.require(report['checkpoint_global_step'] == full['step']
                       and resolve_path(report['checkpoint']) == resolve_path(view_manifest['view_path'])
                       and report['checkpoint_inventory_sha256'] == original.digest(view_inventory_path), 'Wrong continuation inference identity')
        common.require(report['checkpoint_files_sha256'] == {row['path']: row['sha256'] for row in view['files']},
                       'Continuation inference file identity differs')
        for suffix in ['.json', '.npz', '.png']:
            sources.append((report_path.with_suffix(suffix), 'evaluation/' + split + suffix))
        for role in ['source_' + split, 'baseline_' + split]:
            path = common.verify_binding(declaration['bindings'][role])
            sources += [(path, 'evaluation/' + role + '.json'), (path.with_suffix('.npz'), 'evaluation/' + role + '.npz')]
    common.require(paired_ids[0] == paired_ids[1], 'Continuation splits did not share one model load')
    common.require(report['reload_proof']['policy_reused_between_splits'] is True,
                   'Continuation validation did not reuse the paired policy')
    for name, item in declaration['bindings'].items():
        path = common.verify_binding(item)
        sources.append((path, 'ancestry/'+name+path.suffix))
    for split in ['train','validation']:
        sample = protocol['samples'][split]
        sources += [(resolve_path(sample['path']), 'evaluation/'+split+'-samples.json'),
                    (resolve_path(sample['arrays']['path']), 'evaluation/'+split+'-samples.npz')]
    for name, item in protocol['bindings'].items():
        sources.append((resolve_path(item['path']), 'dataset/'+name+'.json'))
    for name in original.PROCESSOR_FILES:
        sources.append((resolve_path(protocol['processor']['path'])/name, 'evaluation/processor/'+name))
    if protocol['processor'].get('semantics_file'):
        sources.append((resolve_path(protocol['processor']['semantics_file']['path']), 'evaluation/processor/semantics.json'))
    for relative, sha in {**protocol['source_sha256'], **declaration['new_source_sha256'], **run['script_sha256']}.items():
        path = resolve_path(relative); common.require(original.digest(path) == sha, 'Executed source changed')
        sources.append((path, relative))
    unique = {}
    for path, relative in sources:
        common.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 * 1024**2,
                       'Archive evidence must be ordinary bounded files')
        common.require(relative not in unique or unique[relative] == path, 'Archive evidence destination collision')
        unique[relative] = path
    checks = {'local_step': full['step'], 'cumulative_step': common.LOCAL_TO_CUMULATIVE[full['step']],
              'finite_prefix': prefix, 'whole_continuation_completed': training['passed'],
              'canonical_full_state_required': sorted(common.FULL_STATE_FILES),
              'inference_view_resumable': False, 'exact_dataset_cursor_resume_supported': False}
    return checkpoint, full, view_manifest, unique, checks


def archive(args, config):
    checkpoint, full, view_manifest, sources, checks = collect(args, config)
    common.require(re.fullmatch('[A-Za-z0-9_-]+', args.record_name) is not None, 'Use a simple archive record name')
    root = config['paths']['archive']; require_storage(config)
    final = guard_path(root/'checkpoints'/args.record_name, config, 'archive')
    report_path = guard_path(config['paths']['reports']/(args.record_name+'-archive.json'), config, 'reports')
    common.require(not final.exists() and not final.is_symlink() and not report_path.exists(), 'Archive output already exists')
    if args.dry_run:
        print(json.dumps({'passed': True, 'canonical_checkpoint': str(checkpoint), 'evidence_files': len(sources), 'checks': checks})); return
    stage = final.with_name('.'+final.name+'.incomplete-'+uuid.uuid4().hex); stage.mkdir(parents=True)
    evidence = []
    for relative, source in sources.items():
        before = (source.stat().st_size, source.stat().st_mtime_ns); sha = original.digest(source)
        target = stage/relative; target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as inp, target.open('xb') as out:
            shutil.copyfileobj(inp, out, 1024**2); out.flush(); os.fsync(out.fileno())
        common.require(original.digest(target) == sha and (source.stat().st_size, source.stat().st_mtime_ns) == before,
                       'Evidence changed while publishing')
        evidence.append({'path': relative, 'size': before[0], 'sha256': sha})
    common.full_inventory(common.verify_binding(view_manifest['canonical_full_inventory']), checkpoint)
    manifest = {'schema_version': 1, 'artifact_type': 'continuation_durable_full_state_archive',
                'canonical_full_checkpoint_ref': 'archive:'+checkpoint.relative_to(root).as_posix(),
                'canonical_full_resume_reference': str(checkpoint), 'full_checkpoint_files': full['files'],
                'model_only_inference_view': view_manifest, 'evidence_files': evidence, 'checks': checks,
                'weights_or_optimizer_copied': False, 'source_preserved': True, 'repository': repository_provenance()}
    write_new(stage/'archive_manifest.json', manifest)
    report = {'passed': True, 'artifact_type': 'continuation_archive_verification', 'record_directory': str(final),
              'canonical_full_checkpoint_ref': manifest['canonical_full_checkpoint_ref'],
              'manifest_sha256': original.digest(stage/'archive_manifest.json'), 'checks': checks,
              'full_state_sha256_and_fsync_verified': True, 'evidence_file_count': len(evidence),
              'weights_or_optimizer_copied': False, 'source_preserved': True}
    write_new(stage/'archive_verification.json', report)
    fsync_directories(stage)
    common.require(not final.exists(), 'Archive appeared during publication'); stage.rename(final)
    descriptor = os.open(final.parent, os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    write_new(report_path, report); print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    view = sub.add_parser('view')
    for name in ['checkpoint', 'full-inventory', 'run-manifest', 'train-report', 'view', 'view-manifest', 'view-inventory']:
        view.add_argument('--'+name, type=Path, required=True)
    view.add_argument('--method', choices=['hardlink', 'verified_copy'], default='hardlink')
    pub = sub.add_parser('archive')
    for name in ['continuation-protocol', 'comparison']:
        pub.add_argument('--'+name, type=Path, required=True)
    pub.add_argument('--record-name', required=True); pub.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(); config = load_config(args.config)
    {'view': create_view, 'archive': archive}[args.command](args, config)


if __name__ == '__main__':
    main()
