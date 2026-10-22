"""Bindings for the separate G1 extension experiment; original contracts stay intact."""
from pathlib import Path
import json

from project_config import resolve_path
import archive_gr00t_dataset as old_archive
import evaluate_gr00t_dataset as original
import continuation_artifacts as previous
from run_gr00t_pilot import read_audit

BASE_STEP = 8192
LOCAL_TO_CUMULATIVE = {2048: 10240, 4096: 12288}
FULL_STATE_FILES = {'optimizer.pt', 'scheduler.pt', 'rng_state.pth'}
NEW_SOURCES = ['scripts/extension_artifacts.py', 'scripts/evaluate_gr00t_extension.py',
               'scripts/archive_gr00t_extension.py', 'scripts/run_gr00t_extension.py',
               'scripts/watch_extension_cap.py']
SELECTION_RULE = {'candidates_cumulative_steps': [8192, 10240, 12288],
                  'metric': 'validation/all/group_balanced_normalized_mae',
                  'tie': 'earliest_cumulative_step',
                  'available_selection_always_allowed': True,
                  'incomplete_run': 'select_best_available_certified_candidate_and_report_missing_steps',
                  'mandatory_improvement_threshold': None, 'persistence': 'descriptive_only'}
TRAINING_SEGMENT = {'additional_updates': 4096, 'source_weight_updates': 8192,
                    'optimizer_and_scheduler': 'fresh', 'learning_rate': 2.5e-5,
                    'warmup_ratio': 0.025, 'warmup_steps': 103, 'scheduler': 'cosine',
                    'sampling_seed': 8234, 'frozen_split_seed': 42,
                    'exact_dataset_cursor_resume_supported': False}
FULL_STATE_POLICY = {'new_checkpoints_require': sorted(FULL_STATE_FILES),
                     'durable_storage_host_sha256_and_fsync_before_next_training_update': True,
                     'inference_views_resumable': False,
                     'canonical_full_checkpoint_is_resume_reference': True,
                     'exact_iterable_dataset_cursor_resume_supported': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return old_archive.read_json(Path(path))


def binding(path):
    return original.binding(Path(path))


def verify_binding(item):
    path = resolve_path(item['path'])
    require(path.is_file() and not path.is_symlink(), 'Binding is not an ordinary file')
    require(original.digest(path) == item['sha256'], 'Changed binding: ' + str(path))
    return path


def load_protocol(path):
    value = read(path)
    require(value.get('artifact_type') == 'g1_extension_evaluation_protocol' and value.get('schema_version') == 1,
            'Unsupported extension evaluation declaration')
    require(value.get('local_to_cumulative') == {'2048': 10240, '4096': 12288}, 'Extension step mapping changed')
    require(value.get('cumulative_base_step') == BASE_STEP, 'Extension source step changed')
    original_path = verify_binding(value['original_evaluation_protocol'])
    protocol = original.load_protocol(original_path)
    prior_path = verify_binding(value['previous_continuation_protocol'])
    prior, prior_original, prior_original_path = previous.load_protocol(prior_path)
    require(prior_original_path == original_path and prior_original == protocol, 'Extension changed original protocol ancestry')
    frozen = {**protocol['source_sha256'], **prior['new_source_sha256']}
    require(value['frozen_source_sha256'] == frozen and len(frozen) == 27, 'Extension must preserve all27 earlier runtime sources')
    for name, sha in frozen.items():
        require(original.digest(resolve_path(name)) == sha, 'Earlier frozen source changed: '+name)
    require(value['samples'] == protocol['samples'] and value['processor'] == protocol['processor'],
            'Extension must reuse the exact frozen samples and training normalization')
    require(value['inference_recipe'] == protocol['recipe'] and value['selection_rule'] == SELECTION_RULE,
            'Extension inference or selection rule changed')
    require(value['training_segment'] == TRAINING_SEGMENT, 'Extension training declaration changed')
    require(value['full_state_policy'] == FULL_STATE_POLICY, 'Extension full-state policy changed')
    expected_bindings = {'source_inventory','source_archive','source_archive_manifest','source_comparison',
                         'source_view_manifest','source_view_inventory','selected_policy',
                         'source_run_manifest','source_train_report','source_closeout','source_independent_proof'}
    expected_bindings |= {role+'_'+split+suffix for role in ['source','baseline']
                          for split in ['train','validation'] for suffix in ['', '_arrays']}
    require(set(value['bindings']) == expected_bindings, 'Extension evidence binding set differs')
    for item in value['bindings'].values():
        verify_binding(item)
    require(set(value['new_source_sha256']) == set(NEW_SOURCES), 'Missing extension source binding')
    for name, sha in value['new_source_sha256'].items():
        require(original.digest(resolve_path(name)) == sha, 'Extension implementation changed: ' + name)
    return value, protocol, original_path


def full_inventory(path, checkpoint):
    """Recheck remote complete signatures and small JSON, without rereading optimizer tensors."""
    value = read(path); checkpoint = resolve_path(checkpoint)
    require(value.get('artifact_type') == 'durable_checkpoint_inventory' and value.get('schema_version') == 1
            and value.get('passed') is True and value.get('per_file_fsync_completed') is True,
            'Canonical full-state inventory is incomplete')
    require(resolve_path(value['checkpoint']) == checkpoint and value['step'] in LOCAL_TO_CUMULATIVE,
            'Canonical full-state inventory identity differs')
    files = old_archive.checkpoint_files(checkpoint, value['step'])
    require(FULL_STATE_FILES <= set(files), 'Canonical checkpoint lacks optimizer/scheduler/RNG state')
    require(all(files[name].stat().st_size > 0 for name in FULL_STATE_FILES), 'Empty full-state artifact')
    require(value.get('remote_identity') is not None and value.get('hash_location') == 'configured_storage_host',
            'Full-state files require storage-host SHA256/fsync certification')
    old_archive.verify_remote_signatures(value['remote_identity'], value['files'])
    old_archive.verify_inventory_records(files, value['files'])
    for row in value['files']:
        if row['path'].endswith('.safetensors'):
            continue
        if row['size'] <= 64 * 1024**2:
            require(original.digest(files[row['path']]) == row['sha256'], 'Canonical metadata changed')
        else:
            require(row['path'] in FULL_STATE_FILES, 'Unknown oversized checkpoint metadata')
    metadata = read(checkpoint/'continuation.json')
    require(metadata.get('artifact_type') == 'larger_dataset_extension_checkpoint'
            and metadata['local_step'] == value['step']
            and metadata['cumulative_base_step'] == BASE_STEP
            and metadata['cumulative_step'] == LOCAL_TO_CUMULATIVE[value['step']], 'Canonical extension metadata differs')
    require(metadata.get('sampling_limit') and metadata['full_state']['passed'] is True
            and metadata['full_state']['scheduler_epoch'] == value['step']
            and metadata['full_state']['exact_iterable_cursor_restore'] is False,
            'Canonical full-state or honest sampler-limit proof missing')
    return value


def model_view_names(inventory):
    names = {row['path'] for row in inventory['files']}
    required = {'config.json', 'trainer_state.json', 'model.safetensors.index.json',
                *original.PROCESSOR_FILES, 'continuation.json'}
    shards = {name for name in names if name.endswith('.safetensors')}
    require(required <= names and shards, 'Canonical model/extension files missing')
    return required | shards


def validate_view_mapping(manifest, full, view_inventory):
    require(manifest.get('artifact_type') == 'extension_model_only_inference_view'
            and manifest.get('resumable_training_checkpoint') is False,
            'Inference view must be explicitly nonresumable')
    require(manifest['local_step'] == full['step'] == view_inventory['step']
            and manifest['cumulative_step'] == LOCAL_TO_CUMULATIVE[full['step']], 'Inference view step differs')
    require(resolve_path(manifest['canonical_full_checkpoint']) == resolve_path(full['checkpoint'])
            and resolve_path(manifest['view_path']) == resolve_path(view_inventory['checkpoint']), 'View ancestry differs')
    require(set(row['path'] for row in view_inventory['files']) == model_view_names(full),
            'Inference view is not the exact model-only file set')
    source = {row['path']: row for row in full['files']}
    for row in view_inventory['files']:
        parent = source[row['path']]
        require(row['size'] == parent['size'] and row['sha256'] == parent['sha256'], 'View bytes differ from canonical full checkpoint')
        if manifest['method'] == 'hardlink':
            require((row['device'], row['inode']) == (parent['device'], parent['inode']), 'View is not the declared hard link')
        else:
            require(manifest['method'] == 'verified_copy', 'Unsupported model-view creation method')


def load_view(manifest_path, inventory_path):
    manifest = read(manifest_path)
    require(verify_binding(manifest['view_inventory']) == resolve_path(inventory_path), 'View inventory binding differs')
    full_path = verify_binding(manifest['canonical_full_inventory'])
    full = full_inventory(full_path, manifest['canonical_full_checkpoint'])
    view = old_archive.load_checkpoint_inventory(resolve_path(inventory_path), resolve_path(manifest['view_path']))
    validate_view_mapping(manifest, full, view)
    return manifest, full, view


def verify_ancestry(run, declaration, original_path):
    require(run['evaluation_protocol_sha256'] == original.digest(original_path), 'Extension original evaluation protocol changed')
    require(original.digest(resolve_path(run['extension_protocol'])) == run['extension_protocol_sha256']
            and read(resolve_path(run['extension_protocol'])) == declaration, 'Extension declaration binding differs')
    for name, sha in run['script_sha256'].items():
        require(original.digest(resolve_path(name)) == sha, 'Extension executed source changed: '+name)
    require(resolve_path(run['initial_checkpoint']['path']) == resolve_path(declaration['source_checkpoint'])
            and run['initial_checkpoint']['source_updates'] == BASE_STEP, 'Extension has wrong weight ancestry')
    require(run['fresh_optimizer_and_scheduler'] is True and run['seed'] == 8234
            and run['learning_rate'] == 2.5e-5 and run['warmup_steps'] == 103, 'Extension segment recipe differs')
    require(run['save_only_model'] is False and run['max_steps'] == 4096
            and run['accumulation'] == 8 and run['state_dropout_prob'] == 0.0,
            'Extension saving or training recipe differs')
    require(verify_binding(declaration['previous_continuation_protocol']) == resolve_path(run['previous_protocol']),
            'Extension preceding declaration differs')
    verify_source_prefix(run['source_prefix'], original.digest(original_path))
    for run_key, declaration_key in [('source_checkpoint_inventory', 'source_inventory'),
                                      ('source_archive_report', 'source_archive'),
                                      ('source_evaluation_train', 'source_train'),
                                      ('source_evaluation_validation', 'source_validation'),
                                      ('source_comparison','source_comparison'),
                                      ('source_view_manifest','source_view_manifest'),
                                      ('source_view_inventory','source_view_inventory'),
                                      ('selected_policy','selected_policy'),
                                      ('source_run_manifest','source_run_manifest'),
                                      ('source_train_report','source_train_report')]:
        path = verify_binding(declaration['bindings'][declaration_key])
        require(resolve_path(run[run_key]) == path, 'Extension source evidence differs: '+run_key)


def verify_source_prefix(lineage, protocol_sha=None):
    """Validate two real local4096 audits; never fabricate one native8192 audit."""
    require(lineage.get('schema_version') == 2
            and lineage.get('artifact_type') == 'certified_multisegment_weight_lineage'
            and lineage.get('cumulative_source_updates') == BASE_STEP
            and lineage.get('source_unsaved1135_updates_excluded') is True,
            'Extension requires certified two-segment source lineage')
    segments = lineage['native_segments']
    require(len(segments) == 2 and lineage['lineage_coverage']['consumed_examples'] == 65536,
            'Source lineage must retain exactly two native segments and65536 exposures')
    for segment, base in zip(segments, [0,4096]):
        require(segment['cumulative_base_step'] == base and segment['local_updates'] == 4096
                and segment['cumulative_end_step'] == base+4096, 'Source native/cumulative segment mapping differs')
        run_path = verify_binding(segment['run_manifest']); report_path = verify_binding(segment['train_report'])
        audit_path = verify_binding(segment['audit']); source_run = read(run_path); source_report = read(report_path)
        if base == 0:
            require(source_run['artifact_type'] == 'larger_dataset_training_run' and source_run['max_steps'] == 8192,
                    'First source segment is not the original bounded run')
        else:
            require(source_run['artifact_type'] == 'larger_dataset_continuation_run'
                    and source_run['cumulative_base_step'] == 4096 and source_run['max_steps'] == 4096
                    and source_run['initial_checkpoint']['source_updates'] == 4096,
                    'Second source segment is not the actual4096 additional-update continuation')
        require(source_report['run_manifest_sha256'] == original.digest(run_path)
                and source_report['finite_audit_sha256'] == original.digest(audit_path)
                and resolve_path(source_run['audit']) == audit_path, 'Source native audit ancestry changed')
        if protocol_sha is not None:
            require(source_run['evaluation_protocol_sha256'] == source_report['evaluation_protocol_sha256'] == protocol_sha,
                    'Source segment original protocol changed')
        require(audit_path.stat().st_size <= 64*1024**2, 'Source audit exceeds bounded metadata size')
        records = read_audit(audit_path); finite, event = old_archive.prefix_integrity(records,4096)
        endpoint = next(i for i,row in enumerate(records) if row['event']=='checkpoint' and row['step']==4096)
        import hashlib
        sha = hashlib.sha256(b''.join(audit_path.read_bytes().splitlines(keepends=True)[:endpoint+1])).hexdigest()
        proof = segment['finite_prefix']; coverage = resolve_path(proof['coverage_path'])
        require(proof['finite_summary'] == finite and proof['audit_prefix_sha256'] == sha
                and original.digest(coverage) == proof['coverage_sha256'] == event['coverage_sha256']
                and resolve_path(event['coverage_path']) == coverage and read(coverage)['consumed_examples'] == 32768,
                'Source native4096 prefix or coverage changed')
    return segments


def certify_training(run_path, report_path, checkpoint, local_step):
    run = read(run_path); report = read(report_path)
    require(run.get('artifact_type') == 'larger_dataset_extension_run'
            and report.get('artifact_type') == 'larger_dataset_extension_training_report', 'Wrong extension training provenance')
    require(report['run_manifest_sha256'] == original.digest(run_path), 'Extension run manifest changed')
    require(report['run'] == run['run'] and report['evaluation_protocol_sha256'] == run['evaluation_protocol_sha256'],
            'Extension final report identity differs')
    require(run['max_steps'] == 4096 and run['cumulative_base_step'] == BASE_STEP
            and run['local_to_cumulative'] == {'2048': 10240, '4096': 12288}, 'Extension recipe identity differs')
    require(resolve_path(checkpoint).parent == resolve_path(run['output']), 'Canonical checkpoint is outside extension run')
    saved = [row for row in report['checkpoints'] if row['step'] == local_step]
    require(len(saved) == 1 and saved[0]['passed'] is True
            and resolve_path(saved[0]['checkpoint']) == resolve_path(checkpoint), 'Checkpoint lacks final training certification')
    audit_path = resolve_path(run['audit'])
    require(audit_path.stat().st_size <= 64 * 1024**2 and original.digest(audit_path) == report['finite_audit_sha256'],
            'Extension finite audit changed or is oversized')
    prefix, event = old_archive.prefix_integrity(read_audit(audit_path), local_step)
    require(event.get('cumulative_step') == LOCAL_TO_CUMULATIVE[local_step] and event.get('full_state_required') is True,
            'Checkpoint lacks truthful local/cumulative full-state save audit')
    coverage = resolve_path(event['coverage_path'])
    require(original.digest(coverage) == event['coverage_sha256']
            and read(coverage)['consumed_examples'] == local_step * 8, 'Extension consumed coverage differs')
    inventory_path = resolve_path(event['checkpoint_inventory_path'])
    require(original.digest(inventory_path) == event['checkpoint_inventory_sha256'], 'Save-time full-state inventory changed')
    full = full_inventory(inventory_path, checkpoint)
    require(full['step'] == local_step and saved[0].get('cumulative_step') == LOCAL_TO_CUMULATIVE[local_step],
            'Certified checkpoint step mapping differs')
    metadata_path = resolve_path(checkpoint)/'continuation.json'
    require(original.digest(metadata_path) == event['continuation_metadata_sha256']
            and event.get('directories_fsync_completed') is True
            and event.get('per_file_fsync_completed') is True, 'Save-time metadata/fsync certification differs')
    metadata = read(metadata_path)
    require(resolve_path(metadata['source_checkpoint']) == resolve_path(run['initial_checkpoint']['path'])
            and metadata['source_checkpoint_inventory_sha256'] == run['source_checkpoint_inventory_sha256'],
            'Canonical full-state ancestry differs')
    return run, report, prefix, event, full
