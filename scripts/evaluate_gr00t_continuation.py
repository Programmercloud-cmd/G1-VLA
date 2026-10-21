"""Declare and compare a truthful weight-warm-start continuation using the frozen evaluator."""
import argparse
from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import subprocess

from project_config import load_config, resolve_path, guard_path, load_runtime
import evaluate_gr00t_dataset as original
import archive_gr00t_dataset as old_archive
import continuation_artifacts as common


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def prepare(args, config):
    path = resolve_path(args.output)
    common.require(not path.exists(), 'Continuation declaration already exists')
    original_path = resolve_path(args.original_protocol); protocol = original.load_protocol(original_path)
    inventory_path = resolve_path(args.source_inventory)
    inventory = old_archive.load_checkpoint_inventory(inventory_path, resolve_path(args.source_checkpoint))
    common.require(inventory['step'] == 4096, 'Continuation must start at certified actual source step4096')
    archive = common.read(resolve_path(args.source_archive))
    common.require(archive['passed'] is True and archive['checkpoint_sha256_verified'] is True
                   and archive['checks']['step'] == 4096 and archive['complete_inventory_verified'] is True
                   and archive['per_file_fsync_completed'] is True and archive['source_preserved'] is True
                   and archive['checkpoint_ref'] == inventory['checkpoint_ref'], 'Source4096 archive is not certified')
    archive_manifest = resolve_path(archive['record_directory'])/'archive_manifest.json'
    if not archive_manifest.exists():
        archive_manifest = resolve_path(archive['record_directory'])/'manifest.json'
    common.require(original.digest(archive_manifest) == archive['manifest_sha256'], 'Source archive manifest changed')
    archived = common.read(archive_manifest)
    common.require({r['path']: (r['size'],r['sha256']) for r in archived['checkpoint_files']}
                   == {r['path']: (r['size'],r['sha256']) for r in inventory['files']}, 'Source archive inventory differs')
    bindings = {'source_inventory': common.binding(inventory_path),
                'source_archive': common.binding(resolve_path(args.source_archive)),
                'source_archive_manifest': common.binding(archive_manifest)}
    records = {}
    for split in ['train', 'validation']:
        for role, step_zero in [('source', False), ('baseline', True)]:
            report_path = resolve_path(getattr(args, role+'_'+split))
            report, arrays = original.load_completed(report_path, protocol, original.digest(original_path), split, step_zero)
            if not step_zero:
                common.require(report['checkpoint_global_step'] == 4096
                               and resolve_path(report['checkpoint']) == resolve_path(args.source_checkpoint)
                               and report['checkpoint_inventory_sha256'] == original.digest(inventory_path), 'Wrong source4096 inference')
                old_archive.validate_reload_proof(report)
                records[split] = report
            bindings[role+'_'+split] = common.binding(report_path)
            bindings[role+'_'+split+'_arrays'] = common.binding(report_path.with_suffix('.npz'))
    common.require(records['train']['process_id'] == records['validation']['process_id'], 'Source4096 needs a paired strict reload')
    declaration = {'schema_version': 1, 'artifact_type': 'g1_continuation_evaluation_protocol',
                   'status': 'declared_before_continuation_training', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                   'original_evaluation_protocol': common.binding(original_path),
                   'source_checkpoint': str(resolve_path(args.source_checkpoint)), 'cumulative_base_step': 4096,
                   'local_to_cumulative': {'2048': 6144, '4096': 8192},
                   'step_semantics': 'HF trainer global_step and checkpoint names remain honest local additional-update counts. Cumulative values describe model-weight ancestry only; original optimizer state is unavailable.',
                   'inference_recipe': protocol['recipe'], 'samples': protocol['samples'], 'processor': protocol['processor'],
                   'bindings': bindings, 'selection_rule': common.SELECTION_RULE,
                   'training_segment': common.TRAINING_SEGMENT,
                   'new_source_sha256': {name: original.digest(resolve_path(name)) for name in common.NEW_SOURCES},
                   'full_state_policy': common.FULL_STATE_POLICY,
                   'scope': 'Same frozen100-window train/validation samples, original stepzero and actual4096 predictions; no mandatory persistence or improvement threshold; no robot competence claim.'}
    write_new(path, declaration); common.load_protocol(path)
    print(json.dumps({'passed': True, 'continuation_protocol': str(path), 'sha256': original.digest(path)}))


def comparison_groups(new_arrays, source_arrays):
    values = original.summarize_arrays(new_arrays['absolute_predictions'], new_arrays['targets'], new_arrays['states'],
                                       new_arrays['lower'], new_arrays['upper'], new_arrays['changing_target_mask'],
                                       source_arrays['absolute_predictions'])
    # The frozen helper names its generic reference "step_zero". Relabel only
    # this NEW comparison record so the actual parent4096 reference is truthful.
    for group in values.values():
        for subset in group.values():
            if isinstance(subset, dict) and 'step_zero' in subset:
                subset['source_4096'] = subset.pop('step_zero')
                subset['change_from_source_4096'] = subset.pop('change_from_step_zero')
    return values


def metric_episode(arrays, mask):
    # NPZ also stores a three-element seed array; only sample-indexed metric
    # arrays are sliced. Bounds have horizon/channel axes, never sample axes.
    return {key: arrays[key][mask] if key not in ['lower', 'upper'] else arrays[key]
            for key in ['absolute_predictions','targets','states','lower','upper','changing_target_mask']}


def compare(args, config):
    import numpy as np
    declaration_path = resolve_path(args.continuation_protocol)
    declaration, protocol, original_path = common.load_protocol(declaration_path)
    manifest_path = resolve_path(args.view_manifest); inventory_path = resolve_path(args.view_inventory)
    view_manifest, full, view = common.load_view(manifest_path, inventory_path)
    run, training, prefix, event, full = common.certify_training(common.verify_binding(view_manifest['training_run']),
                                                              common.verify_binding(view_manifest['training_report']),
                                                              view_manifest['canonical_full_checkpoint'], full['step'])
    common.verify_ancestry(run, declaration, original_path)
    results = {}; ids = []
    for split in ['train', 'validation']:
        path = resolve_path(getattr(args, split+'_report'))
        report, arrays = original.load_completed(path, protocol, original.digest(original_path), split, False)
        old_archive.validate_reload_proof(report); ids.append(report['process_id'])
        common.require(report['checkpoint_global_step'] == full['step']
                       and resolve_path(report['checkpoint']) == resolve_path(view_manifest['view_path'])
                       and report['checkpoint_inventory_sha256'] == original.digest(inventory_path), 'Wrong continuation view inference')
        common.require(report['checkpoint_files_sha256'] == {row['path']: row['sha256'] for row in view['files']}, 'View weight identity differs')
        source_report, source_arrays = original.load_completed(common.verify_binding(declaration['bindings']['source_'+split]),
                                                               protocol, original.digest(original_path), split, False)
        for key in ['model_semantics_sha256', 'processor_semantics_sha256', 'statistics_sha256']:
            common.require(report[key] == source_report[key], 'Continuation inference semantics changed: '+key)
        groups = comparison_groups(arrays, source_arrays)
        episodes = {}
        for episode in sorted(set(arrays['episode_indices'].tolist())):
            mask = arrays['episode_indices'] == episode
            episodes[str(episode)] = comparison_groups(metric_episode(arrays, mask),
                                                       {'absolute_predictions': source_arrays['absolute_predictions'][mask]})
        source_score = source_report['group_balanced_normalized_mae']; score = report['group_balanced_normalized_mae']
        results[split] = {'report': common.binding(path), 'arrays': common.binding(path.with_suffix('.npz')),
                          'local_step': full['step'], 'cumulative_weight_update_lineage': common.LOCAL_TO_CUMULATIVE[full['step']],
                          'source_4096_report': declaration['bindings']['source_'+split],
                          'group_balanced_normalized_mae': score, 'source_4096_score': source_score,
                          'score_change_from_source_4096': score-source_score, 'groups_vs_source_4096': groups,
                          'per_episode_vs_source_4096': episodes}
    common.require(ids[0] == ids[1], 'Continuation splits must share one strict model load')
    validation = common.read(resolve_path(args.validation_report))
    common.require(validation['reload_proof']['policy_reused_between_splits'] is True, 'Validation did not reuse shared policy')
    output = guard_path(args.output, config, 'reports')
    record = {'schema_version': 1, 'artifact_type': 'g1_continuation_checkpoint_comparison', 'passed': True,
              'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'continuation_protocol_sha256': original.digest(declaration_path),
              'original_protocol_sha256': original.digest(original_path),
              'view_manifest': common.binding(manifest_path), 'view_inventory': common.binding(inventory_path),
              'canonical_full_resume_reference': view_manifest['canonical_full_resume_reference'],
              'inference_view_resumable': False, 'local_step': full['step'],
              'cumulative_step': common.LOCAL_TO_CUMULATIVE[full['step']], 'same_model_process_id': ids[0],
              'evaluations': results, 'sampling_limit': view_manifest['sampling_limit'],
              'scope': 'Original raw evaluator reports retain local global_step and original protocol identity. This additive record supplies cumulative weight ancestry and explicit changes versus actual source4096.'}
    write_new(output, record); print(json.dumps({'passed': True, 'comparison': str(output)}))


def evaluate(args, config):
    declaration, protocol, original_path = common.load_protocol(resolve_path(args.continuation_protocol))
    manifest, full, view = common.load_view(resolve_path(args.view_manifest), resolve_path(args.view_inventory))
    run, training, prefix, event, full = common.certify_training(common.verify_binding(manifest['training_run']),
                                                              common.verify_binding(manifest['training_report']),
                                                              manifest['canonical_full_checkpoint'], full['step'])
    common.verify_ancestry(run, declaration, original_path)
    for value in [args.train_report, args.validation_report]:
        path = guard_path(value, config, 'reports')
        for output in [path, path.with_suffix('.npz'), path.with_suffix('.png')]:
            common.require(not output.exists() and not output.is_symlink(), 'Inference output already occupied')
    output = guard_path(args.output, config, 'reports')
    common.require(not output.exists() and not output.is_symlink(), 'Comparison output already exists')
    runtime = load_runtime(config)
    command = [str(runtime['interpreters']['gr00t']), '-B', str(resolve_path('scripts/launch_gr00t_evaluation.py')),
               '--config', str(config['config_path']), 'evaluate', '--protocol', str(original_path), '--split', 'both',
               '--checkpoint', manifest['view_path'], '--checkpoint-inventory', str(resolve_path(args.view_inventory)),
               '--train-reference', str(common.verify_binding(declaration['bindings']['baseline_train'])),
               '--validation-reference', str(common.verify_binding(declaration['bindings']['baseline_validation'])),
               '--train-report', str(resolve_path(args.train_report)), '--validation-report', str(resolve_path(args.validation_report))]
    subprocess.run(command, check=True)
    compare(args, config)


def select(args, config):
    declaration, protocol, original_path = common.load_protocol(resolve_path(args.continuation_protocol))
    candidates = []
    parent_path = common.verify_binding(declaration['bindings']['source_validation'])
    parent, arrays = original.load_completed(parent_path, protocol, original.digest(original_path), 'validation', False)
    candidates.append({'cumulative_step': 4096, 'validation_score': parent['group_balanced_normalized_mae'], 'report': common.binding(parent_path)})
    seen = set()
    for path in args.comparisons:
        item = common.read(resolve_path(path)); step = item['cumulative_step']
        common.require(item['artifact_type'] == 'g1_continuation_checkpoint_comparison' and item['passed'] is True
                       and item['continuation_protocol_sha256'] == original.digest(resolve_path(args.continuation_protocol))
                       and item.get('local_step') in common.LOCAL_TO_CUMULATIVE
                       and common.LOCAL_TO_CUMULATIVE[item['local_step']] == step
                       and step in [6144, 8192] and step not in seen, 'Invalid or repeated continuation comparison')
        seen.add(step)
        view_manifest_path = common.verify_binding(item['view_manifest'])
        view_inventory_path = common.verify_binding(item['view_inventory'])
        manifest, full, view = common.load_view(view_manifest_path, view_inventory_path)
        run, training, prefix, event, full = common.certify_training(common.verify_binding(manifest['training_run']),
                                                                  common.verify_binding(manifest['training_report']),
                                                                  manifest['canonical_full_checkpoint'], full['step'])
        common.verify_ancestry(run, declaration, original_path)
        ids = []
        for split in ['train', 'validation']:
            raw = common.verify_binding(item['evaluations'][split]['report'])
            report, arrays = original.load_completed(raw, protocol, original.digest(original_path), split, False)
            old_archive.validate_reload_proof(report); ids.append(report['process_id'])
            common.require(report['checkpoint_global_step'] == item['local_step'] == full['step']
                           and resolve_path(report['checkpoint']) == resolve_path(manifest['view_path'])
                           and report['checkpoint_inventory_sha256'] == original.digest(view_inventory_path)
                           and report['checkpoint_files_sha256'] == {row['path']: row['sha256'] for row in view['files']},
                           'Comparison checkpoint identity differs')
        common.require(ids[0] == ids[1] and report['reload_proof']['policy_reused_between_splits'] is True,
                       'Ranking requires a complete paired shared-model evaluation')
        candidates.append({'cumulative_step': step, 'validation_score': report['group_balanced_normalized_mae'],
                           'comparison': common.binding(resolve_path(path))})
    candidates.sort(key=lambda row: (row['validation_score'], row['cumulative_step']))
    complete = seen == {6144, 8192}
    output = guard_path(args.output, config, 'reports')
    write_new(output, {'schema_version': 1, 'artifact_type': 'g1_continuation_checkpoint_ranking', 'passed': True,
                      'continuation_protocol_sha256': original.digest(resolve_path(args.continuation_protocol)),
                      'selection_rule': declaration['selection_rule'], 'ranked_candidates': candidates,
                      'formal_continuation_selection_complete': complete,
                      'selected_cumulative_step': candidates[0]['cumulative_step'] if complete else None,
                      'missing_cumulative_steps': sorted({6144, 8192}-seen),
                      'scope': 'New continuation rule; original four-checkpoint selector is unchanged. Incomplete cases are descriptive only.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ['original-protocol', 'source-checkpoint', 'source-inventory', 'source-archive',
                 'source-train', 'source-validation', 'baseline-train', 'baseline-validation', 'output']:
        prep.add_argument('--'+name, type=Path, required=True)
    for name in ['evaluate', 'compare']:
        item = sub.add_parser(name)
        for flag in ['continuation-protocol', 'view-manifest', 'view-inventory', 'train-report', 'validation-report', 'output']:
            item.add_argument('--'+flag, type=Path, required=True)
    choice = sub.add_parser('select'); choice.add_argument('--continuation-protocol', type=Path, required=True)
    choice.add_argument('--comparisons', type=Path, nargs='*', default=[]); choice.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); config = load_config(args.config)
    {'prepare': prepare, 'evaluate': evaluate, 'compare': compare, 'select': select}[args.command](args, config)


if __name__ == '__main__':
    main()
