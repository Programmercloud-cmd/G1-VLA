"""Run the declared larger-dataset learning curve without changing pilot contracts."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import signal
import subprocess
import sys
import time

from project_config import (PROJECT_ROOT as ROOT, load_config, load_runtime, resolve_path,
                            guard_derived, guard_path, runtime_environment)
from prepare_pilot import digest
from convert_single_camera import ensure_python
from run_gr00t_smoke import GIB, available_memory, existing_ancestor, prepare_offline_workdir, smoke_command
from run_gr00t_pilot import finite_summary, read_audit, stop_process_group, require_finite, json_finite

STEPS = 8192
ACCUMULATION = 8
CHECKPOINT_STEPS = [2048, 4096, 6144, 8192]
OUTPUT_START_GIB = 80
PROCESSOR_FILES = ('processor_config.json', 'statistics.json', 'embodiment_id.json')
PILOT_EPISODES = {'episode_0037', 'episode_0040', 'episode_0042', 'episode_0046', 'episode_0047'}
SOURCE_FILES = ['scripts/run_gr00t_dataset.py', 'scripts/run_gr00t_pilot.py',
                'scripts/run_gr00t_smoke.py', 'scripts/evaluate_gr00t_dataset.py',
                'scripts/evaluate_gr00t_pilot.py', 'scripts/compare_pilot_temporal_change.py',
                'scripts/archive_gr00t_dataset.py', 'scripts/archive_smoke_checkpoint.py',
                'scripts/convert_single_camera.py', 'scripts/prepare_pilot.py', 'scripts/prepare_dataset_split.py',
                'scripts/project_config.py', 'configs/g1_inspire_config.py', 'configs/modality.json']


def write_new_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def episode_geometry(snapshot):
    episodes = snapshot.get('episodes', [])
    if not episodes or [e['derived_episode_index'] for e in episodes] != list(range(len(episodes))):
        raise ValueError('Snapshot indices must be contiguous and nonempty')
    lengths = [e['rows'] - 15 for e in episodes]
    if any(n <= 0 for n in lengths):
        raise ValueError('Every episode must have a complete 16-target window')
    return lengths, min(math.ceil(sum(lengths) / 256), len(lengths))


def validate_split(split, train, validation):
    if split.get('schema_version') != 1 or split.get('artifact_type') != 'frozen_dataset_split' or split.get('seed') != 42:
        raise ValueError('Expected the frozen seed-42 episode split')
    names = {}
    for role, snapshot in [('train', train), ('validation', validation)]:
        episode_geometry(snapshot)
        names[role] = [e['episode'] for e in snapshot['episodes']]
        if names[role] != split['splits'][role] or len(set(names[role])) != len(names[role]):
            raise ValueError('Snapshot does not match the explicit split episode order')
        declared = {e['episode']: e for e in split['episodes'] if e['split'] == role}
        if set(declared) != set(names[role]):
            raise ValueError('Split episode metadata inventory differs')
        for record in snapshot['episodes']:
            source = declared[record['episode']]
            if any(record.get(key) != source.get(key) for key in
                   ['rows', 'derived_episode_index', 'source_data_json_sha256', 'effective_goal']):
                raise ValueError('Snapshot source/label binding differs from split')
    if set(names['train']) & set(names['validation']):
        raise ValueError('Training and validation episodes overlap')
    if not PILOT_EPISODES.issubset(names['train']):
        raise ValueError('All five previously used pilot episodes must remain in training')
    if set(names['train'] + names['validation']) & {'episode_0092', 'episode_0136', 'episode_0165'}:
        raise ValueError('Held episodes cannot enter this experiment')
    return episode_geometry(train)


def dataset_command(config, runtime, dataset, output, shards):
    if type(shards) is not int or shards < 1:
        raise ValueError('A positive measured shard count is required')
    command = smoke_command(config, runtime, dataset, output)
    for flag, value in [('--max-steps', STEPS), ('--save-steps', 2048), ('--save-total-limit', 4),
                        ('--gradient-accumulation-steps', ACCUMULATION), ('--num-shards-per-epoch', shards)]:
        command[command.index(flag) + 1] = str(value)
    return command + ['--learning-rate', '0.0001', '--warmup-ratio', '0.05',
                      '--state-dropout-prob', '0.0', '--no-resume-from-checkpoint']


def check_resources(project_free, reports_free, output_free, host_available, gpu_total_mib,
                    gpu_free_mib, reserve_gib=8, startup=True):
    limits = {'project': project_free >= reserve_gib * GIB, 'reports': reports_free >= reserve_gib * GIB,
              'output': output_free >= (OUTPUT_START_GIB if startup else 8) * GIB,
              'host_ram': host_available >= (60 if startup else 16) * GIB}
    if startup:
        limits['gpu'] = gpu_total_mib >= 40960 and gpu_free_mib >= 40960
    if not all(limits.values()):
        raise RuntimeError('Insufficient resource reserve: ' + ', '.join(k for k, v in limits.items() if not v))
    return limits


def gpu_memory():
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.total,memory.free,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return tuple(int(x.strip()) for x in result.stdout.strip().split(','))


def verify_baseline(path, role, protocol_path, protocol):
    value = json.loads(path.read_text())
    samples = protocol['samples'][role]
    if (value.get('passed') is not True or value.get('role') != 'step_zero_reference'
            or value.get('checkpoint_global_step') != 0 or value.get('split') != role
            or value.get('evaluation_protocol_sha256') != digest(protocol_path)
            or value.get('processor_semantics_sha256') != protocol['processor']['semantics_sha256']
            or value.get('statistics_sha256') != protocol['processor']['files_sha256']['statistics.json']
            or value.get('sample_manifest_sha256') != samples['sha256']):
        raise ValueError('Complete the matching step-zero reference for ' + role)
    arrays = path.with_suffix('.npz')
    if value.get('artifacts', {}).get(arrays.name) != digest(arrays):
        raise ValueError('Step-zero arrays differ from the report: ' + role)
    return {str(path): digest(path), str(arrays): digest(arrays)}


class CoverageAudit:
    """Count consumed samples, keeping preloaded and collated lookahead separate."""
    def __init__(self, episodes):
        self.episodes = episodes
        self.object_rows = {}
        self.pending = deque()
        self.counts = Counter()

    def register(self, datapoints, coordinates):
        if len(datapoints) != len(coordinates):
            raise ValueError('Shard samples and source coordinates differ')
        for point, (episode, row) in zip(datapoints, coordinates):
            if not 0 <= episode < len(self.episodes) or not 0 <= row < self.episodes[episode]['rows'] - 15:
                raise ValueError('Invalid training sample coordinate')
            self.object_rows[id(point)] = (int(episode), int(row))

    def collated(self, features):
        if len(features) != 1:
            raise ValueError('Coverage audit requires microbatch one')
        try:
            self.pending.append(self.object_rows[id(features[0])])
        except KeyError as error:
            raise ValueError('Untracked sample reached the training collator') from error

    def consume(self):
        if not self.pending:
            raise ValueError('A loss was computed without a tracked sample')
        coordinate = self.pending.popleft()
        self.counts[coordinate] += 1
        return {'episode_index': coordinate[0], 'row': coordinate[1]}

    def summary(self):
        result = []
        for i, episode in enumerate(self.episodes):
            counts = [n for (ep, _), n in self.counts.items() if ep == i]
            result.append({'episode': episode['episode'], 'episode_index': i,
                           'eligible_windows': episode['rows'] - 15,
                           'consumed_examples': sum(counts), 'unique_window_starts': len(counts)})
        total = sum(e['consumed_examples'] for e in result)
        eligible = sum(e['eligible_windows'] for e in result)
        return {'consumed_examples': total, 'unique_window_starts': len(self.counts),
                'eligible_windows': eligible, 'example_equivalent_passes': total / eligible,
                'unique_window_fraction': len(self.counts) / eligible,
                'episodes_observed': sum(e['consumed_examples'] > 0 for e in result),
                'collated_but_not_consumed': len(self.pending), 'episodes': result,
                'policy': 'Only samples whose training loss was computed count. Prefetched shards and collator lookahead do not.'}


def check_runtime(pipeline, manifest):
    config = pipeline.config
    expected = {'max_steps': STEPS, 'global_batch_size': 1, 'gradient_accumulation_steps': ACCUMULATION,
                'save_steps': 2048, 'save_total_limit': 4, 'save_only_model': True,
                'learning_rate': 0.0001, 'warmup_ratio': 0.05, 'num_gpus': 1,
                'dataloader_num_workers': 0, 'remove_unused_columns': False,
                'resume_from_checkpoint': False, 'skip_weight_loading': False}
    if any(getattr(config.training, key) != value for key, value in expected.items()):
        raise ValueError('Effective larger-dataset training recipe differs')
    if (config.data.seed != 42 or config.data.shard_size != 256 or config.data.episode_sampling_rate != 1.0
            or config.data.num_shards_per_epoch != manifest['num_shards_per_epoch']
            or config.data.override_pretraining_statistics is not True):
        raise ValueError('Training sampling or statistics override differs')
    if Path(config.training.start_from_checkpoint).resolve() != Path(manifest['pretrained']['path']).resolve():
        raise ValueError('Initialization must use the declared pinned pretrained model')
    tuning = {'tune_llm': False, 'tune_visual': False, 'tune_projector': True,
              'tune_diffusion_model': True, 'tune_vlln': True}
    if any(getattr(config.model, key) != value for key, value in tuning.items()):
        raise ValueError('Effective tuning flags differ from validated recipe')
    dropout = {'training_config': config.model.state_dropout_prob, 'model_config': pipeline.model.config.state_dropout_prob,
               'action_head': pipeline.model.action_head.state_dropout_prob, 'processor': pipeline.processor.state_dropout_prob}
    if any(value != 0.0 for value in dropout.values()):
        raise ValueError('Zero state dropout must reach both model and processor')
    mixture = pipeline.train_dataset
    if (pipeline.eval_dataset is not None or len(mixture.datasets) != 1
            or Path(mixture.datasets[0].dataset_path).resolve() != Path(manifest['train_dataset']).resolve()
            or len(mixture.datasets[0]) != manifest['num_shards_per_epoch']):
        raise ValueError('Runtime dataset must contain only the declared training split')
    leaf = mixture.datasets[0]
    if list(map(int, leaf.episode_loader.episode_lengths)) != [e['rows'] for e in manifest['episodes']]:
        raise ValueError('Runtime episode lengths differ from frozen training manifest')
    processor_dir = Path(manifest['output']) / 'audited-processor'
    pipeline.processor.save_pretrained(processor_dir)
    hashes = {name: digest(processor_dir / name) for name in PROCESSOR_FILES}
    if hashes != manifest['processor']['files_sha256']:
        raise ValueError('Runtime processor/statistics differ from frozen train-only bootstrap')
    return {'event': 'runtime', 'passed': True, 'state_dropout_prob': dropout,
            'processor_files_sha256': hashes, 'train_only_dataset': str(leaf.dataset_path),
            'num_shards': len(leaf), 'resume_from_checkpoint': False}


def install_audits(manifest, stream):
    import torch
    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    coverage = CoverageAudit(manifest['episodes'])
    started = time.monotonic()
    counter = 0

    def emit(record):
        record['elapsed_seconds'] = time.monotonic() - started
        stream.write(json.dumps(record, allow_nan=False) + '\n'); stream.flush()

    def finite(value, label):
        try:
            return require_finite(value, label)
        except FloatingPointError:
            emit({'event': 'nonfinite', 'kind': label, 'value': repr(float(value))})
            raise

    original_shard = ShardedSingleStepDataset.get_shard
    def audited_shard(self, index):
        points = original_shard(self, index)
        coverage.register(points, [(int(ep), int(row)) for ep, rows in self.sharded_episodes[index] for row in rows])
        return points
    ShardedSingleStepDataset.get_shard = audited_shard
    original_setup = Gr00tN1d7Pipeline.setup
    def audited_setup(self):
        original_setup(self)
        emit(check_runtime(self, manifest))
        original_collator = self.data_collator
        def collator(features):
            result = original_collator(features)
            coverage.collated(features)
            return result
        self.data_collator = collator
    Gr00tN1d7Pipeline.setup = audited_setup

    class Audits(TrainerCallback):
        def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            scheduler = getattr(lr_scheduler, 'scheduler', lr_scheduler)
            if (state.global_step != 0 or optimizer is None or optimizer.state
                    or scheduler is None or scheduler.last_epoch not in (-1, 0)):
                raise ValueError('Larger-dataset run requires a fresh optimizer/scheduler')
            emit({'event': 'fresh_optimizer', 'passed': True, 'initial_step': state.global_step,
                  'optimizer_state_entries': len(optimizer.state), 'scheduler_epoch': scheduler.last_epoch})

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            if not gradients:
                raise RuntimeError('No parameter gradients')
            norm = torch.linalg.vector_norm(torch.stack(list(torch._foreach_norm(gradients))).float()).item()
            emit({'event': 'gradient', 'step': state.global_step + 1,
                  'post_clip_norm': finite(norm, 'post-clip gradient norm'), 'finite': True})

        def on_step_end(self, args, state, control, optimizer=None, **kwargs):
            emit({'event': 'step_end', 'step': state.global_step,
                  'learning_rate': finite(optimizer.param_groups[0]['lr'], 'learning rate')})

        def on_save(self, args, state, control, **kwargs):
            if state.global_step not in CHECKPOINT_STEPS:
                raise ValueError('Unexpected checkpoint step')
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            hashes = {name: digest(checkpoint / name) for name in PROCESSOR_FILES}
            if hashes != manifest['processor']['files_sha256']:
                raise ValueError('Saved checkpoint processor changed')
            result = {'step': state.global_step, 'checkpoint': str(checkpoint), **coverage.summary()}
            if result['consumed_examples'] != state.global_step * ACCUMULATION:
                raise ValueError('Consumed sample count differs from optimizer progress')
            path = Path(manifest['coverage_prefix'] + f'-{state.global_step}.json')
            write_new_json(path, result)
            emit({'event': 'checkpoint', 'step': state.global_step, 'passed': True,
                  'processor_files_sha256': hashes, 'coverage_path': str(path), 'coverage_sha256': digest(path)})

    original_init, original_loss, original_train = Gr00tTrainer.__init__, Gr00tTrainer.compute_loss, Gr00tTrainer.train
    def audited_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(Audits()); self.args.logging_nan_inf_filter = False
    def audited_loss(self, *args, **kwargs):
        nonlocal counter
        result = original_loss(self, *args, **kwargs)
        loss = result[0] if isinstance(result, tuple) else result
        value = finite(loss.detach().float().item(), 'microbatch loss')
        sample = coverage.consume(); counter += 1
        emit({'event': 'micro_loss', 'microbatch': counter, 'step_before': self.state.global_step,
              'loss': value, 'finite': True, **sample})
        return result
    def audited_train(self, *args, **kwargs):
        # Upstream adds CheckpointFormatCallback after Trainer construction. Its
        # processor/config copies must finish before our on_save verification.
        callbacks = self.callback_handler.callbacks
        self.callback_handler.callbacks = [c for c in callbacks if not isinstance(c, Audits)] + [c for c in callbacks if isinstance(c, Audits)]
        return original_train(self, *args, **kwargs)
    Gr00tTrainer.__init__, Gr00tTrainer.compute_loss = audited_init, audited_loss
    Gr00tTrainer.train = audited_train
    return coverage


def verify_bindings(manifest):
    for path, expected in manifest['input_files_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Bound input changed: ' + path)
    for relative, expected in manifest['script_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Bound execution source changed: ' + relative)


def verify_dataset_inventory(root, expected):
    paths=list(root.rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('Dataset symlinks are not accepted')
    if {p.relative_to(root).as_posix() for p in paths if p.is_file()}!=set(expected):
        raise ValueError('Dataset file inventory changed')


def worker(config, path):
    ensure_python(config, 'gr00t')
    manifest = json.loads(path.read_text())
    verify_bindings(manifest)
    runtime = load_runtime(config)
    expected = dataset_command(config, runtime, Path(manifest['train_dataset']), Path(manifest['output']), manifest['num_shards_per_epoch'])
    if manifest['upstream_command'] != expected:
        raise ValueError('Worker command differs from the declared bounded recipe')
    from evaluate_gr00t_dataset import load_protocol
    protocol = load_protocol(Path(manifest['evaluation_protocol']))
    if protocol['processor'] != manifest['processor'] or protocol['pretrained'] != manifest['pretrained']:
        raise ValueError('Worker initialization or processor binding changed')
    for record in protocol['datasets'].values():
        verify_dataset_inventory(Path(record['path']), record['files_sha256'])
    os.chdir(prepare_offline_workdir(config, runtime))
    audit = guard_path(manifest['audit'], config, 'reports')
    with audit.open('x', buffering=1) as stream:
        install_audits(manifest, stream)
        sys.argv = expected[1:]
        runpy.run_path(expected[1], run_name='__main__')


def checkpoint_results(output, records, processor):
    saved = [r for r in records if r['event'] == 'checkpoint']
    result = []
    for step in CHECKPOINT_STEPS:
        path = output / f'checkpoint-{step}'
        state_path = path / 'trainer_state.json'
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        events = [r for r in saved if r['step'] == step]
        passed = (state.get('global_step') == step and len(events) == 1 and events[0].get('passed') is True
                  and all((path / name).is_file() and digest(path / name) == expected
                          for name, expected in processor['files_sha256'].items()))
        result.append({'step': step, 'checkpoint': str(path), 'passed': passed,
                       'bytes': sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),
                       'coverage': events[0] if events else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    for flag in ['train-dataset', 'train-manifest', 'validation-manifest', 'split-manifest',
                 'evaluation-protocol', 'train-baseline-report', 'validation-baseline-report']:
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--run')
    parser.add_argument('--timeout-seconds', type=int, default=21600)
    parser.add_argument('--dry-run', action='store_true', help='Validate bound inputs and print command; no writes or GPU launch')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(); config = load_config(args.config)
    if args.worker:
        worker(config, resolve_path(args.worker)); return
    for name in ['train_dataset', 'train_manifest', 'validation_manifest', 'split_manifest',
                 'evaluation_protocol', 'train_baseline_report', 'validation_baseline_report', 'run']:
        if not getattr(args, name): parser.error('--' + name.replace('_', '-') + ' is required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run) or not 1 <= args.timeout_seconds <= 21600:
        raise ValueError('Use a new simple run name and a timeout within six hours')
    from evaluate_gr00t_dataset import load_protocol, verify_binding, verify_files
    protocol_path = resolve_path(args.evaluation_protocol); protocol = load_protocol(protocol_path)
    paths = {name: resolve_path(getattr(args, name)) for name in ['train_manifest', 'validation_manifest', 'split_manifest']}
    for name, path in paths.items(): verify_binding(protocol, name, path)
    train = json.loads(paths['train_manifest'].read_text())
    validation = json.loads(paths['validation_manifest'].read_text())
    lengths, shards = validate_split(json.loads(paths['split_manifest'].read_text()), train, validation)
    dataset = guard_derived(args.train_dataset, config)
    if dataset != Path(protocol['datasets']['train']['path']).resolve():
        raise ValueError('Training dataset differs from frozen protocol')
    if any(s.get('split_manifest_sha256') != digest(paths['split_manifest']) for s in [train, validation]):
        raise ValueError('Snapshot split binding changed')
    inputs = {str(path): digest(path) for path in [*paths.values(), protocol_path]}
    for role in ['train', 'validation']:
        verify_files(Path(protocol['datasets'][role]['path']), protocol['datasets'][role]['files_sha256'])
        baseline = resolve_path(getattr(args, role + '_baseline_report'))
        inputs.update(verify_baseline(baseline, role, protocol_path, protocol))
        for relative, expected in protocol['datasets'][role]['files_sha256'].items():
            inputs[str(Path(protocol['datasets'][role]['path']) / relative)] = expected
    for record in protocol['bindings'].values():
        inputs[str(resolve_path(record['path']))] = record['sha256']
    # Preparation hashes these pinned weights; verify them once again in the
    # worker before loading, without repeatedly reading them in CLI dry runs.
    for relative, expected in protocol['pretrained']['files_sha256'].items():
        inputs[str(Path(protocol['pretrained']['path']) / relative)] = expected
    runtime = load_runtime(config)
    model_record = runtime['models']['nvidia/GR00T-N1.7-3B']
    if (Path(protocol['pretrained']['path']).resolve() != Path(model_record['snapshot']).resolve()
            or protocol['pretrained']['revision'] != model_record['revision']):
        raise ValueError('Protocol initialization differs from pinned runtime model')
    groot = runtime['sources']['gr00t']; pins = json.loads((ROOT / 'locks/runtime-sources.json').read_text())
    if subprocess.check_output(['git', '-C', str(groot), 'rev-parse', 'HEAD'], text=True).strip() != pins['Isaac-GR00T']['commit']:
        raise ValueError('GR00T checkout differs from source lock')
    subprocess.run(['git', '-C', str(groot), 'diff', '--quiet', 'HEAD', '--'], check=True)
    output = guard_path(config['paths']['runs'] / args.run, config, 'runs')
    evidence = {suffix: guard_path(config['paths']['reports'] / (args.run + suffix), config, 'reports')
                for suffix in ['-run.json', '-train.json', '-train.log', '-finite.jsonl']}
    coverage_prefix = str(guard_path(config['paths']['reports'] / (args.run + '-coverage'), config, 'reports'))
    coverage_paths = [Path(coverage_prefix + f'-{step}.json') for step in CHECKPOINT_STEPS]
    if any(p.exists() or p.is_symlink() for p in [output, *evidence.values(), *coverage_paths]):
        raise FileExistsError('Run outputs already exist; choose fresh names')
    upstream = ['gr00t/model/gr00t_n1d7/setup.py', 'gr00t/experiment/launch_finetune.py',
                'gr00t/data/dataset/sharded_single_step_dataset.py', 'gr00t/data/dataset/sharded_mixture_dataset.py']
    source_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    inputs.update({str(groot / name): digest(groot / name) for name in upstream})
    inputs.update({str(ROOT / 'locks' / name): digest(ROOT / 'locks' / name) for name in ['runtime-sources.json', 'model-sources.json']})
    manifest = {'schema_version': 1, 'artifact_type': 'larger_dataset_training_run',
                'created_at': datetime.now(timezone.utc).isoformat(), 'run': args.run,
                'repository_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'train_dataset': str(dataset), **{k: str(v) for k, v in paths.items()},
                'episodes': [{key: e[key] for key in ['episode', 'rows', 'derived_episode_index',
                                                    'source_data_json_sha256', 'effective_goal']} for e in train['episodes']],
                'input_files_sha256': inputs, 'script_sha256': source_hashes,
                'evaluation_protocol': str(protocol_path), 'evaluation_protocol_sha256': digest(protocol_path),
                'processor': protocol['processor'], 'pretrained': protocol['pretrained'],
                'output': str(output), 'audit': str(evidence['-finite.jsonl']), 'coverage_prefix': coverage_prefix,
                'checkpoint_steps': CHECKPOINT_STEPS, 'max_steps': STEPS, 'microbatch': 1, 'accumulation': ACCUMULATION,
                'state_dropout_prob': 0.0, 'seed': 42, 'fresh_optimizer_and_scheduler': True,
                'expected_training_examples': STEPS * ACCUMULATION, 'eligible_training_windows': sum(lengths),
                'num_shards_per_epoch': shards, 'timeout_seconds': args.timeout_seconds,
                'upstream_command': dataset_command(config, runtime, dataset, output, shards),
                'resource_requirements': {'startup_output_gib': OUTPUT_START_GIB, 'startup_host_available_gib': 60,
                                          'startup_gpu_free_mib': 40960, 'project_reserve_gib': max(8, config['reserve_gib'])},
                'scope': 'Bounded larger-dataset learning curve. Persistence is descriptive; no robot-success claim.'}
    if args.dry_run:
        print(json.dumps(manifest, indent=2)); return
    subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--error-unmatch', *SOURCE_FILES], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', 'HEAD', '--', 'scripts', 'configs'], check=True)
    gpu_total, gpu_free, _ = gpu_memory()
    reserve = max(8, config['reserve_gib'])
    def resources(startup):
        return check_resources(shutil.disk_usage(ROOT).free, shutil.disk_usage(existing_ancestor(evidence['-run.json'])).free,
                               shutil.disk_usage(existing_ancestor(output)).free, available_memory(),
                               gpu_total, gpu_free, reserve, startup)
    resources(True)
    output.parent.mkdir(parents=True, exist_ok=True); evidence['-run.json'].parent.mkdir(parents=True, exist_ok=True)
    write_new_json(evidence['-run.json'], manifest)
    process = None; aborted = None; started = time.monotonic(); last_progress = 0
    peak_gpu = 0; minimum_host = available_memory(); minimum_output = shutil.disk_usage(output.parent).free
    def interrupted(signum, frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        with evidence['-train.log'].open('x') as log:
            command = [str(runtime['interpreters']['gr00t']), str(Path(__file__).resolve()),
                       '--config', str(config['config_path']), '--worker', str(evidence['-run.json'])]
            process = subprocess.Popen(command, cwd=ROOT, env=runtime_environment(config, offline=True),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                minimum_host = min(minimum_host, available_memory())
                minimum_output = min(minimum_output, shutil.disk_usage(output.parent).free)
                try: resources(False)
                except RuntimeError: aborted = 'resource_reserve'
                if time.monotonic() - started > args.timeout_seconds: aborted = 'wall_time_limit'
                try: peak_gpu = max(peak_gpu, gpu_memory()[2])
                except (subprocess.CalledProcessError, ValueError): pass
                if aborted: stop_process_group(process); break
                if time.monotonic() - last_progress >= 30:
                    records = read_audit(evidence['-finite.jsonl'])
                    steps = [r['step'] for r in records if r['event'] == 'step_end']
                    print(json.dumps({'phase': 'training', 'step': steps[-1] if steps else 0,
                                      'elapsed_seconds': round(time.monotonic()-started, 1),
                                      'peak_gpu_mib': peak_gpu, 'available_host_gib': available_memory()/GIB}), flush=True)
                    last_progress = time.monotonic()
                time.sleep(2)
    except KeyboardInterrupt: aborted = 'interrupted'
    finally:
        if process is not None: stop_process_group(process)
        records = read_audit(evidence['-finite.jsonl']); finite = finite_summary(records, STEPS, ACCUMULATION)
        checkpoints = checkpoint_results(output, records, manifest['processor'])
        audits = {name: [r for r in records if r['event'] == name] for name in ['runtime', 'fresh_optimizer']}
        runtime_passed = all(len(rows) == 1 and rows[0].get('passed') is True for rows in audits.values())
        report = {'schema_version': 1, 'artifact_type': 'larger_dataset_training_report',
                  'passed': process is not None and process.returncode == 0 and not aborted and finite['passed']
                            and runtime_passed and all(c['passed'] for c in checkpoints),
                  'run': args.run, 'run_manifest': str(evidence['-run.json']), 'run_manifest_sha256': digest(evidence['-run.json']),
                  'evaluation_protocol_sha256': manifest['evaluation_protocol_sha256'],
                  'exit_code': process.returncode if process else None, 'aborted': aborted,
                  'elapsed_seconds': time.monotonic()-started, 'sampled_peak_gpu_mib': peak_gpu,
                  'min_available_host_gib': minimum_host/GIB, 'min_output_free_gib': minimum_output/GIB,
                  'finite_audit': finite, 'runtime_audits': audits, 'checkpoints': checkpoints,
                  'finite_audit_sha256': digest(evidence['-finite.jsonl']) if evidence['-finite.jsonl'].exists() else None,
                  'total_output_bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}
        write_new_json(evidence['-train.json'], json_finite(report)); print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
