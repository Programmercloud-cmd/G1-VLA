#!/usr/bin/env python3
"""Freeze and evaluate a new train/validation experiment without changing pilot contracts."""
import argparse
import copy
import dataclasses
from datetime import datetime, timezone
from enum import Enum
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from project_config import PROJECT_ROOT, load_config, resolve_path, guard_path, guard_derived
from convert_single_camera import ensure_python
from run_gr00t_smoke import GIB, available_memory, prepare_offline_workdir
from evaluate_gr00t_pilot import GROUPS, SEEDS, HORIZON, digest, semantic_digest, write_new_json
from compare_pilot_temporal_change import temporal_subset, METHOD as TEMPORAL_METHOD, THRESHOLD

RECIPE = {
    'samples_per_split': 100, 'selection_seed': 42, 'inference_seeds': SEEDS,
    'horizon': HORIZON, 'denoising_steps': 4, 'state_dropout_prob': 0.0,
    'joint_groups': [x[0] for x in GROUPS], 'checkpoint_steps': [2048, 4096, 6144, 8192],
    'normalization_owner': 'train', 'reference': 'pretrained_step_zero_with_training_processor',
    'selection_metric': 'validation/all/group_balanced_normalized_mae',
    'checkpoint_tie_rule': 'lowest_optimizer_step', 'persistence': 'descriptive_only',
    'mandatory_improvement_threshold': None,
}
SAMPLING = {
    'allocation': 'At least one window per eligible episode when episode count <=100. Allocate remaining slots round-robin over a seed-42 permutation, respecting available distinct full windows.',
    'starts': 'For each episode, divide its valid start range into allocated equal index bins and draw one start uniformly per bin using numpy PCG64 seeded42. No predictions, errors, or visual labels enter selection.',
    'order': 'Sort final samples by episode index then start. Train and validation each use a fresh seed42 RNG.',
    'validity': 'Exactly16 targets, finite state/actions, no episode-boundary padding. Insufficient eligible episodes/windows fail preparation.',
    'examples': 'Five episodes at evenly spaced positions in the sorted episode list; first selected window in each; per-group joint with largest target peak-to-peak divided by maximum training normalization span, earliest joint breaks ties.',
}
PROCESSOR_FILES = ['processor_config.json', 'statistics.json', 'embodiment_id.json']
SOURCE_FILES = ['scripts/evaluate_gr00t_dataset.py', 'scripts/evaluate_gr00t_pilot.py',
                'scripts/compare_pilot_temporal_change.py', 'scripts/project_config.py',
                'scripts/run_gr00t_smoke.py', 'configs/g1_inspire_config.py',
                'scripts/archive_gr00t_dataset.py',
                'scripts/run_gr00t_dataset.py', 'scripts/run_gr00t_pilot.py',
                'scripts/archive_smoke_checkpoint.py', 'scripts/prepare_pilot.py',
                'scripts/prepare_dataset_split.py', 'scripts/convert_single_camera.py',
                'scripts/prepare_gr00t_dataset.py',
                'experiments/larger-dataset-v1/evaluation-design.json',
                'vendor/Isaac-GR00T/gr00t/policy/gr00t_policy.py',
                'vendor/Isaac-GR00T/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py',
                'vendor/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py',
                'vendor/Isaac-GR00T/gr00t/model/gr00t_n1d7/setup.py',
                'vendor/Isaac-GR00T/gr00t/data/dataset/lerobot_episode_loader.py',
                'vendor/Isaac-GR00T/gr00t/data/dataset/sharded_mixture_dataset.py']


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    name = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    return {'path': name, 'sha256': digest(path)}


def verify_binding(protocol, name, path):
    require(digest(path) == protocol['bindings'][name]['sha256'], 'Protocol binding differs: ' + name)


def validate_conversion(conversion_path, split_path, manifests, roots, inventories):
    """Bind prepared inputs to the converter's exact source/value verification."""
    conversion = read(conversion_path)
    require(conversion.get('schema_version') == 1 and conversion.get('artifact_type') == 'larger_dataset_conversion',
            'Unsupported conversion manifest')
    require(set(conversion.get('splits', {})) == {'train', 'validation'}, 'Conversion split inventory differs')

    def check_binding(item, path, label):
        require(resolve_path(item['path']) == Path(path).resolve() and item['sha256'] == digest(path),
                'Conversion binding differs: ' + label)

    check_binding(conversion['split_manifest'], split_path, 'split_manifest')
    for name in ['train', 'validation']:
        part = conversion['splits'][name]
        require(resolve_path(part['dataset_path']) == Path(roots[name]).resolve(),
                'Conversion dataset path differs: ' + name)
        check_binding(part['snapshot_manifest'], manifests[name], name + '_manifest')
        require(bool(inventories[name]) and part['files_sha256'] == inventories[name],
                'Conversion dataset inventory differs: ' + name)
    statistics = conversion['statistics']
    require(statistics.get('owner') == 'train', 'Conversion statistics must belong to training split')
    for name, filename in [('stats_json', 'stats.json'), ('relative_stats_json', 'relative_stats.json')]:
        check_binding(statistics[name], Path(roots['train']) / 'meta' / filename, name)
        require(inventories['train'].get('meta/' + filename) == statistics[name]['sha256'],
                'Conversion statistics inventory differs: ' + name)
    return conversion


def load_protocol(path):
    """CPU/stdlib preflight; hashes small bindings, processor and samples, not model shards."""
    value = read(path)
    require(value.get('schema_version') == 1 and value.get('artifact_type') == 'larger_dataset_evaluation_protocol', 'Unsupported evaluation protocol')
    require(value.get('recipe') == RECIPE and value.get('sampling') == SAMPLING, 'Evaluation recipe or sampling changed')
    require(value.get('target_change') == {'method': TEMPORAL_METHOD, 'threshold': THRESHOLD}, 'Target-change definition changed')
    require(set(value['bindings']) == {'train_manifest','validation_manifest','split_manifest','conversion_manifest','train_statistics','train_relative_statistics'}, 'Missing or extra protocol data binding')
    require(value['processor'].get('statistics_owner') == 'train', 'Normalization must belong to training split')
    require(set(value['source_sha256']) == set(SOURCE_FILES), 'Source identity is incomplete')
    train_root = resolve_path(value['datasets']['train']['path'])
    require(train_root != resolve_path(value['datasets']['validation']['path']), 'Split roots overlap')
    for name, filename in [('train_statistics','stats.json'),('train_relative_statistics','relative_stats.json')]:
        item=value['bindings'][name]
        require(resolve_path(item['path']) == train_root/'meta'/filename, 'Normalization is not from training dataset')
        require(value['datasets']['train']['files_sha256'].get('meta/'+filename) == item['sha256'], 'Training normalization inventory differs')
    semantic_file=value['processor']['semantics_file']
    semantic_path=resolve_path(semantic_file['path'])
    require(digest(semantic_path)==semantic_file['sha256'] and semantic_digest(read(semantic_path)) == value['processor']['semantics_sha256'], 'Processor semantic identity is inconsistent')
    for key, item in value['bindings'].items():
        verify_binding(value, key, resolve_path(item['path']))
    validate_conversion(resolve_path(value['bindings']['conversion_manifest']['path']),
                        resolve_path(value['bindings']['split_manifest']['path']),
                        {name: resolve_path(value['bindings'][name + '_manifest']['path']) for name in ['train', 'validation']},
                        {name: resolve_path(value['datasets'][name]['path']) for name in ['train', 'validation']},
                        {name: value['datasets'][name]['files_sha256'] for name in ['train', 'validation']})
    for key, sha in value['source_sha256'].items():
        require(digest(PROJECT_ROOT / key) == sha, 'Evaluation source changed: ' + key)
    for name, sha in value['processor']['files_sha256'].items():
        require(digest(resolve_path(value['processor']['path']) / name) == sha, 'Bootstrap processor changed: ' + name)
    require(set(value['processor']['files_sha256']) == set(PROCESSOR_FILES), 'Incomplete bootstrap processor')
    for split in ['train', 'validation']:
        sample = value['samples'][split]
        require(digest(resolve_path(sample['path'])) == sample['sha256'], 'Frozen samples changed')
        require(digest(resolve_path(sample['arrays']['path'])) == sample['arrays']['sha256'], 'Frozen target arrays changed')
    return value


def file_inventory(path):
    result = {}
    for item in sorted(Path(path).rglob('*')):
        require(not item.is_symlink(), 'Dataset symlink is not allowed: ' + str(item))
        if item.is_file():
            result[str(item.relative_to(path))] = digest(item)
    require(bool(result), 'Empty dataset')
    return result


def verify_files(root, expected):
    require(file_inventory(root) == expected, 'Dataset inventory changed: ' + str(root))


def select_windows(episodes, count=100, seed=42):
    """Model-independent selection using only valid target sequence lengths."""
    import numpy as np
    require(bool(episodes), 'No episodes')
    require([x['episode_index'] for x in episodes] == list(range(len(episodes))), 'Noncontiguous episode indices')
    capacities = [int(x['length']) - HORIZON + 1 for x in episodes]
    require(all(x > 0 for x in capacities), 'Every episode must have a complete target horizon')
    require(sum(capacities) >= count, 'Insufficient distinct complete windows')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes)).tolist()
    allocated = [0] * len(episodes)
    remaining = count
    while remaining:
        for i in order:
            if allocated[i] < capacities[i]:
                allocated[i] += 1
                remaining -= 1
                if not remaining:
                    break
    result = []
    for i, episode in enumerate(episodes):
        n = allocated[i]
        for k in range(n):
            low, high = k * capacities[i] // n, (k + 1) * capacities[i] // n
            start = int(rng.integers(low, high))
            result.append({'episode_index': i, 'source_episode': episode['source_episode'],
                           'start': start, 'length': int(episode['length']),
                           'source_row_offset': start, 'progress': start / max(1, capacities[i] - 1)})
    require(len({(x['episode_index'], x['start']) for x in result}) == count, 'Duplicate selected windows')
    return result


def validate_split(split, manifests):
    require(split.get('artifact_type') == 'frozen_dataset_split' and split.get('seed') == 42, 'Expected frozen seed42 split')
    train, validation = split['splits']['train'], split['splits']['validation']
    require(len(set(train)) == len(train) and len(set(validation)) == len(validation), 'Repeated split episodes')
    require(not set(train) & set(validation), 'Train/validation overlap')
    for name in ['train', 'validation']:
        rows = manifests[name]['episodes']
        require([r['episode'] for r in rows] == split['splits'][name], 'Snapshot membership/order differs from split')
        require(all(r['rows'] >= HORIZON for r in rows), 'Incomplete source horizon')
        index = {r['episode']: r for r in split['episodes'] if r['split'] == name}
        require(set(index) == set(split['splits'][name]), 'Split episode records differ')
        for i, row in enumerate(rows):
            item = index[row['episode']]
            require(item['derived_episode_index'] == i and item['rows'] == row['rows'], 'Split indices/lengths differ')


def jsonable(value):
    if dataclasses.is_dataclass(value): return jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum): return value.value
    if hasattr(value, 'model_dump'): return jsonable(value.model_dump(mode='json'))
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'): return value.tolist()
    return value


def processor_semantics(processor, root):
    import numpy as np
    from gr00t.data.embodiment_tags import EmbodimentTag
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    modalities = processor.get_modality_configs()[tag]
    sa = processor.state_action_processor
    require(modalities['action'].modality_keys == RECIPE['joint_groups'] and modalities['state'].modality_keys == RECIPE['joint_groups'], 'Processor joint group order differs')
    require(list(modalities['action'].delta_indices) == list(range(HORIZON)), 'Processor horizon differs')
    params = sa.norm_params[tag]['action']
    lower = np.concatenate([np.broadcast_to(params[key]['min'], (HORIZON, e-b)) for key,b,e,_ in GROUPS], axis=-1)
    upper = np.concatenate([np.broadcast_to(params[key]['max'], (HORIZON, e-b)) for key,b,e,_ in GROUPS], axis=-1)
    value = {'processor_config': read(root / 'processor_config.json'), 'embodiment_id': read(root / 'embodiment_id.json'),
             'modalities': jsonable(modalities), 'statistics': jsonable(sa.statistics[tag]),
             'use_relative_action': sa.use_relative_action, 'use_percentiles': sa.use_percentiles,
             'clip_outliers': sa.clip_outliers, 'apply_sincos_state_encoding': sa.apply_sincos_state_encoding,
             'lower': lower.tolist(), 'upper': upper.tolist()}
    require(processor.state_dropout_prob == 0.0, 'Processor state dropout is not zero')
    require(np.isfinite(lower).all() and np.isfinite(upper).all() and (upper >= lower).all(), 'Invalid training normalization')
    require(all((~np.isclose(lower[:,b:e], upper[:,b:e])).any() for _,b,e,_ in GROUPS), 'Group lacks active normalization cells')
    return value, lower, upper


def bootstrap_processor(pretrained, train_dataset, output):
    """Match pinned launcher processor overrides and single-training-dataset stats merge."""
    import gr00t.model  # Registers AutoProcessor; no model weights are loaded.
    from transformers import AutoProcessor
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics
    module_name = '_larger_g1_modality'
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / 'configs/g1_inspire_config.py')
        module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module
        spec.loader.exec_module(module)
    module = sys.modules[module_name]
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    m = get_default_config().model
    # These are the default launch_finetune overrides of the unchanged validated recipe.
    processor = AutoProcessor.from_pretrained(str(pretrained), local_files_only=True, trust_remote_code=True,
        modality_configs={tag: module.g1_inspire_config}, use_percentiles=m.use_percentiles,
        image_crop_size=m.image_crop_size, image_target_size=m.image_target_size,
        random_rotation_angle=None, color_jitter_params=None,
        model_name='nvidia/Cosmos-Reason2-2B', model_type=m.backbone_model_type,
        formalize_language=m.formalize_language, apply_sincos_state_encoding=m.apply_sincos_state_encoding,
        max_action_horizon=m.action_horizon, use_albumentations=m.use_albumentations_transforms,
        extra_augmentation_config=None, shortest_image_edge=m.shortest_image_edge,
        crop_fraction=m.crop_fraction, letter_box_transform=m.letter_box_transform,
        transformers_loading_kwargs={'trust_remote_code':True,'local_files_only':True},
        use_alternate_vl_dit=m.use_alternate_vl_dit, use_relative_action=True,
        exclude_state=m.exclude_state, state_dropout_prob=0.0, use_mean_std=m.use_mean_std)
    require(tag not in processor.state_action_processor.statistics, 'Pretrained source already has NEW_EMBODIMENT statistics')
    loader = LeRobotEpisodeLoader(dataset_path=train_dataset, modality_configs=module.g1_inspire_config)
    stats = loader.get_dataset_statistics()
    merged = {key: merge_statistics([stats[key]], [1.0], is_relative_stats=key == 'relative_action')
              for key in ['state', 'action', 'relative_action']}
    processor.set_statistics({tag: merged}, override=False)
    processor.eval()
    processor.save_pretrained(output)
    return processor


def prepare(args, config):
    import numpy as np
    import pyarrow.parquet as pq
    runtime = ensure_python(config, 'gr00t')
    roots = {name: guard_derived(getattr(args, name + '_dataset'), config) for name in ['train', 'validation']}
    manifests = {name: resolve_path(getattr(args, name + '_manifest')) for name in roots}
    split_path = resolve_path(args.split_manifest)
    split = read(split_path)
    manifest_data = {k: read(v) for k,v in manifests.items()}
    validate_split(split, manifest_data)
    require(roots['train'] != roots['validation'] and not roots['train'].is_relative_to(roots['validation']) and not roots['validation'].is_relative_to(roots['train']), 'Split roots overlap')
    conversion_path = resolve_path(args.conversion_manifest)
    inventories = {k: file_inventory(v) for k,v in roots.items()}
    validate_conversion(conversion_path, split_path, manifests, roots, inventories)
    conversion_binding = binding(conversion_path)
    protocol_path = resolve_path(args.protocol)
    processor_root = guard_derived(args.processor_output, config)
    require(not protocol_path.exists() and not processor_root.exists(), 'Choose new protocol/processor output paths')
    os.chdir(prepare_offline_workdir(config, runtime))
    pretrained = Path(runtime['models']['nvidia/GR00T-N1.7-3B']['snapshot']).resolve()
    for filename in ['stats.json', 'relative_stats.json']:
        require((roots['train'] / 'meta' / filename).is_file(), 'Compute training statistics before bootstrap')
    processor = bootstrap_processor(pretrained, roots['train'], processor_root)
    semantics, lower, upper = processor_semantics(processor, processor_root)
    semantics_path=processor_root/'evaluation-semantics.json'
    write_new_json(semantics_path,semantics)
    bindings = {name + '_manifest': binding(path) for name,path in manifests.items()}
    bindings.update(split_manifest=binding(split_path), conversion_manifest=conversion_binding,
                    train_statistics=binding(roots['train'] / 'meta/stats.json'),
                    train_relative_statistics=binding(roots['train'] / 'meta/relative_stats.json'))
    frozen_samples = {}
    for name, root in roots.items():
        metadata = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line]
        source_rows = manifest_data[name]['episodes']
        require(len(metadata) == len(source_rows), 'Converted episode count differs')
        episodes = []
        arrays = {}
        source_details = {}
        for i, (meta, source) in enumerate(zip(metadata, source_rows)):
            require(meta['episode_index'] == i and meta['length'] == source['rows'], 'Converted episode mapping differs')
            episodes.append({'episode_index': i, 'source_episode': source['episode'], 'length': source['rows']})
            source_json=manifests[name].parent/'raw/pick_up_black_box'/source['episode']/'data.json'
            require(digest(source_json)==source['files_sha256']['data.json'],'Snapshot source metadata changed')
            source_document=read(source_json)
            require(source_document['text']['goal']==source['goal'],'Snapshot effective task differs')
            source_details[i]={'indices':[row['idx'] for row in source_document['data']],
                               'goal':source['goal'],'data_json':binding(source_json)}
            table = pq.read_table(root / f'data/chunk-000/episode_{i:06d}.parquet')
            action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
            state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
            require(action.shape == state.shape == (source['rows'], 26), 'State/action shape differs')
            require(np.isfinite(action).all() and np.isfinite(state).all(), 'Nonfinite source data')
            require(table['frame_index'].to_pylist() == list(range(source['rows'])), 'Converted row order differs')
            arrays[i] = (action, state)
        samples = select_windows(episodes)
        for sample in samples:
            detail=source_details[sample['episode_index']];start=sample['start']
            sample.update(source_indices=detail['indices'][start:start+HORIZON],effective_goal=detail['goal'],source_data_json=detail['data_json'])
        target = np.asarray([arrays[x['episode_index']][0][x['start']:x['start']+HORIZON] for x in samples])
        state = np.asarray([arrays[x['episode_index']][1][x['start']] for x in samples])
        changing, scores, _ = temporal_subset(target, lower, upper)
        examples = []
        for ep in np.linspace(0, len(episodes)-1, min(5, len(episodes)), dtype=int):
            indices = [i for i,x in enumerate(samples) if x['episode_index'] == int(ep)]
            if not indices: continue
            index = indices[0]; span = np.max(upper-lower, axis=0)
            movement = np.divide(np.ptp(target[index], axis=0), span, out=np.zeros(26), where=span>0)
            examples.append({'sample_index': index, 'joint_indices': [b+int(np.argmax(movement[b:e])) for _,b,e,_ in GROUPS]})
        sample_path = protocol_path.parent / (name + '-samples.json')
        arrays_path = sample_path.with_suffix('.npz')
        require(not sample_path.exists() and not arrays_path.exists(), 'Sample outputs already exist')
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_path, targets=target, states=state, lower=lower, upper=upper,
            changing_target_mask=changing, target_change_scores=scores, seeds=np.asarray(SEEDS),
            episode_indices=np.asarray([x['episode_index'] for x in samples]), starts=np.asarray([x['start'] for x in samples]))
        sample_record = {'schema_version': 1, 'split': name, 'samples': samples, 'examples': examples,
                         'dataset_manifest_sha256': digest(manifests[name]), 'split_manifest_sha256': digest(split_path),
                         'arrays_sha256': digest(arrays_path), 'changing_target_mask': changing.tolist(),
                         'selection': SAMPLING, 'seeds': SEEDS, 'horizon': HORIZON}
        write_new_json(sample_path, sample_record)
        frozen_samples[name] = dict(binding(sample_path), arrays=binding(arrays_path))
    # Verify preparation did not mutate either converted split.
    for name, root in roots.items(): verify_files(root, inventories[name])
    require(digest(conversion_path) == conversion_binding['sha256'], 'Conversion manifest changed during preparation')
    validate_conversion(conversion_path, split_path, manifests, roots, inventories)
    model_files = {p.name: digest(p) for p in sorted(pretrained.iterdir())
                   if p.is_file() and (p.name in ['config.json', 'model.safetensors.index.json'] or p.suffix == '.safetensors')}
    require(any(x.endswith('.safetensors') for x in model_files), 'Missing pretrained model weights')
    protocol = {'schema_version': 1, 'artifact_type': 'larger_dataset_evaluation_protocol', 'status': 'frozen_before_step_zero',
        'recipe': RECIPE, 'sampling': SAMPLING, 'target_change': {'method': TEMPORAL_METHOD, 'threshold': THRESHOLD},
        'bindings': bindings, 'source_sha256': {p: digest(PROJECT_ROOT/p) for p in SOURCE_FILES},
        'datasets': {name: {'path': str(root), 'files_sha256': inventories[name], 'episode_count': len(manifest_data[name]['episodes'])} for name,root in roots.items()},
        'processor': {'path': str(processor_root), 'files_sha256': {x: digest(processor_root/x) for x in PROCESSOR_FILES},
                      'semantics_file': binding(semantics_path), 'semantics_sha256': semantic_digest(semantics), 'statistics_owner': 'train'},
        'pretrained': {'path': str(pretrained), 'revision': runtime['models']['nvidia/GR00T-N1.7-3B']['revision'], 'files_sha256': model_files},
        'samples': frozen_samples,
        'comparison_semantics': 'Native and normalized MAE/RMSE by group, target-change subset, and episode; step zero and persistence descriptive, no mandatory improvement gate. Empty subsets unavailable. Nonfinite or inconsistent evidence invalid.',
        'checkpoint_selection': 'Among all four valid declared checkpoints, minimize the arithmetic mean of four validation all-window normalized MAEs. Equal scores choose earliest step. Validation is used for selection, not a final untouched test set.'}
    write_new_json(protocol_path, protocol)
    print(json.dumps({'protocol': str(protocol_path), 'sha256': digest(protocol_path), 'samples_per_split': 100, 'processor': str(processor_root)}), flush=True)


def summarize_arrays(prediction, target, state, lower, upper, changing, reference=None):
    """Descriptive metrics only; empty subsets remain explicitly unavailable."""
    import numpy as np
    prediction, target, state, lower, upper = [np.asarray(x, dtype=np.float64) for x in (prediction, target, state, lower, upper)]
    require(prediction.shape == (len(target), 3, HORIZON, 26) and target.shape == (len(target), HORIZON, 26), 'Invalid prediction/target shape')
    require(state.shape == (len(target),26) and np.asarray(changing).shape == (len(target),4), 'Invalid state/subset shape')
    require(all(np.isfinite(x).all() for x in (prediction,target,state,lower,upper)), 'Nonfinite evaluation evidence')
    require(lower.shape == upper.shape == (HORIZON,26) and (upper>=lower).all(), 'Invalid normalization bounds')
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)
        require(reference.shape == prediction.shape and np.isfinite(reference).all(), 'Invalid step-zero reference predictions')
    active = ~np.isclose(lower,upper); safe = np.where(active,upper-lower,1)
    result = {}
    for gi,(name,b,e,unit) in enumerate(GROUPS):
        result[name] = {'unit':unit}
        for subset,mask in [('all',np.ones(len(target),bool)),('changing_target',changing[:,gi]),('small_target_change',~changing[:,gi])]:
            count = int(np.sum(mask))
            def metric(values):
                selected = values[mask,:,:,b:e]
                normalized = (2*selected/safe[None,None,:,b:e])[:,:,active[:,b:e]]
                return {'sample_count':count,
                        'mae':float(np.abs(selected).mean()) if count else None,
                        'rmse':float(np.sqrt(np.square(selected).mean())) if count else None,
                        'normalized_mae':float(np.abs(normalized).mean()) if normalized.size else None,
                        'normalized_rmse':float(np.sqrt(np.square(normalized).mean())) if normalized.size else None,
                        'per_seed_mae':[float(np.abs(selected[:,s]).mean()) for s in range(3)] if count else []}
            data = {'model':metric(prediction-target[:,None]),
                    'persistence':metric(np.broadcast_to(state[:,None,None,:],prediction.shape)-target[:,None])}
            if reference is not None:
                data['step_zero'] = metric(reference-target[:,None])
                data['change_from_step_zero'] = {
                    key:data['model'][key]-data['step_zero'][key] if data['model'][key] is not None else None
                    for key in ['mae','rmse','normalized_mae','normalized_rmse']}
            result[name][subset] = data
    return result


def selection_score(groups):
    values = [groups[name]['all']['model']['normalized_mae'] for name,_,_,_ in GROUPS]
    require(all(x is not None and math.isfinite(x) and x>=0 for x in values), 'Checkpoint selection requires four finite normalized group MAEs')
    return sum(values)/4


def read_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as stream:
        return {key:stream[key].copy() for key in stream.files}


def load_completed(path, protocol, protocol_sha, split, step_zero=False):
    import numpy as np
    report = read(path); arrays_path = Path(path).with_suffix('.npz')
    require(report.get('passed') is True and report.get('evaluation_protocol_sha256') == protocol_sha, 'Incomplete or unrelated evaluation')
    require(report.get('split') == split and report.get('sample_manifest_sha256') == protocol['samples'][split]['sha256'], 'Wrong evaluation split/samples')
    require(report.get('processor_semantics_sha256') == protocol['processor']['semantics_sha256'], 'Evaluation processor differs')
    require(report.get('statistics_sha256') == protocol['processor']['files_sha256']['statistics.json'], 'Evaluation statistics differ')
    require(report.get('recipe') == RECIPE and report.get('evaluation_mode') == {'model_training':False,'processor_training':False}, 'Evaluation recipe/mode differs')
    require(report.get('artifacts',{}).get(arrays_path.name) == digest(arrays_path), 'Saved prediction artifact changed')
    if step_zero:
        require(report.get('role') == 'step_zero_reference' and report.get('checkpoint_global_step') == 0, 'Expected pretrained step-zero reference')
    else:
        require(report.get('role') == 'checkpoint' and report.get('checkpoint_global_step') in RECIPE['checkpoint_steps'], 'Unexpected checkpoint step')
    arrays = read_npz(arrays_path)
    frozen = read_npz(resolve_path(protocol['samples'][split]['arrays']['path']))
    for key,value in frozen.items():
        require(key in arrays and np.array_equal(value,arrays[key]), 'Evaluation target identity differs: '+key)
    prediction = arrays['absolute_predictions']
    require(prediction.shape == (100,3,HORIZON,26) and np.isfinite(prediction).all(), 'Invalid completed prediction arrays')
    groups = summarize_arrays(prediction,arrays['targets'],arrays['states'],arrays['lower'],arrays['upper'],arrays['changing_target_mask'])
    require(math.isclose(selection_score(groups),report['group_balanced_normalized_mae'],rel_tol=1e-12,abs_tol=1e-12), 'Reported selection score differs from saved predictions')
    return report,arrays


def load_policy(model_path, processor_path, step_zero):
    """Pinned Gr00tPolicy inference methods with an explicitly bound processor directory.

    Step zero has no NEW_EMBODIMENT processor at the pretrained model root, so
    reproduce its constructor wiring while loading the independently frozen processor.
    """
    import torch
    import gr00t.model
    from transformers import AutoModel, AutoProcessor
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.policy import BasePolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    overrides = dict(tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,
                     tune_vlln=True,state_dropout_prob=0.0,backbone_trainable_params_fp32=True,load_bf16=False) if step_zero else {}
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    BasePolicy.__init__(policy,strict=True)
    model, loading = AutoModel.from_pretrained(str(model_path),local_files_only=True,output_loading_info=True,**overrides)
    require(not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']), 'Model load has missing/unexpected/mismatched tensors')
    model.eval(); model.to(device='cuda:0',dtype=torch.bfloat16)
    model.action_head.num_inference_timesteps = 4
    processor = AutoProcessor.from_pretrained(str(processor_path),local_files_only=True); processor.eval()
    policy.model=model; policy.processor=processor; policy.embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
    policy.modality_configs={k:v for k,v in processor.get_modality_configs()[policy.embodiment_tag.value].items() if k!='rl_info'}
    policy.collate_fn=processor.collator
    language=policy.modality_configs['language']
    require(len(language.modality_keys)==1 and len(language.delta_indices)==1,'Unexpected language modality')
    policy.language_key=language.modality_keys[0]
    require(processor.state_dropout_prob==model.action_head.state_dropout_prob==model.config.state_dropout_prob==0.0,'State dropout differs from zero')
    return policy


def shared_policy(shared, model_path, processor_path, step_zero, protocol_sha):
    identity=(str(model_path),str(processor_path),bool(step_zero),protocol_sha)
    if 'policy' in shared:
        require(shared['policy_identity']==identity,'Cannot reuse policy across different checkpoints/processors/protocols')
    else:
        shared['policy']=load_policy(model_path,processor_path,step_zero)
        shared['policy_identity']=identity
    return shared['policy']


def checkpoint_evidence(args, protocol, model_path, shared):
    """Hash local pretrained weights once; use the fresh remote-host checkpoint inventory otherwise."""
    if args.step_zero:
        expected=protocol['pretrained']['files_sha256']
        signature={name:(int((model_path/name).stat().st_size),int((model_path/name).stat().st_mtime_ns)) for name in expected}
        if 'pretrained_signature' not in shared:
            for name,sha in expected.items():
                require(digest(model_path/name)==sha,'Pretrained model source changed: '+name)
            shared['pretrained_signature']=signature
        require(signature==shared['pretrained_signature'],'Pretrained files changed during paired evaluation')
        return dict(expected),None
    require(args.checkpoint_inventory is not None,'Checkpoint evaluation requires a fresh durable --checkpoint-inventory')
    from archive_gr00t_dataset import load_checkpoint_inventory
    inventory_path=resolve_path(args.checkpoint_inventory)
    inventory=load_checkpoint_inventory(inventory_path,model_path)
    inventory_sha=digest(inventory_path)
    if 'checkpoint_inventory_sha256' in shared:
        require(shared['checkpoint_inventory_sha256']==inventory_sha,'Checkpoint inventory changed during paired evaluation')
    shared['checkpoint_inventory_sha256']=inventory_sha
    return {row['path']:row['sha256'] for row in inventory['files']},binding(inventory_path)


def model_semantics(config):
    # These fields describe serialization/loading location and precision metadata.
    # Both inference paths explicitly cast the complete model to bfloat16.
    excluded = {'_name_or_path','torch_dtype','dtype','transformers_version'}
    return {key:value for key,value in config.items() if key not in excluded}


def plot_examples(path, sample_record, prediction, target, state, reference):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples=sample_record['examples']
    fig,axes=plt.subplots(len(examples),4,figsize=(17,3*len(examples)),squeeze=False)
    for row,example in enumerate(examples):
        index=example['sample_index']; sample=sample_record['samples'][index]
        for col,((name,_,_,_),joint) in enumerate(zip(GROUPS,example['joint_indices'])):
            ax=axes[row,col]; x=np.arange(HORIZON); draws=prediction[index,:,:,joint]
            mean,std=draws.mean(axis=0),draws.std(axis=0)
            ax.plot(x,target[index,:,joint],color='black',label='target')
            ax.plot(x,mean,label='model'); ax.fill_between(x,mean-std,mean+std,alpha=.2)
            ax.axhline(state[index,joint],color='gray',linestyle=':',label='current-state persistence')
            if reference is not None:ax.plot(x,reference[index,:,:,joint].mean(axis=0),label='step zero')
            ax.set_title(f"{sample['source_episode']} start {sample['start']} / {name}[{joint}]")
            ax.set_xlabel('target horizon offset')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Fixed target-selected examples; mean ± standard deviation over three fixed seeds')
    fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)


def evaluate(args, config, shared=None):
    import numpy as np
    if shared is None:shared={}
    protocol_path=resolve_path(args.protocol); protocol=load_protocol(protocol_path); protocol_sha=digest(protocol_path)
    split=args.split; sample_record=read(resolve_path(protocol['samples'][split]['path']))
    frozen=read_npz(resolve_path(protocol['samples'][split]['arrays']['path']))
    require(args.report is not None,'Single-split evaluation requires --report')
    report_path=guard_path(args.report,config,'reports')
    for path in [report_path,report_path.with_suffix('.npz'),report_path.with_suffix('.png')]:
        require(not path.exists() and not path.is_symlink(),'Evaluation outputs already exist')
    require(bool(args.step_zero) != bool(args.checkpoint),'Choose either --step-zero or --checkpoint')
    require(bool(args.reference) == (not args.step_zero),'Checkpoint evaluation requires same-split step-zero --reference')
    model_path=resolve_path(protocol['pretrained']['path']) if args.step_zero else resolve_path(args.checkpoint)
    processor_path=resolve_path(protocol['processor']['path']) if args.step_zero else model_path
    if not (processor_path/'processor_config.json').exists():processor_path=processor_path/'processor'
    for name,sha in protocol['processor']['files_sha256'].items():
        require(digest(processor_path/name)==sha,'Checkpoint processor differs from training bootstrap: '+name)
    checkpoint_files,inventory_binding=checkpoint_evidence(args,protocol,model_path,shared)
    if args.step_zero:
        step=0
    else:
        step=read(model_path/'trainer_state.json')['global_step']
        require(step in RECIPE['checkpoint_steps'],'Checkpoint is not on the declared schedule')
    root=resolve_path(protocol['datasets'][split]['path']); verify_files(root,protocol['datasets'][split]['files_sha256'])
    reference_path=resolve_path(args.reference) if args.reference else None
    reference_report,reference_arrays=load_completed(reference_path,protocol,protocol_sha,split,True) if reference_path else (None,None)
    reserve=config['reserve_gib']*GIB
    require(available_memory()>=40*GIB and shutil.disk_usage(PROJECT_ROOT).free>=reserve,'Insufficient evaluation host/disk reserve')
    runtime=ensure_python(config,'gr00t'); os.chdir(prepare_offline_workdir(config,runtime))
    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    require(torch.cuda.is_available() and ('policy' in shared or torch.cuda.mem_get_info(0)[0]>=16*GIB),'Insufficient evaluation VRAM')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); torch.manual_seed(42)
    reused_policy='policy' in shared
    policy=shared_policy(shared,model_path,processor_path,args.step_zero,protocol_sha)
    semantics,lower,upper=processor_semantics(policy.processor,processor_path)
    require(semantic_digest(semantics)==protocol['processor']['semantics_sha256'],'Runtime processor differs from bootstrap')
    require(np.array_equal(lower,frozen['lower']) and np.array_equal(upper,frozen['upper']),'Runtime normalization differs')
    model_config=jsonable(policy.model.config.to_dict()); model_semantics_sha=semantic_digest(model_semantics(model_config))
    if reference_report:
        require(reference_report['model_semantics_sha256']==model_semantics_sha,'Model configuration changed beyond loading metadata')
    loader=LeRobotEpisodeLoader(dataset_path=root,modality_configs=policy.modality_configs)
    predictions=[]; cached=None; trajectory=None
    for index,sample in enumerate(sample_record['samples']):
        require(available_memory()>=16*GIB and shutil.disk_usage(report_path.parent).free>=reserve,'Evaluation resource reserve crossed')
        if sample['episode_index']!=cached:
            trajectory=loader[sample['episode_index']]; cached=sample['episode_index']
        data=extract_step_data(trajectory,sample['start'],policy.modality_configs,EmbodimentTag.NEW_EMBODIMENT)
        require(data.text==sample['effective_goal'],'Loaded instruction differs from frozen accepted goal')
        target=np.concatenate([data.actions[key] for key in RECIPE['joint_groups']],axis=-1).astype(np.float32)
        state=np.concatenate([data.states[key][-1] for key in RECIPE['joint_groups']],axis=-1).astype(np.float32)
        require(np.array_equal(target,frozen['targets'][index]) and np.array_equal(state,frozen['states'][index]),'Loaded target/state differs from frozen arrays')
        observation={'video':{k:np.asarray(v,dtype=np.uint8)[None] for k,v in data.images.items()},
                     'state':{k:np.asarray(v,dtype=np.float32)[None] for k,v in data.states.items()},
                     'language':{policy.language_key:[[data.text]]}}
        draws=[]
        for seed in SEEDS:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with torch.inference_mode(): action,_=policy.get_action(observation)
            joined=np.concatenate([action[key][0] for key in RECIPE['joint_groups']],axis=-1).astype(np.float32)
            require(joined.shape==(HORIZON,26) and np.isfinite(joined).all(),'Invalid action prediction')
            draws.append(joined)
        predictions.append(draws)
        if (index+1)%10==0:print(json.dumps({'phase':'evaluation','split':split,'step':step,'completed_samples':index+1,'total_samples':100}),flush=True)
    prediction=np.asarray(predictions)
    reference_prediction=reference_arrays['absolute_predictions'] if reference_arrays else None
    groups=summarize_arrays(prediction,frozen['targets'],frozen['states'],lower,upper,frozen['changing_target_mask'],reference_prediction)
    episodes={}
    for episode in sorted(set(frozen['episode_indices'].tolist())):
        mask=frozen['episode_indices']==episode
        episodes[str(episode)]=summarize_arrays(prediction[mask],frozen['targets'][mask],frozen['states'][mask],lower,upper,frozen['changing_target_mask'][mask],reference_prediction[mask] if reference_prediction is not None else None)
    require(not policy.model.training and not policy.processor.training,'Evaluation switched to training mode')
    require(digest(protocol_path)==protocol_sha,'Protocol changed during evaluation')
    load_protocol(protocol_path)  # Recheck the source, processor and sample bindings before publishing.
    after_files,after_inventory=checkpoint_evidence(args,protocol,model_path,shared)
    require(after_files==checkpoint_files and after_inventory==inventory_binding,'Checkpoint identity changed during inference')
    np.savez_compressed(report_path.with_suffix('.npz'),absolute_predictions=prediction,**frozen)
    plot_examples(report_path.with_suffix('.png'),sample_record,prediction,frozen['targets'],frozen['states'],reference_prediction)
    record={'schema_version':1,'artifact_type':'larger_dataset_evaluation','generated_at':datetime.now(timezone.utc).isoformat(),
        'passed':True,'role':'step_zero_reference' if args.step_zero else 'checkpoint','split':split,'checkpoint_global_step':step,
        'checkpoint':str(model_path),'candidate_checkpoint':str(model_path),'checkpoint_files_sha256':checkpoint_files,
        'fresh_process':True,'process_id':os.getpid(),
        'reload_proof':{'strict_tensor_loading':True,'missing_keys':[],'unexpected_keys':[],'mismatched_keys':[],
                        'error_msgs':[],'model_load_count_in_process':1,'policy_reused_between_splits':reused_policy,
                        'finite_predictions':True,'single_prediction_shape':[HORIZON,26],
                        'saved_prediction_shape':list(prediction.shape),
                        'scope':'This evaluator CLI runs separately from training; paired splits share its one strict model load.'},
        'checkpoint_inventory':inventory_binding['path'] if inventory_binding else None,
        'checkpoint_inventory_sha256':inventory_binding['sha256'] if inventory_binding else None,
        'evaluation_protocol_sha256':protocol_sha,'recipe':RECIPE,'sample_count':100,'inferences':300,
        'sample_manifest_sha256':protocol['samples'][split]['sha256'],
        'processor_semantics_sha256':semantic_digest(semantics),'statistics_sha256':digest(processor_path/'statistics.json'),
        'model_config':model_config,'model_semantics_sha256':model_semantics_sha,
        'evaluation_mode':{'model_training':False,'processor_training':False},
        'groups':groups,'per_episode':episodes,'group_balanced_normalized_mae':selection_score(groups),
        'reference_report_sha256':digest(reference_path) if reference_path else None,
        'baseline_report_sha256':digest(reference_path) if reference_path else None,
        'reference_npz_sha256':digest(reference_path.with_suffix('.npz')) if reference_path else None,
        'scope':'Offline train/validation action error only; no mandatory improvement gate. Validation is a selection set, not a final test.',
        'elapsed_seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'artifacts':{p.name:digest(p) for p in [report_path.with_suffix('.npz'),report_path.with_suffix('.png')]}}
    write_new_json(report_path,record)
    print(json.dumps({'passed':True,'report':str(report_path),'split':split,'step':step,'group_balanced_normalized_mae':record['group_balanced_normalized_mae']}),flush=True)


def evaluate_requested(args,config):
    if args.split!='both':
        return evaluate(args,config)
    require(args.report is None and args.reference is None,'Paired evaluation uses --train-report/--validation-report and split-specific references')
    require(args.train_report is not None and args.validation_report is not None,'Paired evaluation requires both report paths')
    require(bool(args.train_reference)==bool(args.validation_reference)==(not args.step_zero),'Paired checkpoints require both same-split step-zero references')
    paths=[guard_path(args.train_report,config,'reports'),guard_path(args.validation_report,config,'reports')]
    require(paths[0]!=paths[1],'Train and validation outputs must differ')
    for path in paths:
        for output in [path,path.with_suffix('.npz'),path.with_suffix('.png')]:
            require(not output.exists() and not output.is_symlink(),'Paired evaluation output already exists')
    shared={}
    for split,path,reference in [('train',paths[0],args.train_reference),('validation',paths[1],args.validation_reference)]:
        one=copy.copy(args);one.split=split;one.report=path;one.reference=reference
        evaluate(one,config,shared)


def rank_checkpoints(records):
    require(len(records)==4 and {x['checkpoint_global_step'] for x in records}==set(RECIPE['checkpoint_steps']),'All four declared checkpoint evaluations are required')
    require(all(x.get('split')=='validation' and x.get('passed') is True for x in records),'Selection requires valid validation reports')
    require(all(math.isfinite(x['group_balanced_normalized_mae']) and x['group_balanced_normalized_mae']>=0 for x in records),'Invalid selection score')
    return sorted(records,key=lambda x:(x['group_balanced_normalized_mae'],x['checkpoint_global_step']))


def select_checkpoint(args, config):
    protocol_path=resolve_path(args.protocol); protocol=load_protocol(protocol_path); sha=digest(protocol_path)
    pairs={}
    for split,paths in [('train',args.train_reports),('validation',args.validation_reports)]:
        records=[]
        for item in paths:
            path=resolve_path(item); report,_=load_completed(path,protocol,sha,split)
            records.append({'split':split,'passed':True,'checkpoint_global_step':report['checkpoint_global_step'],
                'group_balanced_normalized_mae':report['group_balanced_normalized_mae'],
                'report':binding(path),'group_all_metrics':{name:report['groups'][name]['all'] for name,_,_,_ in GROUPS},
                'checkpoint_files_sha256':report['checkpoint_files_sha256'],
                'reference_report_sha256':report['reference_report_sha256']})
        require(len(records)==4 and {x['checkpoint_global_step'] for x in records}==set(RECIPE['checkpoint_steps']),'All four train/validation checkpoints required')
        require(len({x['reference_report_sha256'] for x in records})==1,'Step-zero reference changed within split')
        pairs[split]=records
    for train in pairs['train']:
        validation=next(x for x in pairs['validation'] if x['checkpoint_global_step']==train['checkpoint_global_step'])
        require(train['checkpoint_files_sha256']==validation['checkpoint_files_sha256'],'Train/validation evaluated different checkpoint weights')
    ranked=rank_checkpoints(pairs['validation'])
    output=guard_path(args.report,config,'reports')
    write_new_json(output,{'schema_version':1,'artifact_type':'larger_dataset_checkpoint_selection','passed':True,
        'evaluation_protocol_sha256':sha,'selection_rule':protocol['checkpoint_selection'],
        'selected_step':ranked[0]['checkpoint_global_step'],'validation_ranked_steps':[x['checkpoint_global_step'] for x in ranked],
        'evaluations':pairs,'persistence_role':'descriptive_only','mandatory_improvement_threshold':None})
    print(json.dumps({'selected_step':ranked[0]['checkpoint_global_step'],'report':str(output)}))


def main():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    sub=parser.add_subparsers(dest='command',required=True)
    freeze=sub.add_parser('prepare')
    for name in ['train-dataset','validation-dataset','train-manifest','validation-manifest','split-manifest','conversion-manifest','processor-output','protocol']:
        freeze.add_argument('--'+name,type=Path,required=True)
    evaluate_parser=sub.add_parser('evaluate')
    evaluate_parser.add_argument('--protocol',type=Path,required=True)
    evaluate_parser.add_argument('--split',choices=['train','validation','both'],required=True)
    evaluate_parser.add_argument('--step-zero',action='store_true')
    evaluate_parser.add_argument('--checkpoint',type=Path)
    evaluate_parser.add_argument('--reference',type=Path)
    evaluate_parser.add_argument('--report',type=Path)
    evaluate_parser.add_argument('--train-report',type=Path)
    evaluate_parser.add_argument('--validation-report',type=Path)
    evaluate_parser.add_argument('--train-reference',type=Path)
    evaluate_parser.add_argument('--validation-reference',type=Path)
    evaluate_parser.add_argument('--checkpoint-inventory',type=Path)
    select=sub.add_parser('select')
    select.add_argument('--protocol',type=Path,required=True)
    select.add_argument('--train-reports',type=Path,nargs=4,required=True)
    select.add_argument('--validation-reports',type=Path,nargs=4,required=True)
    select.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(); config=load_config(args.config)
    {'prepare':prepare,'evaluate':evaluate_requested,'select':select_checkpoint}[args.command](args,config)


if __name__=='__main__':main()
