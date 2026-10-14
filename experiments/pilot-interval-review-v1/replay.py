#!/usr/bin/env python3
"""Verify/replay the bounded interval summary using NumPy and saved files only.

No model, accelerator framework, inference, training, or data writes are used.
Run: vendor/Isaac-GR00T/.venv/bin/python -B experiments/pilot-interval-review-v1/replay.py
Use --emit to print the recomputed ten numeric records instead of the short result.
Saved NPZs, metadata and raw data must be restored at the relative bound paths.
The full ignored context inventory is optional; its hash is checked when present.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GROUPS = [('left_arm', 0, 7), ('right_arm', 7, 14), ('left_hand', 14, 20), ('right_hand', 20, 26)]


def digest(path):
    value = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reproduce(summary):
    for key, binding in summary['required_input_bindings'].items():
        require(digest(binding['path']) == binding['sha256'], f'Input hash mismatch: {key}')
    for binding in summary['full_report_references'].values():
        if (ROOT / binding['path']).exists():
            require(digest(binding['path']) == binding['sha256'], f'Full report hash mismatch: {binding["path"]}')
    inputs = summary['required_input_bindings']
    z = dict(np.load(ROOT / inputs['candidate_arrays']['path'], allow_pickle=False))
    old = dict(np.load(ROOT / inputs['checkpoint_512_arrays']['path'], allow_pickle=False))
    for key in z:
        if key != 'absolute_predictions':
            require(np.array_equal(z[key], old[key]), f'Array identity differs: {key}')
    samples = read(inputs['sample_manifest']['path'])['samples']
    pilot = read(inputs['pilot_manifest']['path'])['episodes']
    names = read(inputs['dataset_info']['path'])['features']['action']['names'][0]
    require(z['absolute_predictions'].shape == (100, 3, 16, 26), 'Unexpected prediction shape')
    require(z['seeds'].tolist() == [42, 43, 44], 'Unexpected seeds')
    require(np.array_equal(z['starts'], [x['start'] for x in samples]), 'Sample starts differ')
    require(np.array_equal(z['episode_indices'], [x['episode_index'] for x in samples]), 'Episode IDs differ')
    valid = z['normalization_active_mask']
    lower, upper = z['lower'].astype(float), z['upper'].astype(float)
    scale = 2 / np.where(valid, upper - lower, 1)
    err = z['absolute_predictions'].astype(float) - z['targets'].astype(float)[:, None]
    olderr = old['absolute_predictions'].astype(float) - z['targets'].astype(float)[:, None]
    norm, oldnorm = np.abs(err * scale), np.abs(olderr * scale)
    group = np.stack([norm[:, :, :, b:e][:, :, valid[:, b:e]].mean(axis=(1, 2)) for _, b, e in GROUPS], axis=1)
    oldgroup = np.stack([oldnorm[:, :, :, b:e][:, :, valid[:, b:e]].mean(axis=(1, 2)) for _, b, e in GROUPS], axis=1)
    scores, oldscores = group.mean(axis=1), oldgroup.mean(axis=1)
    output = []
    for episode in range(5):
        ids = [i for i, sample in enumerate(samples) if sample['episode_index'] == episode]
        worst = min(ids, key=lambda i: (-scores[i], samples[i]['start'], i))
        available = [i for i in ids if samples[i]['start'] + 15 < samples[worst]['start'] or
                     samples[i]['start'] > samples[worst]['start'] + 15]
        available.sort(key=lambda i: (scores[i], samples[i]['start'], i))
        median = available[(len(available) - 1) // 2]
        record = next(x for x in pilot if x['derived_episode_index'] == episode)
        rows = read(inputs[f'raw_{record["episode"]}']['path'])['data']
        arrays = {field: np.asarray([sum((row[field][part]['qpos'] for part in
                                         ['left_arm', 'right_arm', 'left_ee', 'right_ee']), [])
                                    for row in rows], dtype=np.float32).astype(float)
                  for field in ['actions', 'states']}
        action, state = arrays['actions'], arrays['states']
        for i in ids:
            start = samples[i]['start']
            require(np.array_equal(action[start:start + 16], z['targets'][i]), 'Raw/saved target mismatch')
            require(np.array_equal(state[start], z['states'][i]), 'Raw/saved state mismatch')
        for role, i in [('highest_error', worst), ('median_comparison', median)]:
            start, end = samples[i]['start'], samples[i]['start'] + 15
            first, last = max(0, start - 30), min(len(rows) - 1, end + 30)
            pair = norm[i].mean(axis=0)
            h, j = min(((h, j) for h in range(16) for j in range(26) if valid[h, j]),
                       key=lambda cell: (-pair[cell], cell[0], cell[1]))
            metrics = {}
            for gi, (name, b, e) in enumerate(GROUPS):
                x = err[i, :, :, b:e]
                metrics[name] = {
                    'candidate_native_mae': float(np.abs(x).mean()),
                    'candidate_normalized_mae': float(group[i, gi]),
                    'checkpoint_512_native_mae': float(np.abs(olderr[i, :, :, b:e]).mean()),
                    'checkpoint_512_normalized_mae': float(oldgroup[i, gi]),
                    'seed_variance_fraction_of_candidate_native_mse': float(x.var(axis=0).mean() / np.square(x).mean()),
                    'commanded_mean_joint_range_native': float(np.ptp(action[start:end + 1, b:e], axis=0).mean()),
                    'measured_mean_joint_range_native': float(np.ptp(state[start:end + 1, b:e], axis=0).mean()),
                    'same_row_command_measured_native_mae': float(np.abs(action[start:end + 1, b:e] - state[start:end + 1, b:e]).mean()),
                }
            output.append({
                'interval_id': f"{record['episode']}_{role}", 'raw_episode': record['episode'],
                'sample_index': i, 'evaluated_rows_inclusive': [start, end],
                'context_rows_inclusive': [first, last],
                'candidate_group_balanced_normalized_mae': float(scores[i]),
                'checkpoint_512_group_balanced_normalized_mae': float(oldscores[i]),
                'candidate_relative_change_vs_512_percent': float(100 * (scores[i] / oldscores[i] - 1)),
                'group_metrics': metrics,
                'highest_normalized_error_cell': {
                    'joint_name': names[j], 'horizon_offset': h, 'source_row': start + h,
                    'candidate_normalized_mae': float(pair[h, j]),
                    'checkpoint_512_normalized_mae': float(oldnorm[i, :, h, j].mean()),
                    'candidate_native_residual_per_seed': err[i, :, h, j].tolist(),
                },
            })
    return output


def same(actual, expected, path='intervals'):
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f'Keys differ: {path}')
        for key in expected:
            same(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'Length differs: {path}')
        for i, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, f'{path}[{i}]')
    elif isinstance(expected, float):
        require(np.isclose(actual, expected, rtol=1e-12, atol=1e-12), f'Numeric value differs: {path}')
    else:
        require(actual == expected, f'Value differs: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit', action='store_true')
    args = parser.parse_args()
    summary = read('experiments/pilot-interval-review-v1/summary.json')
    require(digest(Path(__file__)) == summary['replay_helper_sha256'], 'Replay helper hash mismatch')
    result = reproduce(summary)
    same(result, summary['intervals'])
    print(json.dumps(result, indent=2, allow_nan=False) if args.emit else
          'Verified all input hashes, all 100 raw/saved target-state bindings, and the 10 interval selections and numeric summaries; no inference or gate rescoring.')
