"""Directed dropout compatibility and the predeclared multi-reference acceptance gate."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_gr00t_pilot as evaluation
import compare_pilot_temporal_change as temporal


def report(dropout=0.2):
    semantics = {'processor_config': {'processor_kwargs': {'state_dropout_prob': dropout,
                  'exclude_state': False, 'max_action_horizon': 16}},
                 'statistics': {'action': [0.0, 1.0]}, 'modalities': ['left_arm', 'right_arm'],
                 'lower': [0.0], 'upper': [1.0]}
    return {**copy.deepcopy(evaluation.EVALUATION_RECIPE), 'sample_manifest_sha256': 'samples',
            'statistics_sha256': 'stats', 'processor_semantics': semantics,
            'processor_semantics_sha256': evaluation.semantic_digest(semantics)}


def rehash(value):
    value['processor_semantics_sha256'] = evaluation.semantic_digest(value['processor_semantics'])
    return value


class DirectedCompatibilityTests(unittest.TestCase):
    def test_only_explicit_directed_change_is_permitted(self):
        before, after = report(), report(0.0)
        evaluation.assert_processor_compatible(before, before)
        with self.assertRaises(ValueError): evaluation.assert_processor_compatible(before, after)
        evaluation.assert_processor_compatible(before, after, allow_state_dropout=True)
        for first, last in [(after, before), (before, before), (before, report(0.1)), (before, report(False))]:
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                evaluation.assert_processor_compatible(first, last, allow_state_dropout=True)

    def test_statistics_modalities_bounds_and_other_dropout_settings_are_not_ignored(self):
        for key, value in [('statistics', {'action': [0, 2]}), ('modalities', ['right_arm', 'left_arm']),
                           ('lower', [-1]), ('upper', [2])]:
            candidate = report(0.0); candidate['processor_semantics'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.assert_processor_compatible(report(), rehash(candidate), True)
        for key, value in [('exclude_state', True), ('max_action_horizon', 15), ('other_dropout', 0.0)]:
            candidate = report(0.0)
            candidate['processor_semantics']['processor_config']['processor_kwargs'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.assert_processor_compatible(report(), rehash(candidate), True)

    def test_claimed_hash_cannot_hide_changed_semantics(self):
        candidate = report(0.0); candidate['processor_semantics']['upper'] = [9]
        with self.assertRaisesRegex(ValueError, 'hash'):
            evaluation.assert_processor_compatible(report(), candidate, True)

    def test_model_architecture_and_other_training_dropout_remain_fixed(self):
        first = {'state_dropout_prob': 0.2, 'horizon': 40, 'diffusion_model_cfg': {'dropout': 0.2}}
        last = {**copy.deepcopy(first), 'state_dropout_prob': 0.0}
        evaluation.assert_no_dropout_model_config(last, first)
        for key, value in [('horizon', 16), ('state_dropout_prob', False), ('diffusion_model_cfg', {'dropout': 0.0})]:
            candidate = {**copy.deepcopy(last), key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.assert_no_dropout_model_config(candidate, first)

    def test_recipe_and_model_evidence_are_required(self):
        protocol = {'bindings': {'sample_manifest': {'sha256': 'samples'}}, 'statistics_sha256': 'stats',
                    'reference_model_config': {'state_dropout_prob': 0.2, 'architecture': 'same'}}
        candidate = report(0.0)
        candidate.update(checkpoint_global_step=1024, evaluation_mode={'model_training': False, 'processor_training': False},
                         model_config={'state_dropout_prob': 0.0, 'architecture': 'same'})
        candidate['model_config_semantics_sha256'] = evaluation.semantic_digest(candidate['model_config'])
        evaluation.assert_protocol_evaluation(candidate, protocol, candidate=True)
        for key, value in [('seeds', [42, 43, 45]), ('horizon', 15), ('denoising_steps', 5),
                           ('statistics_sha256', 'changed'), ('sample_manifest_sha256', 'changed'),
                           ('checkpoint_global_step', 512), ('evaluation_mode', {'model_training': False, 'processor_training': True})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.assert_protocol_evaluation({**candidate, key: value}, protocol, candidate=True)

    def test_bound_artifact_bytes_cannot_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'reference.json'; path.write_text('original')
            protocol = {'bindings': {'baseline_report': {'sha256': evaluation.digest(path)}}}
            evaluation.require_protocol_binding(protocol, 'baseline_report', path)
            path.write_text('modified')
            with self.assertRaises(ValueError): evaluation.require_protocol_binding(protocol, 'baseline_report', path)


class AcceptanceTests(unittest.TestCase):
    def groups(self):
        return {name: {'changing_target': {'sample_count': 50, 'candidate': {'mae': 0.9, 'normalized_mae': 1.8},
                                         'persistence': {'mae': 1.0, 'normalized_mae': 2.0}},
                       'all': {'sample_count': 100, 'candidate': {'mae': 0.9, 'normalized_mae': 1.8},
                               'reference': {'mae': 0.9, 'normalized_mae': 1.8}}}
                for name, _, _, _ in evaluation.GROUPS}

    def test_exact_ten_percent_and_equal_previous_model_pass_all_sixteen_gates(self):
        result = temporal.acceptance_decision(self.groups())
        self.assertTrue(result['passed']); self.assertEqual(len(result['checks']), 16)

    def test_every_group_and_both_metrics_must_meet_each_threshold(self):
        for name, _, _, _ in evaluation.GROUPS:
            for subset in ['changing_target', 'all']:
                for metric in ['mae', 'normalized_mae']:
                    groups = self.groups(); groups[name][subset]['candidate'][metric] += 0.0001
                    with self.subTest(group=name, subset=subset, metric=metric):
                        result = temporal.acceptance_decision(groups)
                        self.assertFalse(result['passed'])
                        self.assertEqual(result['failed_checks'], [f'{name}/{subset}/{metric}'])

    def test_empty_missing_nonfinite_and_zero_persistence_fail(self):
        for mutate in [lambda g: g['changing_target'].update(sample_count=0),
                       lambda g: g['all'].update(sample_count=99),
                       lambda g: g['changing_target']['candidate'].update(normalized_mae=None),
                       lambda g: g['all']['candidate'].update(mae=float('nan')),
                       lambda g: g['changing_target']['persistence'].update(mae=0)]:
            groups = self.groups(); mutate(groups['right_hand'])
            self.assertFalse(temporal.acceptance_decision(groups)['passed'])

    def test_threshold_or_group_list_cannot_be_weakened(self):
        rules = copy.deepcopy(evaluation.ACCEPTANCE_RULES)
        rules['changing_target']['minimum_relative_improvement'] = 0.0
        with self.assertRaises(ValueError): temporal.acceptance_decision(self.groups(), rules)
        groups = self.groups(); del groups['left_hand']
        with self.assertRaises(ValueError): temporal.acceptance_decision(groups)


class SavedArrayComparisonTests(unittest.TestCase):
    def setUp(self):
        try: import numpy as np
        except ImportError: self.skipTest('Requires the configured numerical environment')
        self.np = np

    def arrays(self, error):
        np = self.np; targets = np.broadcast_to(np.linspace(0, 1, 16)[None, :, None], (100, 16, 26)).copy()
        return {'absolute_predictions': np.repeat(targets[:, None], 3, axis=1) + error,
                'targets': targets, 'states': np.zeros((100, 26)), 'lower': np.zeros((16, 26)),
                'upper': np.ones((16, 26)), 'seeds': np.array(evaluation.SEEDS),
                'episode_indices': np.repeat(np.arange(5), 20), 'starts': np.tile(np.arange(20), 5)}

    def test_actual_native_and_normalized_metrics_feed_the_gate(self):
        before, after = self.arrays(0.4), self.arrays(0.1)
        temporal.assert_comparable(report(), report(0.0), before, after, allow_state_dropout=True)
        mask, _, _ = temporal.temporal_subset(before['targets'], before['lower'], before['upper'])
        groups = temporal.comparison_groups(before, after, mask, include_all=True)
        self.assertTrue(temporal.acceptance_decision(groups)['passed'])
        for name in groups:
            self.assertAlmostEqual(groups[name]['all']['candidate']['mae'], 0.1)
            self.assertAlmostEqual(groups[name]['all']['candidate']['normalized_mae'], 0.2)
            self.assertAlmostEqual(groups[name]['changing_target']['persistence']['mae'], 0.5)

    def test_changed_targets_states_sampling_or_normalization_are_rejected(self):
        before = self.arrays(0.4)
        for key in ['targets', 'states', 'seeds', 'episode_indices', 'starts', 'lower', 'upper']:
            after = self.arrays(0.1); after[key].flat[0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                temporal.assert_comparable(report(), report(0.0), before, after, allow_state_dropout=True)


if __name__ == '__main__': unittest.main()
