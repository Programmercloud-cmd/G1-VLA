"""CPU contract checks for additive full-state views and extension comparisons."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import extension_artifacts as common
import archive_gr00t_extension as archive
import evaluate_gr00t_extension as evaluation
import archive_gr00t_dataset as old


class FullStateViewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.source = self.root/'canonical/checkpoint-2048'
        self.source.mkdir(parents=True)
        metadata = {'artifact_type': 'larger_dataset_extension_checkpoint', 'local_step': 2048,
                    'cumulative_step': 10240, 'cumulative_base_step':8192, 'sampling_limit': 'Native iterable cursor is not restorable.',
                    'full_state': {'passed': True, 'scheduler_epoch': 2048, 'exact_iterable_cursor_restore': False}}
        values = {'config.json': {}, 'trainer_state.json': {'global_step': 2048},
                  'processor_config.json': {}, 'statistics.json': {}, 'embodiment_id.json': {'new_embodiment': 10},
                  'model.safetensors.index.json': {'weight_map': {'parameter': 'model-00001-of-00001.safetensors'}},
                  'continuation.json': metadata}
        for name, value in values.items():
            (self.source/name).write_text(json.dumps(value))
        for name in ['model-00001-of-00001.safetensors', *common.FULL_STATE_FILES, 'training_args.bin']:
            (self.source/name).write_bytes((name+' fixture').encode())
        self.inventory_path = self.root/'full-inventory.json'
        self.full = self.inventory(self.source, self.inventory_path)

    def inventory(self, checkpoint, destination):
        records = old.hash_checkpoint_in_place(checkpoint)
        value = {'schema_version': 1, 'artifact_type': 'durable_checkpoint_inventory', 'passed': True,
                 'checkpoint': str(checkpoint), 'step': 2048, 'files': records,
                 'per_file_fsync_completed': True, 'hash_location': 'configured_storage_host',
                 'remote_identity': {'host': 'fixture', 'path': str(checkpoint)}}
        destination.write_text(json.dumps(value)); return value

    def make_view(self, method='hardlink'):
        view = self.root/'inference/checkpoint-2048'
        selected = [row for row in self.full['files'] if row['path'] in common.model_view_names(self.full)]
        result = subprocess.run([sys.executable, '-B', '-c', archive.VIEW_PROGRAM, str(self.source), str(view),
                                 json.dumps(selected), method], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        inventory = self.inventory(view, self.root/'view-inventory.json')
        manifest = {'artifact_type': 'extension_model_only_inference_view', 'resumable_training_checkpoint': False,
                    'local_step': 2048, 'cumulative_step': 10240, 'canonical_full_checkpoint': str(self.source),
                    'view_path': str(view), 'method': method}
        return manifest, inventory

    @patch.object(old, 'verify_remote_signatures')
    def test_full_state_signatures_and_small_metadata_are_verified(self, remote):
        result = common.full_inventory(self.inventory_path, self.source)
        self.assertEqual(result['step'], 2048); remote.assert_called_once()

    @patch.object(old, 'verify_remote_signatures')
    def test_missing_native_state_and_false_sampler_claim_rejected(self, remote):
        (self.source/'optimizer.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'lacks optimizer'):
            common.full_inventory(self.inventory_path, self.source)
        self.setUp()
        metadata = json.loads((self.source/'continuation.json').read_text())
        metadata['full_state']['exact_iterable_cursor_restore'] = True
        (self.source/'continuation.json').write_text(json.dumps(metadata))
        self.full = self.inventory(self.source, self.inventory_path)
        with self.assertRaisesRegex(ValueError, 'honest sampler'):
            common.full_inventory(self.inventory_path, self.source)

    @patch.object(old, 'verify_remote_signatures')
    def test_large_optimizer_not_read_across_mounted_view(self, remote):
        with (self.source/'optimizer.pt').open('wb') as stream:
            stream.truncate(65 * 1024**2)
        self.full = self.inventory(self.source, self.inventory_path)
        actual = common.original.digest
        def digest(path):
            self.assertNotEqual(Path(path).name, 'optimizer.pt', 'Large optimizer was reread locally')
            return actual(path)
        with patch.object(common.original, 'digest', side_effect=digest):
            common.full_inventory(self.inventory_path, self.source)

    def test_remote_view_program_hardlinks_exact_model_files_only(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        manifest, view = self.make_view()
        common.validate_view_mapping(manifest, self.full, view)
        self.assertFalse(common.FULL_STATE_FILES & {row['path'] for row in view['files']})
        for row in view['files']:
            self.assertEqual((self.source/row['path']).stat().st_ino,
                             (Path(view['checkpoint'])/row['path']).stat().st_ino)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})

    def test_verified_copy_fallback_has_equal_bytes_and_independent_inodes(self):
        manifest, view = self.make_view('verified_copy')
        common.validate_view_mapping(manifest, self.full, view)
        self.assertTrue(all((self.source/r['path']).stat().st_ino != (Path(view['checkpoint'])/r['path']).stat().st_ino
                            for r in view['files']))

    def test_view_bytes_identity_ancestry_and_local_counts_cannot_be_faked(self):
        manifest, view = self.make_view()
        for field, value in [('resumable_training_checkpoint', True), ('cumulative_step', 12288),
                             ('local_step', 10240), ('canonical_full_checkpoint', '/different/checkpoint-2048')]:
            bad = dict(manifest); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                common.validate_view_mapping(bad, self.full, view)
        bad = copy.deepcopy(view); bad['files'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'View bytes'):
            common.validate_view_mapping(manifest, self.full, bad)
        bad = copy.deepcopy(view); bad['files'][0]['inode'] += 1
        with self.assertRaisesRegex(ValueError, 'hard link'):
            common.validate_view_mapping(manifest, self.full, bad)

    def test_view_rejects_existing_destination_and_state_files(self):
        manifest, view = self.make_view()
        selected = [row for row in self.full['files'] if row['path'] in common.model_view_names(self.full)]
        result = subprocess.run([sys.executable, '-B', '-c', archive.VIEW_PROGRAM, str(self.source), view['checkpoint'],
                                 json.dumps(selected), 'hardlink'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        extra = copy.deepcopy(view); extra['files'].append(next(row for row in self.full['files'] if row['path']=='optimizer.pt'))
        with self.assertRaisesRegex(ValueError, 'exact model-only'):
            common.validate_view_mapping(manifest, self.full, extra)


class ExtensionMetricTests(unittest.TestCase):
    def test_per_episode_metrics_do_not_slice_seed_axis_or_normalization_bounds(self):
        arrays = {'absolute_predictions':np.zeros((100,3,16,26)), 'targets':np.zeros((100,16,26)),
                  'states':np.zeros((100,26)), 'changing_target_mask':np.zeros((100,4),dtype=bool),
                  'lower':np.zeros((16,26)), 'upper':np.ones((16,26)), 'seeds':np.array([42,43,44])}
        mask=np.arange(100)<4
        sliced=evaluation.metric_episode(arrays,mask)
        self.assertEqual(sliced['absolute_predictions'].shape,(4,3,16,26))
        self.assertIs(sliced['lower'],arrays['lower']);self.assertNotIn('seeds',sliced)

    def test_uncertified_training_cannot_reach_model_subprocess(self):
        args=SimpleNamespace(extension_protocol=Path('/declaration'),view_manifest=Path('/view'),view_inventory=Path('/inventory'))
        view={'training_run':{},'training_report':{},'canonical_full_checkpoint':'/checkpoint-2048'}
        with patch.object(common,'load_protocol',return_value=({}, {}, Path('/original'))), \
             patch.object(common,'load_view',return_value=(view,{'step':2048},{})), \
             patch.object(common,'verify_binding',return_value=Path('/bound')), \
             patch.object(common,'certify_training',side_effect=ValueError('uncertified finite prefix')), \
             patch.object(evaluation.subprocess,'run') as model:
            with self.assertRaisesRegex(ValueError,'uncertified finite prefix'):evaluation.evaluate(args,{})
            model.assert_not_called()

    def test_changes_use_source8192_predictions_and_explicit_reference_names(self):
        arrays = {'absolute_predictions': np.full((2,3,16,26), .25), 'targets': np.zeros((2,16,26)),
                  'states': np.zeros((2,26)), 'lower': np.zeros((16,26)), 'upper': np.ones((16,26)),
                  'changing_target_mask': np.array([[True,False,True,False],[False,True,True,False]])}
        result = evaluation.comparison_groups(arrays, {'absolute_predictions': np.full((2,3,16,26), .5)})
        for group in result.values():
            self.assertEqual(group['all']['model']['normalized_mae'], .5)
            self.assertEqual(group['all']['source_8192']['normalized_mae'], 1.)
            self.assertEqual(group['all']['change_from_source_8192']['normalized_mae'], -.5)
            self.assertNotIn('step_zero', group['all'])
        self.assertIsNone(result['right_hand']['changing_target']['model']['mae'])

    def test_frozen_sources_and_protocol_bytes_remain_unchanged(self):
        protocol = json.loads((ROOT/'experiments/larger-dataset-v1/evaluation-protocol.json').read_text())
        prior = json.loads((ROOT/'experiments/larger-dataset-v1-continuation-v1/evaluation-protocol-startup-fix-v2.json').read_text())
        frozen = {**protocol['source_sha256'],**prior['new_source_sha256']}
        self.assertEqual(len(frozen), 27)
        for name, expected in frozen.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)

    def test_new_selection_scope_is_declared_separately(self):
        self.assertEqual(common.SELECTION_RULE['candidates_cumulative_steps'], [8192,10240,12288])
        self.assertEqual(common.SELECTION_RULE['tie'], 'earliest_cumulative_step')
        self.assertTrue(common.SELECTION_RULE['available_selection_always_allowed'])
        self.assertIsNone(common.SELECTION_RULE['mandatory_improvement_threshold'])
        self.assertEqual(common.LOCAL_TO_CUMULATIVE, {2048:10240,4096:12288})


class DeclarationBindingTests(unittest.TestCase):
    def test_changed_samples_missing_evidence_and_changed_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);original_path=folder/'original.json';original_path.write_text('{}')
            old_protocol=json.loads((ROOT/'experiments/larger-dataset-v1/evaluation-protocol.json').read_text())
            prior=json.loads((ROOT/'experiments/larger-dataset-v1-continuation-v1/evaluation-protocol-startup-fix-v2.json').read_text())
            original_protocol={'samples':{'train':'fixed100','validation':'fixed100'},'processor':{'stats':'train-only'},'recipe':{'seeds':[42,43,44]},'source_sha256':old_protocol['source_sha256']}
            bound=folder/'evidence.json';bound.write_text('{}')
            names={'source_inventory','source_archive','source_archive_manifest','source_comparison',
                   'source_view_manifest','source_view_inventory','selected_policy','source_run_manifest',
                   'source_train_report','source_closeout','source_independent_proof'}
            names|={role+'_'+split+suffix for role in ['source','baseline'] for split in ['train','validation'] for suffix in ['','_arrays']}
            declaration={'schema_version':1,'artifact_type':'g1_extension_evaluation_protocol',
                         'local_to_cumulative':{'2048':10240,'4096':12288},'cumulative_base_step':8192,
                         'original_evaluation_protocol':common.binding(original_path),
                         'previous_continuation_protocol':common.binding(bound),
                         'frozen_source_sha256':{**old_protocol['source_sha256'],**prior['new_source_sha256']},
                         'samples':original_protocol['samples'],'processor':original_protocol['processor'],
                         'inference_recipe':original_protocol['recipe'],'selection_rule':common.SELECTION_RULE,
                         'training_segment':common.TRAINING_SEGMENT,'full_state_policy':common.FULL_STATE_POLICY,
                         'bindings':{name:common.binding(bound) for name in names},'new_source_sha256':{str(bound):common.original.digest(bound)}}
            path=folder/'declaration.json'
            def check(value):
                path.write_text(json.dumps(value));return common.load_protocol(path)
            with patch.object(common.original,'load_protocol',return_value=original_protocol), \
                 patch.object(common.previous,'load_protocol',return_value=(prior,original_protocol,original_path)), \
                 patch.object(common,'NEW_SOURCES',[str(bound)]):
                self.assertEqual(check(declaration)[0],declaration)
                bad=copy.deepcopy(declaration);bad['samples']['train']='different'
                with self.assertRaisesRegex(ValueError,'exact frozen samples'):check(bad)
                bad=copy.deepcopy(declaration);del bad['bindings']['source_train']
                with self.assertRaisesRegex(ValueError,'binding set'):check(bad)
                bad=copy.deepcopy(declaration);bad['new_source_sha256'][str(bound)]='0'*64
                with self.assertRaisesRegex(ValueError,'implementation changed'):check(bad)
                bad=copy.deepcopy(declaration);bad['full_state_policy']['inference_views_resumable']=True
                with self.assertRaisesRegex(ValueError,'full-state policy changed'):check(bad)


class SourceLineageTests(unittest.TestCase):
    def test_two_real_native_prefixes_are_required_not_one_fabricated8192(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);segments=[]
            for base in [0,4096]:
                coverage=root/f'{base}-coverage.json';coverage.write_text(json.dumps({'consumed_examples':32768}))
                event={'event':'checkpoint','step':4096,'passed':True,'coverage_path':str(coverage),'coverage_sha256':common.original.digest(coverage)}
                audit=root/f'{base}-audit.jsonl';audit.write_text(json.dumps(event)+'\n')
                run=root/f'{base}-run.json';run.write_text(json.dumps({'artifact_type':'larger_dataset_training_run' if base==0 else 'larger_dataset_continuation_run',
                    'max_steps':8192 if base==0 else 4096,'cumulative_base_step':base,'initial_checkpoint':{'source_updates':4096},
                    'audit':str(audit),'evaluation_protocol_sha256':'original'}))
                report=root/f'{base}-train.json';report.write_text(json.dumps({'run_manifest_sha256':common.original.digest(run),
                    'finite_audit_sha256':common.original.digest(audit),'evaluation_protocol_sha256':'original'}))
                segments.append({'cumulative_base_step':base,'local_updates':4096,'cumulative_end_step':base+4096,
                    'run_manifest':common.binding(run),'train_report':common.binding(report),'audit':common.binding(audit),
                    'finite_prefix':{'finite_summary':{'passed':True},'audit_prefix_sha256':common.original.digest(audit),
                                     'coverage_path':str(coverage),'coverage_sha256':common.original.digest(coverage)}})
            lineage={'schema_version':2,'artifact_type':'certified_multisegment_weight_lineage','cumulative_source_updates':8192,
                     'native_segments':segments,'lineage_coverage':{'consumed_examples':65536},'source_unsaved1135_updates_excluded':True}
            def native_prefix(records,step):
                self.assertEqual(step,4096);return {'passed':True},records[0]
            with patch.object(old,'prefix_integrity',side_effect=native_prefix) as finite:
                common.verify_source_prefix(lineage,'original');self.assertEqual(finite.call_count,2)
                bad=copy.deepcopy(lineage);bad['native_segments'][0]['local_updates']=8192
                with self.assertRaisesRegex(ValueError,'mapping differs'):common.verify_source_prefix(bad,'original')
                bad=copy.deepcopy(lineage);bad['native_segments']=bad['native_segments'][:1]
                with self.assertRaisesRegex(ValueError,'exactly two'):common.verify_source_prefix(bad,'original')
                bad=copy.deepcopy(lineage);bad['source_unsaved1135_updates_excluded']=False
                with self.assertRaisesRegex(ValueError,'two-segment source'):common.verify_source_prefix(bad,'original')
                Path(segments[1]['audit']['path']).write_text('changed')
                with self.assertRaisesRegex(ValueError,'Changed binding'):common.verify_source_prefix(lineage,'original')


class ExtensionRankingTests(unittest.TestCase):
    def test_parent_can_remain_best_ties_choose_earliest_and_partial_is_descriptive(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);declaration_path=root/'protocol.json';declaration_path.write_text('{}')
            source=root/'source-validation.json';source.write_text('{}')
            declaration={'bindings':{'source_validation':{'path':str(source)},'source_inventory':{'path':str(source)}},'source_checkpoint':'/parent/checkpoint-4096','selection_rule':common.SELECTION_RULE}
            identity={'size':1,'sha256':'fixture','path':'model.safetensors'}
            paths=[];reports={str(source):{'group_balanced_normalized_mae':.1}}
            for local,cumulative in common.LOCAL_TO_CUMULATIVE.items():
                evaluations={}
                for split in ['train','validation']:
                    report=root/f'{local}-{split}.json';report.write_text('{}');evaluations[split]={'report':{'path':str(report)}}
                    reports[str(report)]={'checkpoint_global_step':local,'checkpoint':str(root/f'checkpoint-{local}'),
                                         'checkpoint_inventory_sha256':'digest','checkpoint_files_sha256':{'model.safetensors':'fixture'},
                                         'process_id':local,'reload_proof':{'policy_reused_between_splits':split=='validation'},
                                         'group_balanced_normalized_mae':.2}
                comparison=root/f'{local}-comparison.json';comparison.write_text(json.dumps({
                    'artifact_type':'g1_extension_checkpoint_comparison','passed':True,
                    'extension_protocol_sha256':'digest','local_step':local,'cumulative_step':cumulative,
                    'view_manifest':{'path':str(root/f'{local}-view.json')},'view_inventory':{'path':str(root/f'{local}-inventory.json')},
                    'evaluations':evaluations}))
                paths.append(comparison)
            def load_view(receipt,inventory):
                local=int(receipt.name.split('-')[0]);return ({'training_run':{},'training_report':{},
                    'canonical_full_checkpoint':str(root/f'checkpoint-{local}'),'view_path':str(root/f'checkpoint-{local}')},
                    {'step':local},{'files':[identity]})
            def certified(run,report,checkpoint,local):return ({},{},{},{},{'step':local})
            def completed(path,*args):return reports[str(path)],{}
            with patch.object(common,'load_protocol',return_value=(declaration,{},declaration_path)), \
                 patch.object(common,'verify_binding',side_effect=lambda item:Path(item.get('path',str(source)))), \
                 patch.object(common,'load_view',side_effect=load_view),patch.object(common,'certify_training',side_effect=certified), \
                 patch.object(common,'verify_ancestry'),patch.object(evaluation.previous,'full_inventory'),patch.object(evaluation.old_archive,'validate_reload_proof'), \
                 patch.object(evaluation.original,'load_completed',side_effect=completed), \
                 patch.object(evaluation.original,'digest',return_value='digest'), \
                 patch.object(evaluation,'guard_path',side_effect=lambda path,*args:Path(path)):
                output=root/'selection.json';args=SimpleNamespace(extension_protocol=declaration_path,comparisons=paths,output=output)
                evaluation.select(args,{});result=json.loads(output.read_text())
                self.assertTrue(result['formal_extension_selection_complete']);self.assertEqual(result['selected_cumulative_step'],8192)
                reports[str(source)]['group_balanced_normalized_mae']=.2;output.unlink();evaluation.select(args,{})
                self.assertEqual(json.loads(output.read_text())['selected_cumulative_step'],8192)
                args.comparisons=paths[:1];output.unlink();evaluation.select(args,{})
                result=json.loads(output.read_text());self.assertFalse(result['formal_extension_selection_complete'])
                self.assertEqual(result['selected_cumulative_step'],8192);self.assertEqual(result['missing_cumulative_steps'],[12288])
                self.assertEqual(result['selection_status'],'partial_best_available')
                reports[str(root/'2048-validation.json')]['group_balanced_normalized_mae']=.05
                output.unlink();evaluation.select(args,{})
                self.assertEqual(json.loads(output.read_text())['selected_cumulative_step'],10240)
                args.comparisons=[];output.unlink();evaluation.select(args,{})
                result=json.loads(output.read_text());self.assertEqual(result['selected_cumulative_step'],8192)
                self.assertEqual(result['missing_cumulative_steps'],[10240,12288]);self.assertFalse(result['planned_comparisons_complete'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
