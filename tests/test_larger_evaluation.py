"""Fixed train/validation selection, statistics isolation, and descriptive scoring."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_gr00t_dataset as ev


class SelectionTests(unittest.TestCase):
    def episodes(self,count,length=200):
        return [dict(episode_index=i,source_episode=f'episode_{i:04d}',length=length+i) for i in range(count)]

    def test_98_train_and_24_validation_all_represented(self):
        for count in [98,24]:
            with self.subTest(count=count):
                result=ev.select_windows(self.episodes(count))
                self.assertEqual(result,ev.select_windows(self.episodes(count)))
                self.assertEqual(len(result),100)
                self.assertEqual({x['episode_index'] for x in result},set(range(count)))
                allocations=[sum(x['episode_index']==i for x in result) for i in range(count)]
                self.assertLessEqual(max(allocations)-min(allocations),1)
                self.assertEqual(len({(x['episode_index'],x['start']) for x in result}),100)
                self.assertTrue(all(0<=x['start']<=x['length']-16 for x in result))

    def test_sampling_reproducible_and_independent_of_extra_metrics(self):
        episodes=self.episodes(24)
        changed=[dict(x,prediction_error=1000-i,target_label='ignored') for i,x in enumerate(episodes)]
        self.assertEqual(ev.select_windows(episodes),ev.select_windows(changed))
        self.assertNotEqual(ev.select_windows(episodes),ev.select_windows(episodes,seed=43))

    def test_short_episodes_capacity_and_noncontiguous_indices_fail(self):
        for episodes in [self.episodes(1,15),self.episodes(1,16),[dict(episode_index=4,source_episode='x',length=300)]]:
            with self.subTest(episodes=episodes),self.assertRaises(ValueError):ev.select_windows(episodes)
        result=ev.select_windows([dict(episode_index=0,source_episode='short',length=16),dict(episode_index=1,source_episode='long',length=300)])
        self.assertEqual(sum(x['episode_index']==0 for x in result),1)

    def test_more_episodes_than_budget_does_not_claim_full_coverage(self):
        result=ev.select_windows(self.episodes(110))
        self.assertEqual(len(result),100)
        self.assertEqual(len({x['episode_index'] for x in result}),100)

    def test_split_overlap_and_relabelled_order_are_rejected(self):
        split={'artifact_type':'frozen_dataset_split','seed':42,'splits':{'train':['a'],'validation':['b']},
               'episodes':[dict(episode='a',split='train',derived_episode_index=0,rows=30),dict(episode='b',split='validation',derived_episode_index=0,rows=30)]}
        manifests={'train':{'episodes':[dict(episode='a',rows=30)]},'validation':{'episodes':[dict(episode='b',rows=30)]}}
        ev.validate_split(split,manifests)
        changed=copy.deepcopy(split);changed['splits']['validation']=['a']
        with self.assertRaisesRegex(ValueError,'overlap'):ev.validate_split(changed,manifests)
        changed=copy.deepcopy(manifests);changed['validation']['episodes'][0]['episode']='c'
        with self.assertRaisesRegex(ValueError,'membership'):ev.validate_split(split,changed)


class MetricsTests(unittest.TestCase):
    def fixture(self):
        target=np.zeros((2,16,26));state=np.zeros((2,26));prediction=np.ones((2,3,16,26))
        lower=np.zeros((16,26));upper=np.ones((16,26))*4;changing=np.zeros((2,4),dtype=bool)
        return prediction,target,state,lower,upper,changing

    def test_native_normalized_metrics_and_no_persistence_gate(self):
        groups=ev.summarize_arrays(*self.fixture())
        for name,_,_,_ in ev.GROUPS:
            self.assertEqual(groups[name]['all']['model']['mae'],1.)
            self.assertEqual(groups[name]['all']['model']['rmse'],1.)
            self.assertEqual(groups[name]['all']['model']['normalized_mae'],.5)
            self.assertEqual(groups[name]['all']['persistence']['mae'],0.)
        self.assertEqual(ev.selection_score(groups),.5)  # Worse than persistence remains valid evidence.
        self.assertIsNone(ev.RECIPE['mandatory_improvement_threshold'])

    def test_empty_subsets_and_constant_cells_are_unavailable(self):
        values=list(self.fixture());values[4][:,:7]=0
        groups=ev.summarize_arrays(*values)
        for name,_,_,_ in ev.GROUPS:
            self.assertIsNone(groups[name]['changing_target']['model']['mae'])
        self.assertIsNone(groups['left_arm']['all']['model']['normalized_mae'])
        self.assertEqual(groups['left_arm']['all']['model']['mae'],1.)
        with self.assertRaises(ValueError):ev.selection_score(groups)

    def test_nonfinite_and_malformed_evidence_are_rejected(self):
        for index in range(5):
            values=list(self.fixture());values[index]=values[index].copy();values[index].flat[0]=np.nan
            with self.subTest(index=index),self.assertRaisesRegex(ValueError,'Nonfinite'):ev.summarize_arrays(*values)
        values=list(self.fixture());values[0]=values[0][:,:2]
        with self.assertRaises(ValueError):ev.summarize_arrays(*values)

    def test_target_change_ignores_constant_state_command_offset(self):
        values=list(self.fixture());values[1][:]=100
        mask,_,_=ev.temporal_subset(values[1],values[3],values[4])
        self.assertFalse(mask.any())
        values[1][0,15,0]+=.21
        mask,_,_=ev.temporal_subset(values[1],values[3],values[4])
        self.assertTrue(mask[0,0]);self.assertEqual(mask.sum(),1)

    def test_reference_regressions_are_reported_without_acceptance_field(self):
        values=self.fixture();reference=np.zeros_like(values[0])
        groups=ev.summarize_arrays(*values,reference=reference)
        self.assertEqual(groups['left_hand']['all']['change_from_step_zero']['mae'],1.)
        self.assertNotIn('acceptance_passed',groups)

    def test_validation_only_ranking_and_earliest_tie(self):
        records=[dict(passed=True,split='validation',checkpoint_global_step=step,group_balanced_normalized_mae=score)
                 for step,score in zip(ev.RECIPE['checkpoint_steps'],[.6,.4,.4,.5])]
        self.assertEqual(ev.rank_checkpoints(records)[0]['checkpoint_global_step'],4096)
        for changed in [records[:3],[dict(x,split='train') for x in records],
                        [dict(x,group_balanced_normalized_mae=float('nan')) for x in records]]:
            with self.assertRaises(ValueError):ev.rank_checkpoints(changed)

    def test_model_semantics_excludes_only_loading_metadata(self):
        a={'_name_or_path':'before','torch_dtype':'float32','state_dropout_prob':0.,'action_horizon':40}
        b={**a,'_name_or_path':'after','torch_dtype':'bfloat16'}
        self.assertEqual(ev.model_semantics(a),ev.model_semantics(b))
        b['action_horizon']=16
        self.assertNotEqual(ev.model_semantics(a),ev.model_semantics(b))


class ProtocolTests(unittest.TestCase):
    def conversion_fixture(self, directory):
        root=Path(directory)
        split_path=root/'split.json'
        split={'artifact_type':'frozen_dataset_split','seed':42,
               'splits':{'train':['a'],'validation':['b']},
               'episodes':[dict(episode=ep,split=name,derived_episode_index=0,rows=30)
                           for name,ep in [('train','a'),('validation','b')]]}
        split_path.write_text(json.dumps(split))
        roots={name:root/name for name in ['train','validation']}
        manifests={name:root/(name+'-manifest.json') for name in roots}
        for name,ep in [('train','a'),('validation','b')]:
            (roots[name]/'meta').mkdir(parents=True)
            (roots[name]/'data.parquet').write_bytes((name+' verified rows').encode())
            for filename in ['stats.json','relative_stats.json']:
                (roots[name]/'meta'/filename).write_text(json.dumps({'split':name}))
            manifests[name].write_text(json.dumps({'episodes':[dict(episode=ep,rows=30)]}))
        inventories={name:ev.file_inventory(path) for name,path in roots.items()}
        conversion={'schema_version':1,'artifact_type':'larger_dataset_conversion',
                    'split_manifest':ev.binding(split_path),
                    'splits':{name:{'dataset_path':str(roots[name]),'snapshot_manifest':ev.binding(manifests[name]),
                                    'files_sha256':inventories[name]} for name in roots},
                    'statistics':{'owner':'train','stats_json':ev.binding(roots['train']/'meta/stats.json'),
                                  'relative_stats_json':ev.binding(roots['train']/'meta/relative_stats.json')}}
        conversion_path=root/'conversion.json';conversion_path.write_text(json.dumps(conversion))
        return conversion_path,split_path,manifests,roots,inventories

    def test_conversion_rejects_replaced_added_or_missing_dataset_files(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.conversion_fixture(directory)
            ev.validate_conversion(*args)
            for role in ['train','validation']:
                for change in ['replace','add','remove']:
                    inventories=copy.deepcopy(args[-1])
                    if change=='replace':inventories[role]['data.parquet']='0'*64
                    elif change=='add':inventories[role]['unverified.parquet']='0'*64
                    else:del inventories[role]['data.parquet']
                    with self.subTest(role=role,change=change),self.assertRaisesRegex(ValueError,'inventory'):
                        ev.validate_conversion(*args[:-1],inventories)

    def test_conversion_rejects_wrong_split_snapshot_path_and_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.conversion_fixture(directory);path=args[0];original=ev.read(path)
            mutations=[
                lambda x:x.update(schema_version=2),
                lambda x:x.update(artifact_type='unverified_conversion'),
                lambda x:x['split_manifest'].update(sha256='0'*64),
                lambda x:x['splits']['train']['snapshot_manifest'].update(sha256='0'*64),
                lambda x:x['splits']['validation']['snapshot_manifest'].update(path=str(args[2]['train'])),
                lambda x:x['splits']['validation'].update(dataset_path=str(args[3]['train'])),
                lambda x:x['statistics'].update(owner='validation'),
                lambda x:x['statistics'].update(stats_json=ev.binding(args[3]['validation']/'meta/stats.json')),
                lambda x:x['statistics'].update(relative_stats_json=ev.binding(args[3]['validation']/'meta/relative_stats.json')),
                lambda x:x['statistics']['relative_stats_json'].update(sha256='0'*64),
            ]
            for i,mutate in enumerate(mutations):
                changed=copy.deepcopy(original);mutate(changed);path.write_text(json.dumps(changed))
                with self.subTest(case=i),self.assertRaises(ValueError):ev.validate_conversion(*args)
            path.write_text(json.dumps(original))
            # An identical document at a different path is not the declared source snapshot.
            copy_path=Path(directory)/'copied-manifest.json';copy_path.write_bytes(args[2]['train'].read_bytes())
            manifests={**args[2],'train':copy_path}
            with self.assertRaisesRegex(ValueError,'binding'):
                ev.validate_conversion(args[0],args[1],manifests,args[3],args[4])

    def test_conversion_handoff_fails_before_processor_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            conversion,split,manifests,roots,_=self.conversion_fixture(directory)
            (roots['validation']/'data.parquet').write_bytes(b'replaced after verified conversion')
            args=SimpleNamespace(train_dataset=roots['train'],validation_dataset=roots['validation'],
                train_manifest=manifests['train'],validation_manifest=manifests['validation'],
                split_manifest=split,conversion_manifest=conversion,
                processor_output=Path(directory)/'processor',protocol=Path(directory)/'protocol.json')
            with patch.object(ev,'ensure_python',return_value={}),patch.object(ev,'guard_derived',side_effect=lambda path,config:Path(path)),patch.object(ev,'bootstrap_processor') as bootstrap:
                with self.assertRaisesRegex(ValueError,'inventory'):ev.prepare(args,{})
                bootstrap.assert_not_called()
                self.assertFalse(args.processor_output.exists())
                self.assertFalse(args.protocol.exists())

    def test_loaded_protocol_rechecks_conversion_content_and_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            conversion,split,manifests,roots,inventories=self.conversion_fixture(directory)
            bindings={name+'_manifest':ev.binding(path) for name,path in manifests.items()}
            bindings.update(split_manifest=ev.binding(split),conversion_manifest=ev.binding(conversion),
                train_statistics=ev.binding(roots['train']/'meta/stats.json'),
                train_relative_statistics=ev.binding(roots['train']/'meta/relative_stats.json'))
            processor=root/'processor';processor.mkdir()
            for name in ev.PROCESSOR_FILES:(processor/name).write_text('{}')
            semantics=processor/'semantics.json';semantics.write_text('{}')
            samples={}
            for name in roots:
                sample=root/(name+'-samples.json');sample.write_text('{}')
                arrays=sample.with_suffix('.npz');arrays.write_bytes(b'frozen target array fixture')
                samples[name]={**ev.binding(sample),'arrays':ev.binding(arrays)}
            value={'schema_version':1,'artifact_type':'larger_dataset_evaluation_protocol',
                'recipe':ev.RECIPE,'sampling':ev.SAMPLING,
                'target_change':{'method':ev.TEMPORAL_METHOD,'threshold':ev.THRESHOLD},
                'bindings':bindings,'source_sha256':{p:ev.digest(ev.PROJECT_ROOT/p) for p in ev.SOURCE_FILES},
                'datasets':{name:{'path':str(path),'files_sha256':inventories[name]} for name,path in roots.items()},
                'processor':{'path':str(processor),'statistics_owner':'train',
                    'files_sha256':{p:ev.digest(processor/p) for p in ev.PROCESSOR_FILES},
                    'semantics_file':ev.binding(semantics),'semantics_sha256':ev.semantic_digest({})},
                'samples':samples}
            path=root/'protocol.json';path.write_text(json.dumps(value))
            ev.load_protocol(path)
            changed=ev.read(conversion);changed['splits']['train']['files_sha256']['data.parquet']='0'*64
            conversion.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'binding differs: conversion_manifest'):ev.load_protocol(path)
            value['bindings']['conversion_manifest']=ev.binding(conversion)
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'Conversion dataset inventory differs'):ev.load_protocol(path)
            del value['bindings']['conversion_manifest'];path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'Missing or extra'):ev.load_protocol(path)

    def test_hash_binding_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stats.json';path.write_text('{}')
            protocol={'bindings':{'train_statistics':ev.binding(path)}}
            ev.verify_binding(protocol,'train_statistics',path)
            path.write_text('{"different":true}')
            with self.assertRaises(ValueError):ev.verify_binding(protocol,'train_statistics',path)

    def test_protocol_rejects_reintroducing_pilot_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'protocol.json'
            recipe=copy.deepcopy(ev.RECIPE);recipe['mandatory_improvement_threshold']=.1
            path.write_text(json.dumps({'schema_version':1,'artifact_type':'larger_dataset_evaluation_protocol','recipe':recipe,'sampling':ev.SAMPLING}))
            with self.assertRaisesRegex(ValueError,'recipe'):ev.load_protocol(path)

    def test_paired_policy_loaded_once_and_cannot_cross_identity(self):
        shared={};model=Path('/tmp/model');processor=Path('/tmp/processor');policy=object()
        with patch.object(ev,'load_policy',return_value=policy) as loader:
            self.assertIs(ev.shared_policy(shared,model,processor,False,'protocol'),policy)
            self.assertIs(ev.shared_policy(shared,model,processor,False,'protocol'),policy)
            self.assertEqual(loader.call_count,1)
            for changed in [(Path('/tmp/other'),processor,False,'protocol'),(model,Path('/tmp/other'),False,'protocol'),(model,processor,False,'other')]:
                with self.assertRaises(ValueError):ev.shared_policy(shared,*changed)

    def test_remote_inventory_path_never_hashes_weights_in_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory=Path(directory)/'inventory.json';inventory.write_text('{}')
            record={'files':[{'path':'model.safetensors','sha256':'f'*64}]}
            args=SimpleNamespace(step_zero=False,checkpoint_inventory=inventory)
            real_digest=ev.digest
            def small_only(path):
                self.assertEqual(Path(path),inventory)
                return real_digest(path)
            with patch('archive_gr00t_dataset.load_checkpoint_inventory',return_value=record) as verifier,patch.object(ev,'digest',side_effect=small_only):
                shared={}
                first=ev.checkpoint_evidence(args,{},Path('/tmp/checkpoint'),shared)
                second=ev.checkpoint_evidence(args,{},Path('/tmp/checkpoint'),shared)
                self.assertEqual(first,second);self.assertEqual(verifier.call_count,2)
                inventory.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'inventory changed'):ev.checkpoint_evidence(args,{},Path('/tmp/checkpoint'),shared)

    def test_local_step_zero_hashes_once_then_detects_metadata_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory);weights=model/'model.safetensors';weights.write_bytes(b'fake weights')
            protocol={'pretrained':{'files_sha256':{'model.safetensors':ev.digest(weights)}}}
            shared={};args=SimpleNamespace(step_zero=True)
            with patch.object(ev,'digest',wraps=ev.digest) as hashing:
                ev.checkpoint_evidence(args,protocol,model,shared)
                ev.checkpoint_evidence(args,protocol,model,shared)
                self.assertEqual(hashing.call_count,1)
                weights.write_bytes(b'changed fake weights')
                with self.assertRaises(ValueError):ev.checkpoint_evidence(args,protocol,model,shared)


@unittest.skipUnless(os.environ.get('VLA_TEST_PROCESSOR_INTEGRATION')=='1','Opt-in local cached-processor CPU equivalence check')
class BootstrapEquivalenceTests(unittest.TestCase):
    def test_bootstrap_matches_pinned_training_processor_without_model_loading(self):
        from project_config import load_runtime
        config=ev.load_config();runtime=load_runtime(config)
        pretrained=Path(runtime['models']['nvidia/GR00T-N1.7-3B']['snapshot'])
        dataset=ROOT/'derived/lerobot/local/g1_inspire_pilot_v1'
        original=Path.cwd()
        os.chdir(ev.prepare_offline_workdir(config,runtime))
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                bootstrap=ev.bootstrap_processor(pretrained,dataset,root/'bootstrap')
                from gr00t.configs.base_config import get_default_config
                from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
                from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
                from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics
                from gr00t.data.embodiment_tags import EmbodimentTag
                tag=EmbodimentTag.NEW_EMBODIMENT.value
                conf=get_default_config().load_dict({'data':{'download_cache':False,'datasets':[{'dataset_paths':[str(dataset)],'mix_ratio':1.,'embodiment_tag':tag}]}})
                conf.model.state_dropout_prob=0.;conf.model.use_relative_action=True
                conf.model.random_rotation_angle=None;conf.model.color_jitter_params=None
                conf.training.start_from_checkpoint=str(pretrained)
                conf.training.transformers_local_files_only=True
                conf.validate()
                captured=[]
                class TrainOnlyFactory:
                    def __init__(self,config):self.config=config
                    def build(self,processor):
                        captured.extend(self.config.data.datasets[0].dataset_paths)
                        stats=LeRobotEpisodeLoader(dataset_path=dataset,modality_configs=conf.data.modality_configs[tag]).get_dataset_statistics()
                        merged={key:merge_statistics([stats[key]],[1.],is_relative_stats=key=='relative_action') for key in ['state','action','relative_action']}
                        processor.set_statistics({tag:merged})
                        class Dataset:
                            def get_dataset_statistics(self):return {tag:merged}
                        return Dataset(),None
                scratch=root/'pipeline';scratch.mkdir()
                pipeline=Gr00tN1d7Pipeline(conf,scratch)
                with patch('gr00t.model.gr00t_n1d7.setup.DatasetFactory',TrainOnlyFactory):
                    pipeline._create_dataset(scratch)
                pipeline.processor.save_pretrained(root/'training')
                self.assertEqual(captured,[str(dataset)])
                self.assertIsNone(pipeline.model)
                for filename in ev.PROCESSOR_FILES:
                    self.assertEqual((root/'bootstrap'/filename).read_bytes(),(root/'training'/filename).read_bytes(),filename)
                a,_,_=ev.processor_semantics(bootstrap,root/'bootstrap')
                b,_,_=ev.processor_semantics(pipeline.processor,root/'training')
                self.assertEqual(ev.semantic_digest(a),ev.semantic_digest(b))
        finally:os.chdir(original)


if __name__=='__main__':unittest.main()
