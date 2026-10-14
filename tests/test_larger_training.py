"""Bounded larger-dataset recipe, split isolation, and consumed-sample audits."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_gr00t_dataset as runner


def snapshot(names):
    return {'episodes': [{'episode': name, 'rows': 40, 'derived_episode_index': i,
                          'source_data_json_sha256': str(i), 'effective_goal': 'reviewed'}
                         for i, name in enumerate(names)]}


class LargerRecipeTests(unittest.TestCase):
    def test_four_saves_pretrained_and_fixed_schedule(self):
        runtime = {'interpreters': {'gr00t': Path('/python')}, 'sources': {'gr00t': Path('/source')},
                   'models': {'nvidia/GR00T-N1.7-3B': {'snapshot': '/pretrained'}}}
        command = runner.dataset_command({}, runtime, Path('/train'), Path('/run'), 98)
        for flag, value in [('--base-model-path', '/pretrained'), ('--max-steps', '8192'),
                            ('--save-steps', '2048'), ('--save-total-limit', '4'),
                            ('--global-batch-size', '1'), ('--gradient-accumulation-steps', '8'),
                            ('--num-shards-per-epoch', '98'), ('--state-dropout-prob', '0.0')]:
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertIn('--no-resume-from-checkpoint', command)
        self.assertIn('--save-only-model', command)
        with self.assertRaises(ValueError): runner.dataset_command({}, runtime, Path('/train'), Path('/run'), 0)

    def test_output_reserve_is_80_not_pilot_35(self):
        args = [9*runner.GIB, 9*runner.GIB, 80*runner.GIB, 60*runner.GIB, 49140, 40960]
        self.assertTrue(all(runner.check_resources(*args).values()))
        for index, bad in [(0, 7*runner.GIB), (1, 7*runner.GIB), (2, 79*runner.GIB),
                           (3, 59*runner.GIB), (4, 40000), (5, 40000)]:
            changed = args.copy(); changed[index] = bad
            with self.subTest(index=index), self.assertRaises(RuntimeError): runner.check_resources(*changed)

    def test_full_window_geometry_and_split_isolation(self):
        train = snapshot(sorted(runner.PILOT_EPISODES))
        val = snapshot(['episode_0050'])
        split = {'schema_version': 1, 'artifact_type': 'frozen_dataset_split', 'seed': 42,
                 'splits': {'train': [e['episode'] for e in train['episodes']], 'validation': ['episode_0050']},
                 'episodes': [dict(e, split=role) for role,s in [('train',train),('validation',val)] for e in s['episodes']]}
        self.assertEqual(runner.validate_split(split, train, val), ([25]*5, 1))
        bad = copy.deepcopy(val); bad['episodes'][0]['effective_goal'] = 'unreviewed'
        with self.assertRaisesRegex(ValueError, 'source/label'): runner.validate_split(split, train, bad)
        bad = copy.deepcopy(train); bad['episodes'][0]['rows'] = 15
        with self.assertRaisesRegex(ValueError, '16-target'): runner.validate_split(split, bad, val)
        bad = copy.deepcopy(split); bad['splits']['train'] = list(reversed(bad['splits']['train']))
        with self.assertRaisesRegex(ValueError, 'order'): runner.validate_split(bad, train, val)

    def test_coverage_counts_consumption_not_preloading_or_lookahead(self):
        audit = runner.CoverageAudit(snapshot(['episode_0037', 'episode_0040'])['episodes'])
        points = [{}, {}, {}]
        audit.register(points, [(0, 1), (0, 2), (1, 3)])
        self.assertEqual(audit.summary()['consumed_examples'], 0)
        audit.collated([points[0]]); audit.collated([points[1]])
        self.assertEqual(audit.consume(), {'episode_index': 0, 'row': 1})
        result = audit.summary()
        self.assertEqual(result['consumed_examples'], 1)
        self.assertEqual(result['unique_window_starts'], 1)
        self.assertEqual(result['collated_but_not_consumed'], 1)
        audit.consume(); audit.collated([points[0]]); audit.consume()
        self.assertEqual(audit.summary()['consumed_examples'], 3)
        self.assertEqual(audit.summary()['unique_window_starts'], 2)
        self.assertEqual(audit.summary()['episodes_observed'], 1)
        with self.assertRaises(ValueError): audit.consume()
        with self.assertRaises(ValueError): audit.collated([{}])
        with self.assertRaises(ValueError): audit.register([{}], [(0, 25)])

    def test_baseline_requires_same_protocol_split_processor_and_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol_path = root/'protocol.json'; protocol_path.write_text('{}')
            report = root/'reference.json'; arrays = report.with_suffix('.npz'); arrays.write_bytes(b'fixture')
            protocol = {'samples': {'train': {'sha256': 'samples'}},
                        'processor': {'semantics_sha256': 'semantics', 'files_sha256': {'statistics.json': 'stats'}}}
            value = {'passed': True, 'role': 'step_zero_reference', 'checkpoint_global_step': 0,
                     'split': 'train', 'evaluation_protocol_sha256': runner.digest(protocol_path),
                     'processor_semantics_sha256': 'semantics', 'statistics_sha256': 'stats',
                     'sample_manifest_sha256': 'samples', 'artifacts': {arrays.name: runner.digest(arrays)}}
            report.write_text(json.dumps(value))
            self.assertEqual(len(runner.verify_baseline(report, 'train', protocol_path, protocol)), 2)
            for key,bad in [('split','validation'), ('checkpoint_global_step',2048), ('statistics_sha256','other')]:
                changed = dict(value, **{key:bad}); report.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError): runner.verify_baseline(report,'train',protocol_path,protocol)
            report.write_text(json.dumps(value)); arrays.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'arrays'): runner.verify_baseline(report,'train',protocol_path,protocol)

    def test_runtime_rejects_validation_data_dropout_and_changed_train_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            payloads={name:'{}' for name in runner.PROCESSOR_FILES}
            def save(path):
                path.mkdir(parents=True,exist_ok=True)
                for name,value in payloads.items(): (path/name).write_text(value)
            save(root/'bootstrap')
            training=NS(max_steps=8192,global_batch_size=1,gradient_accumulation_steps=8,
                        save_steps=2048,save_total_limit=4,save_only_model=True,learning_rate=.0001,
                        warmup_ratio=.05,num_gpus=1,dataloader_num_workers=0,remove_unused_columns=False,
                        resume_from_checkpoint=False,skip_weight_loading=False,start_from_checkpoint=str(root/'model'))
            data=NS(seed=42,shard_size=256,episode_sampling_rate=1.,num_shards_per_epoch=1,override_pretraining_statistics=True)
            model=NS(state_dropout_prob=0.,tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True)
            class Leaf:
                dataset_path=root/'train'
                episode_loader=NS(episode_lengths=[40])
                def __len__(self): return 1
            pipeline=NS(config=NS(training=training,data=data,model=model),
                        model=NS(config=NS(state_dropout_prob=0.),action_head=NS(state_dropout_prob=0.)),
                        processor=NS(state_dropout_prob=0.,save_pretrained=save),
                        train_dataset=NS(datasets=[Leaf()]),eval_dataset=None)
            manifest={'pretrained':{'path':str(root/'model')},'output':str(root/'out'),
                      'train_dataset':str(root/'train'),'num_shards_per_epoch':1,
                      'episodes':snapshot(['episode_0037'])['episodes'],
                      'processor':{'files_sha256':{n:runner.digest(root/'bootstrap'/n) for n in runner.PROCESSOR_FILES}}}
            self.assertTrue(runner.check_runtime(pipeline,manifest)['passed'])
            pipeline.eval_dataset=object()
            with self.assertRaisesRegex(ValueError,'only the declared training'): runner.check_runtime(pipeline,manifest)
            pipeline.eval_dataset=None;pipeline.processor.state_dropout_prob=.2
            with self.assertRaisesRegex(ValueError,'Zero state dropout'): runner.check_runtime(pipeline,manifest)
            pipeline.processor.state_dropout_prob=0.;payloads['statistics.json']='{"changed":true}'
            with self.assertRaisesRegex(ValueError,'train-only bootstrap'): runner.check_runtime(pipeline,manifest)

    def test_extra_dataset_file_or_symlink_cannot_be_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').write_text('x')
            runner.verify_dataset_inventory(root,{'data':'fixture'})
            (root/'extra.parquet').write_text('x')
            with self.assertRaisesRegex(ValueError,'inventory'): runner.verify_dataset_inventory(root,{'data':'fixture'})
            (root/'extra.parquet').unlink();(root/'link').symlink_to(root/'data')
            with self.assertRaisesRegex(ValueError,'symlinks'): runner.verify_dataset_inventory(root,{'data':'fixture'})


class ActualLargerAuditTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from transformers import TrainingArguments, TrainerCallback
            from gr00t.experiment.trainer import Gr00tTrainer
            from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
            from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
        except ImportError:
            self.skipTest('Use configured GR00T interpreter for actual CPU audit tests')
        self.torch, self.Args, self.Callback, self.Trainer = torch, TrainingArguments, TrainerCallback, Gr00tTrainer
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        for cls,name in [(Gr00tTrainer,'__init__'),(Gr00tTrainer,'compute_loss'),(Gr00tTrainer,'train'),
                         (Gr00tN1d7Pipeline,'setup'),(ShardedSingleStepDataset,'get_shard')]:
            original = getattr(cls,name); self.addCleanup(setattr,cls,name,original)
        self.path = self.root/'audit.jsonl'; self.stream = self.path.open('w'); self.addCleanup(self.stream.close)
        self.coverage = runner.install_audits({'episodes':snapshot(['episode_0037'])['episodes']}, self.stream)

    def train(self, nonfinite=None):
        torch = self.torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, features):
                loss=(self.weight*features).square().mean()
                if nonfinite=='loss': loss=loss*torch.tensor(float('nan'))
                return {'loss':loss}
        model=Tiny()
        if nonfinite=='gradient': model.weight.register_hook(lambda g:g*float('nan'))
        points=[{'features':torch.tensor([1.0])} for _ in range(4)]
        self.coverage.register(points,[(0,i) for i in range(4)])
        def collator(features):
            self.coverage.collated(features)
            return {'features':torch.stack([f['features'] for f in features])}
        args=self.Args(output_dir=str(self.root/'out'),use_cpu=True,max_steps=2,
                       per_device_train_batch_size=1,gradient_accumulation_steps=2,save_strategy='no',
                       logging_strategy='no',report_to=[],disable_tqdm=True,dataloader_num_workers=0,
                       remove_unused_columns=False)
        trainer=self.Trainer(model=model,args=args,train_dataset=points,data_collator=collator)
        # Mimic upstream adding checkpoint-format callback after construction.
        trainer.add_callback(self.Callback())
        trainer.train(); self.stream.flush()
        return trainer

    def test_actual_trainer_finite_audit_and_consumed_coverage(self):
        trainer=self.train()
        records=runner.read_audit(self.path)
        self.assertTrue(runner.finite_summary(records,2,2)['passed'])
        self.assertEqual(self.coverage.summary()['consumed_examples'],4)
        self.assertEqual([r['event'] for r in records].count('fresh_optimizer'),1)
        self.assertEqual(trainer.callback_handler.callbacks[-1].__class__.__name__,'Audits')

    def test_nonfinite_loss_aborts(self):
        with self.assertRaises(FloatingPointError): self.train('loss')
        self.assertEqual(self.coverage.summary()['consumed_examples'],0)

    def test_nonfinite_gradients_abort(self):
        with self.assertRaises(FloatingPointError): self.train('gradient')
        self.assertFalse(any(r['event']=='step_end' for r in runner.read_audit(self.path)))


if __name__ == '__main__':
    unittest.main()
