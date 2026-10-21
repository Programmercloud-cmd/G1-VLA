"""Continuation lineage, loading, recipe and native full-state safety contracts."""
from dataclasses import dataclass, field, asdict
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_gr00t_continuation as continuation


@dataclass
class Data:
    seed: int = 42
    identity: dict = field(default_factory=lambda: {'split':42,'nested':[1,2]})


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    training: dict = field(default_factory=lambda: {'resume':False,'learning_rate':5e-5})


class ContinuationTests(unittest.TestCase):
    def test_native_seed_adapter_changes_only_seed_and_rejects_unexpected_base(self):
        config = Config();before = asdict(config)
        proof = continuation.configure_segment(config)
        expected = copy.deepcopy(before);expected['data']['seed'] = 4138
        self.assertEqual(asdict(config), expected)
        self.assertEqual(proof['upstream_config_before_override_sha256'], proof['config_with_base_seed_sha256'])
        self.assertEqual(proof['changed_fields'], ['data.seed'])
        with self.assertRaisesRegex(ValueError, 'Unexpected upstream base seed'):
            continuation.configure_segment(config)

    def test_command_loads_learned_weights_with_fresh_optimizer_and_full_state_saves(self):
        runtime = {'interpreters':{'gr00t':Path('/python')},'sources':{'gr00t':Path('/source')},
                   'models':{'nvidia/GR00T-N1.7-3B':{'snapshot':Path('/pretrained')}}}
        command = continuation.continuation_command({},runtime,Path('/train'),Path('/new'),98,Path('/old/checkpoint-4096'))
        values = {'--base-model-path':'/old/checkpoint-4096','--max-steps':'4096','--save-steps':'2048',
                  '--save-total-limit':'2','--learning-rate':'5e-05','--warmup-ratio':'0.025',
                  '--gradient-accumulation-steps':'8','--state-dropout-prob':'0.0'}
        for flag,value in values.items():self.assertEqual(command[command.index(flag)+1],value)
        self.assertIn('--no-save-only-model',command);self.assertNotIn('--save-only-model',command)
        self.assertIn('--no-resume-from-checkpoint',command)

    def test_strict_load_rejects_any_unloaded_tensor_and_wrong_source(self):
        proof = continuation.validate_loading_info('/learned/checkpoint-4096',{},'/learned/checkpoint-4096')
        self.assertTrue(proof['strict_tensor_loading'])
        for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'every source tensor'):
                continuation.validate_loading_info('/learned/checkpoint-4096',{key:['mask_token']},'/learned/checkpoint-4096')
        with self.assertRaisesRegex(ValueError,'did not use checkpoint4096'):
            continuation.validate_loading_info('/pretrained',{},'/learned/checkpoint-4096')

    def test_local_and_cumulative_progress_are_separate_and_bounded(self):
        self.assertEqual([continuation.cumulative_step(x) for x in [0,2048,4096]],[4096,6144,8192])
        for value in [-1,4097,True,0.5]:
            with self.assertRaises(ValueError):continuation.cumulative_step(value)

    def test_full_state_output_guard_requires_120_gib_and_retains_runtime_reserve(self):
        g = continuation.GIB
        self.assertTrue(all(continuation.check_resources(20*g,20*g,120*g,60*g,49140,48660).values()))
        with self.assertRaisesRegex(RuntimeError,'output'):
            continuation.check_resources(20*g,20*g,119*g,60*g,49140,48660)
        with self.assertRaisesRegex(RuntimeError,'project'):
            continuation.check_resources(7*g,20*g,200*g,60*g,49140,48660,startup=False)

    def test_remote_storage_operation_is_bounded_by_remaining_deadline(self):
        with patch.object(continuation.subprocess,'run') as run:
            run.return_value.returncode=0
            continuation.remote_python_bounded(['ssh','host'],'print(1)','/path with spaces',deadline=time.monotonic()+2)
            self.assertGreater(run.call_args.kwargs['timeout'],0)
            self.assertLessEqual(run.call_args.kwargs['timeout'],2)
            self.assertIn("'/path with spaces'",run.call_args.args[0][-1])
            run.reset_mock()
            with self.assertRaises(TimeoutError):
                continuation.remote_python_bounded(['ssh','host'],'print(1)',deadline=time.monotonic()-1)
            run.assert_not_called()

    def test_parent_aborts_if_independent_watchdog_exits(self):
        self.assertIsNone(continuation.require_watchdog_alive(SimpleNamespace(poll=lambda:None)))
        with self.assertRaisesRegex(RuntimeError,'watchdog exited'):
            continuation.require_watchdog_alive(SimpleNamespace(poll=lambda:7))

    def test_worker_requires_armed_live_watchdog_for_its_exact_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='armed-fixture-') as folder:
            path=Path(folder)/'watchdog.json'
            proof={'status':'armed','run':'fixture','cap_seconds':21600,'watchdog_pid':9,
                   'watchdog_identity':{'pid':9},'processes':{'worker':{'pid':8}},
                   'monotonic_deadline':time.monotonic()+100}
            snapshot=Path(folder)/'watchdog-armed.json';snapshot.write_text(json.dumps(proof))
            path.write_text(json.dumps(proof));manifest={'cap_watchdog_report':str(path),
                'cap_watchdog_startup_report':str(snapshot),'run':'fixture','timeout_seconds':21600}
            import watch_training_cap as cap
            with patch.object(cap,'inspect',return_value={'pid':8}),patch.object(cap,'same_process',return_value=True):
                self.assertTrue(continuation.check_armed_watchdog(manifest,worker_pid=8)['passed'])
                proof['status']='watchdog_error_no_further_signals';path.write_text(json.dumps(proof))
                with self.assertRaisesRegex(ValueError,'snapshot differs'):
                    continuation.check_armed_watchdog(manifest,worker_pid=8)
            proof['status']='armed';path.write_text(json.dumps(proof))
            with patch.object(cap,'inspect',return_value=None),self.assertRaisesRegex(ValueError,'worker identity'):
                continuation.check_armed_watchdog(manifest,worker_pid=8)


class NativeStateTests(unittest.TestCase):
    def test_actual_upstream_configuration_adapter_changes_only_native_seed(self):
        try:
            from gr00t.configs.base_config import get_default_config
        except ImportError:
            self.skipTest('Use configured GR00T Python for actual configuration checks')
        config=get_default_config();proof=continuation.configure_segment(config)
        self.assertEqual(config.data.seed,4138)
        self.assertEqual(proof['config_with_base_seed_sha256'],proof['upstream_config_before_override_sha256'])

    def test_pinned_cpu_trainer_saves_and_restores_native_state_but_reseeds_loader(self):
        try:
            import torch
            from transformers import TrainingArguments, TrainerCallback
            from gr00t.experiment.trainer import Gr00tTrainer
        except ImportError:
            self.skipTest('Use configured GR00T Python for the pinned native CPU checkpoint test')

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.0))
            def forward(self,features):return {'loss':(self.weight*features).square().mean()}

        class Samples(torch.utils.data.Dataset):
            seed=4138
            resets=[]
            def __len__(self):return 8
            def __getitem__(self,index):return {'features':torch.tensor([1.0])}
            def reset_seed(self,seed):self.resets.append(seed);self.seed=seed

        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def arguments(path):
                return TrainingArguments(output_dir=str(path),use_cpu=True,max_steps=2,
                    per_device_train_batch_size=1,save_steps=1,save_only_model=False,
                    logging_strategy='no',report_to=[],disable_tqdm=True,dataloader_num_workers=0,
                    optim='adamw_torch',learning_rate=5e-5,remove_unused_columns=False)
            first=Gr00tTrainer(model=Tiny(),args=arguments(root/'first'),train_dataset=Samples())
            first.train()
            checkpoint=root/'first/checkpoint-1'
            proof=continuation.validate_full_state(checkpoint,1,require_cuda=False)
            self.assertTrue(proof['passed']);self.assertEqual(proof['scheduler_epoch'],1)
            self.assertFalse(proof['exact_iterable_cursor_restore'])
            observed=[]
            class Observe(TrainerCallback):
                def on_train_begin(self,args,state,control,optimizer=None,lr_scheduler=None,**kwargs):
                    observed.append((state.global_step,len(optimizer.state),lr_scheduler.last_epoch))
            samples=Samples()
            resumed=Gr00tTrainer(model=Tiny(),args=arguments(root/'second'),train_dataset=samples)
            resumed.add_callback(Observe());resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(observed,[(1,1,1)])
            self.assertIn(4139,samples.resets)
            self.assertEqual(resumed.state.global_step,2)
            self.assertEqual(int(next(iter(resumed.optimizer.state.values()))['step']),2)
            self.assertTrue(torch.equal(first.model.weight,resumed.model.weight))
            with self.assertRaisesRegex(ValueError,'honest local progress'):
                continuation.validate_full_state(checkpoint,6145,require_cuda=False)
            self.assertEqual(json.loads((checkpoint/'trainer_state.json').read_text())['global_step'],1)


if __name__ == '__main__':unittest.main()
