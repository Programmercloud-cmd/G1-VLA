"""Extension recipe and honest two-native-segment ancestry regression checks."""
from collections import Counter
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_gr00t_extension as extension


class ExtensionTests(unittest.TestCase):
    def test_native_seed_changes_only_seed_and_rejects_second_override(self):
        from gr00t.configs.base_config import get_default_config
        config=get_default_config();before=asdict(config)
        proof=extension.configure_segment(config)
        expected=copy.deepcopy(before);expected['data']['seed']=8234
        self.assertEqual(asdict(config),expected)
        self.assertEqual(proof['changed_fields'],['data.seed'])
        self.assertEqual(proof['upstream_config_before_override_sha256'],proof['config_with_base_seed_sha256'])
        with self.assertRaisesRegex(ValueError,'Unexpected upstream base seed'):
            extension.configure_segment(config)

    def test_command_uses_full_source_weights_but_fresh_native_optimizer(self):
        runtime={'interpreters':{'gr00t':Path('/python')},'sources':{'gr00t':Path('/source')},
                 'models':{'nvidia/GR00T-N1.7-3B':{'snapshot':Path('/pretrained')}}}
        command=extension.extension_command({},runtime,Path('/train'),Path('/new'),98,Path('/full8192/checkpoint-4096'))
        for flag,value in {'--base-model-path':'/full8192/checkpoint-4096','--max-steps':'4096',
                           '--save-steps':'2048','--save-total-limit':'2','--learning-rate':'2.5e-05',
                           '--warmup-ratio':'0.025','--gradient-accumulation-steps':'8',
                           '--state-dropout-prob':'0.0','--num-shards-per-epoch':'98'}.items():
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertIn('--no-resume-from-checkpoint',command)
        self.assertIn('--no-save-only-model',command)

    def test_model_loading_is_strict_from8192_not_pretrained(self):
        proof=extension.validate_loading_info('/full8192/checkpoint-4096',{},'/full8192/checkpoint-4096')
        self.assertEqual(proof['source_updates'],8192)
        for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'every source tensor'):
                extension.validate_loading_info('/full8192/checkpoint-4096',{key:['mask_token']},'/full8192/checkpoint-4096')
        with self.assertRaisesRegex(ValueError,'did not use cumulative8192'):
            extension.validate_loading_info('/pretrained',{},'/full8192/checkpoint-4096')

    def test_local_counts_never_fabricate_cumulative_trainer_state(self):
        self.assertEqual([extension.cumulative_step(x) for x in [0,2048,4096]],[8192,10240,12288])
        for value in [-1,4097,True,0.5]:
            with self.assertRaises(ValueError):extension.cumulative_step(value)

    def test_native_cosine_fresh_cycle_reaches103_warmup_and_finishes_zero(self):
        import torch
        from transformers import get_scheduler
        parameter=torch.nn.Parameter(torch.tensor(1.))
        optimizer=torch.optim.AdamW([parameter],lr=extension.LEARNING_RATE)
        scheduler=get_scheduler('cosine',optimizer,num_warmup_steps=103,num_training_steps=4096)
        self.assertFalse(optimizer.state);self.assertEqual(scheduler.last_epoch,0)
        self.assertEqual(optimizer.param_groups[0]['lr'],0.)
        for step in range(1,4097):
            optimizer.step();scheduler.step()
            if step==103:self.assertEqual(optimizer.param_groups[0]['lr'],2.5e-5)
        self.assertEqual(scheduler.last_epoch,4096)
        self.assertEqual(optimizer.param_groups[0]['lr'],0.)


class PriorLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory();cls.root=Path(cls.folder.name)
        cls.episodes=[{'episode':'episode_fixture','rows':18}]
        def write(path,value):path.write_text(json.dumps(value));return path
        def build(name,row,extra=0):
            audit=cls.root/(name+'-finite.jsonl');coverage=cls.root/(name+'-coverage.json')
            counts=extension.CoverageAudit(cls.episodes);counts.counts=Counter({(0,row):32768})
            data=counts.summary()
            if name=='parent':
                lineage=extension.CoverageAudit(cls.episodes);lineage.counts=Counter({(0,0):32768,(0,1):32768})
                data['lineage_coverage']=lineage.summary()
            write(coverage,data)
            with audit.open('w') as stream:
                for step in range(4096+extra):
                    for micro in range(8):
                        stream.write(json.dumps({'event':'micro_loss','microbatch':step*8+micro+1,
                            'step_before':step,'loss':.1,'finite':True,'episode_index':0,
                            'row':row if step<4096 else 2})+'\n')
                    stream.write(json.dumps({'event':'gradient','step':step+1,'post_clip_norm':1.,'finite':True})+'\n')
                    stream.write(json.dumps({'event':'step_end','step':step+1})+'\n')
                    if step==4095:stream.write(json.dumps({'event':'checkpoint','step':4096,'passed':True,
                        'coverage_path':str(coverage),'coverage_sha256':extension.digest(coverage)})+'\n')
            return {'audit':str(audit),'episodes':copy.deepcopy(cls.episodes)}, {'finite_audit_sha256':extension.digest(audit)}
        origin,origin_report=build('origin',0,extra=1135)
        cls.origin_path=write(cls.root/'origin-run.json',origin)
        cls.origin_report_path=write(cls.root/'origin-train.json',origin_report)
        prefix,_=extension.previous.source_prefix(origin,origin_report)
        cls.parent,cls.report=build('parent',1)
        cls.parent.update(source_run_manifest=str(cls.origin_path),source_train_report=str(cls.origin_report_path),
                          cumulative_base_step=4096,cumulative_target_step=8192,source_prefix=prefix)
        cls.report.update(passed=True,local_optimizer_updates=4096,cumulative_lineage_updates=8192)
        cls.parent_path=write(cls.root/'parent-run.json',cls.parent)
        cls.report['run_manifest']=str(cls.parent_path)
        cls.report_path=write(cls.root/'parent-train.json',cls.report)

    @classmethod
    def tearDownClass(cls):cls.folder.cleanup()

    def test_two4096_prefixes_preserve65536_exposures_and_exclude_unsaved1135(self):
        proof,records=extension.source_prefix(self.parent,self.report,self.parent_path,self.report_path)
        self.assertEqual([(x['cumulative_base_step'],x['local_updates'],x['cumulative_end_step'])
                          for x in proof['native_segments']],[(0,4096,4096),(4096,4096,8192)])
        self.assertTrue(all(set(x['audit'])=={'path','sha256','bytes'} for x in proof['native_segments']))
        losses=[x for x in records if x['event']=='micro_loss']
        self.assertEqual(len(losses),65536);self.assertEqual({x['row'] for x in losses},{0,1})
        self.assertEqual(proof['lineage_coverage']['consumed_examples'],65536)
        self.assertEqual(proof['lineage_coverage']['unique_window_starts'],2)
        self.assertTrue(proof['source_unsaved1135_updates_excluded'])
        self.assertTrue(all(x['finite_prefix']['finite_summary']['last_step']==4096 for x in proof['native_segments']))

    def test_different_episode_identity_rejected_before_combining(self):
        changed=copy.deepcopy(self.parent);changed['episodes'][0]['episode']='wrong_episode'
        with self.assertRaisesRegex(ValueError,'different episode ordering'):
            extension.source_prefix(changed,self.report,self.parent_path,self.report_path)

    def test_false8192_native_base_rejected(self):
        changed=copy.deepcopy(self.parent);changed['cumulative_base_step']=8192
        with self.assertRaisesRegex(ValueError,'completed prior4096-to8192'):
            extension.source_prefix(changed,self.report,self.parent_path,self.report_path)


if __name__=='__main__':unittest.main()
