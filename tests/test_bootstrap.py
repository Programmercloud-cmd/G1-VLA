"""Meaningful path isolation, dry-run and pinned reuse regression checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import project_config as pc
import setup_pipeline as setup
import verify_pipeline as readiness


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.patch=patch.object(pc,'PROJECT_ROOT',self.root);self.patch.start()
    def tearDown(self):
        self.patch.stop();self.temp.cleanup()
    def config(self, supplied=None):
        path=self.root/'machine.json';path.write_text(json.dumps(supplied or {}));return pc.load_config(path)
    def test_relative_paths_are_checkout_relative(self):
        c=self.config({'paths':{'derived':'results','cache':'shared-cache'}})
        self.assertEqual(c['paths']['derived'],self.root/'results')
        self.assertEqual(c['paths']['hf_cache'],self.root/'shared-cache/huggingface/hub')
    def test_interpreter_symlink_is_not_resolved(self):
        link=self.root/'env/bin/python';link.parent.mkdir(parents=True);link.symlink_to(sys.executable)
        c=self.config({'interpreters':{'unitree':'env/bin/python'}})
        self.assertEqual(c['interpreters']['unitree'],link)
        self.assertNotEqual(c['interpreters']['unitree'],link.resolve())
    def test_raw_output_and_symlink_escape_rejected(self):
        c=self.config();derived=c['paths']['derived'];derived.mkdir();outside=self.root/'outside';outside.mkdir();(derived/'escape').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):pc.guard_derived(derived/'escape/file',c)
        with self.assertRaises(ValueError):pc.guard_derived(c['paths']['raw_data']/'file',c)
    def test_optional_local_archive_needs_no_mount(self):
        c=self.config();self.assertEqual(pc.guard_path(self.root/'storage/checkpoints/a',c,'archive'),self.root/'storage/checkpoints/a')
    def test_enabled_storage_requires_mounted_source(self):
        c=self.config({'storage':{'enabled':True,'host':'host','remote_path':'/data'}})
        with patch.object(pc.os.path,'ismount',return_value=False):
            with self.assertRaises(ValueError):pc.guard_path(self.root/'storage/file',c,'archive')
    def test_environment_drops_credentials(self):
        c=self.config()
        with patch.dict(os.environ,{'HF_TOKEN':'test-secret'}):
            self.assertNotIn('HF_TOKEN',pc.runtime_environment(c))
    def test_invalid_low_or_nan_reserve_rejected(self):
        for value in [7,float('nan')]:
            with self.assertRaises(ValueError):self.config({'reserve_gib':value})
    def test_existing_mismatched_environment_is_not_modified(self):
        with patch.object(setup,'installed_versions',return_value={'python':'3.10.12','packages':{}}):
            ok,mismatch,_=setup.environment_matches('unitree',Path('/unused'))
            self.assertFalse(ok);self.assertIn('torch',mismatch)
    def test_portable_requirements_keep_cpu_torch(self):
        c=self.config();rendered=setup.render_requirements('unitree',c)
        self.assertIn('torch==2.7.1+cpu',rendered)
        self.assertIn('-e '+str(c['sources']['unitree']),rendered)
        self.assertNotIn('/home/test2',rendered)
    def test_only_download_child_receives_environment_token(self):
        c=self.config();process=Mock();process.poll.return_value=0;process.returncode=0
        with patch.object(setup,'reserve'), patch.object(setup.subprocess,'Popen',return_value=process) as popen, patch.dict(os.environ,{'HF_TOKEN':'dummy-token'}):
            setup.command(['true'],c)
            self.assertNotIn('HF_TOKEN',popen.call_args.kwargs['env'])
            setup.command(['true'],c,model_credentials=True)
            self.assertEqual(popen.call_args.kwargs['env']['HF_TOKEN'],'dummy-token')
    def test_training_resource_shortfall_fails(self):
        c=self.config()
        with patch.object(readiness.Path,'read_text',return_value='MemTotal: 1024 kB\nMemAvailable: 512 kB\n'):
            with self.assertRaises(ValueError):readiness.resources(c,training=True)
    def test_selected_small_gpu_fails_training_readiness(self):
        selected={'device_index':0,'total_bytes':24*1024**3,'free_bytes':23*1024**3}
        with self.assertRaises(ValueError):readiness.validate_gpu_capacity(selected,training=True)
        self.assertEqual(readiness.validate_gpu_capacity(selected,training=False),selected)
    def test_dataset_missing_or_wrong_dimensions_fails(self):
        c=self.config();dataset=c['paths']['derived']/'pilot';meta=dataset/'meta';meta.mkdir(parents=True)
        with self.assertRaises(ValueError):readiness.dataset_metadata(dataset,c)
        for name in ['info.json','modality.json','stats.json','relative_stats.json','episodes.jsonl','tasks.jsonl']:(meta/name).write_text('{}')
        (meta/'info.json').write_text(json.dumps({'codebase_version':'v2.1','total_episodes':1,'features':{'observation.state':{'shape':[14]},'action':{'shape':[26]}}}))
        with self.assertRaises(ValueError):readiness.dataset_metadata(dataset,c)
    def test_plan_does_not_create_configured_directories(self):
        c=self.config()
        with patch.object(setup,'model_status',return_value=[]):result=setup.plan(c,['tools'])
        self.assertEqual(result['mode'],'plan');self.assertFalse(c['paths']['tools'].exists())


if __name__=='__main__':unittest.main()
