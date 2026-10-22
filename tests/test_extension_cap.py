"""Exercise the new runner identity using the unchanged local watchdog engine."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import watch_training_cap as engine
import watch_extension_cap as adapter


class ExtensionCapTests(unittest.TestCase):
    def test_adapter_changes_only_runner_and_rejects_old_worker_identity(self):
        old=ROOT/'scripts/run_gr00t_continuation.py';new=ROOT/'scripts/run_gr00t_extension.py'
        with patch.object(engine,'RUNNER',old),patch.object(engine,'MANIFEST',Path('/fixture')):
            before=dict(vars(engine));adapter.configure_runner();self.assertEqual(engine.RUNNER,new)
            self.assertTrue(all(v is getattr(engine,k) for k,v in before.items() if k!='RUNNER'))
            record={'uid':os.getuid(),'argv':[sys.executable,str(old),'--worker','/fixture'],
                    'pid':10,'pgrp':10,'session':10}
            with self.assertRaisesRegex(ValueError,'Unexpected runner'):engine.validate_role(record,'worker')
            record['argv'][1]=str(new);engine.validate_role(record,'worker')
            with self.assertRaisesRegex(ValueError,'Unexpected original'):adapter.configure_runner()
        self.assertEqual(engine.RUNNER,old)

    def test_real_canonical_bootstrap_preserves_new_worker_process_identity(self):
        from project_config import load_config,runtime_environment
        with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='extension-bootstrap-') as folder:
            folder=Path(folder);config_path=folder/'config.json';config_path.write_text(json.dumps({'interpreters':{'gr00t':sys.executable}}))
            environment=runtime_environment(load_config(config_path),offline=True);environment.pop('VLA_RUNTIME_STARTUP',None)
            script=folder/'bootstrap.py';before=folder/'before.json';after=folder/'after.json';manifest=folder/'manifest.json'
            script.write_text("import os,sys,json;from pathlib import Path\n"+f"sys.path.insert(0,{str(ROOT/'scripts')!r})\n"+
                "from convert_single_camera import ensure_python\nfrom project_config import load_config\nimport watch_training_cap as cap\n"+
                "config,before,after=map(Path,sys.argv[1:4])\nif not before.exists():before.write_text(json.dumps(cap.inspect(os.getpid())))\n"+
                "ensure_python(load_config(config),'gr00t')\nafter.write_text(json.dumps(cap.inspect(os.getpid())))\n")
            command=[sys.executable,str(script),str(config_path),str(before),str(after),str(ROOT/'scripts/run_gr00t_extension.py'),'--worker',str(manifest)]
            result=subprocess.run(command,cwd=ROOT,env=environment,start_new_session=True,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            first,last=json.loads(before.read_text()),json.loads(after.read_text());self.assertTrue(engine.same_process(first,last))
            with patch.object(engine,'RUNNER',ROOT/'scripts/run_gr00t_continuation.py'),patch.object(engine,'MANIFEST',manifest):
                adapter.configure_runner();engine.validate_role(last,'worker')

    def test_real_adapter_caps_only_two_identity_bound_disposable_processes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='extension-cap-') as folder:
            folder=Path(folder);pidfile=folder/'child.json';manifest=folder/'manifest.json';manifest.write_text('{}')
            runner=str(ROOT/'scripts/run_gr00t_extension.py')
            code=("import json,subprocess,sys,time;from pathlib import Path;"+
                  f"p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',{runner!r},'--worker',{str(manifest)!r}],start_new_session=True);"+
                  f"Path({str(pidfile)!r}).write_text(json.dumps(p.pid));time.sleep(60)")
            launcher=subprocess.Popen([sys.executable,'-c',code,runner,'--run','extension-cap-fixture','--timeout-seconds','1'],start_new_session=True)
            child=None
            try:
                until=time.monotonic()+5
                while not pidfile.exists() and time.monotonic()<until:time.sleep(.02)
                child=json.loads(pidfile.read_text());report=folder/'watchdog.json'
                result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/watch_extension_cap.py'),
                    '--launcher-pid',str(launcher.pid),'--worker-pid',str(child),'--manifest',str(manifest),
                    '--report',str(report),'--run','extension-cap-fixture','--cap-seconds','1'],capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr);proof=json.loads(report.read_text())
                self.assertEqual(proof['status'],'cap_enforced');self.assertFalse(proof['external_storage_access_in_loop'])
                self.assertEqual([(x['role'],x['pid']) for x in proof['signals']],[('worker',child),('launcher',launcher.pid)])
                self.assertTrue(any(x.endswith('watch_extension_cap.py') for x in proof['watchdog_identity']['argv']))
                self.assertEqual(proof['script_sha256'],__import__('hashlib').sha256((ROOT/'scripts/watch_training_cap.py').read_bytes()).hexdigest())
                launcher.wait(timeout=5)
            finally:
                for pid in [child,launcher.pid]:
                    if pid is not None:
                        try:os.killpg(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                launcher.wait(timeout=5)


if __name__=='__main__':unittest.main()
