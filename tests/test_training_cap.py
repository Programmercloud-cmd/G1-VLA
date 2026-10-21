"""Local deadline protection signals only identity-bound disposable fixtures."""
import copy
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
import watch_training_cap as cap


class LocalCapTests(unittest.TestCase):
    def test_real_runtime_exec_preserves_canonical_worker_identity_without_B(self):
        from project_config import load_config, runtime_environment
        with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='bootstrap-fixture-') as folder:
            folder=Path(folder);config_path=folder/'config.json'
            config_path.write_text(json.dumps({'interpreters':{'gr00t':sys.executable}}))
            config=load_config(config_path);environment=runtime_environment(config,offline=True)
            environment.pop('VLA_RUNTIME_STARTUP',None)
            self.assertEqual(environment['PYTHONDONTWRITEBYTECODE'],'1')
            script=folder/'bootstrap.py'
            script.write_text(
                "import json,os,sys;from pathlib import Path\n"
                f"sys.path.insert(0,{str(ROOT/'scripts')!r})\n"
                "from project_config import load_config\n"
                "from convert_single_camera import ensure_python\n"
                "import watch_training_cap as cap\n"
                "config,before,after=map(Path,sys.argv[1:])\n"
                "if not before.exists():before.write_text(json.dumps(cap.inspect(os.getpid())))\n"
                "ensure_python(load_config(config),'gr00t')\n"
                "after.write_text(json.dumps(cap.inspect(os.getpid())))\n")
            for old_B in [False,True]:
                before=folder/f'before-{old_B}.json';after=folder/f'after-{old_B}.json'
                command=[sys.executable,*(['-B'] if old_B else []),str(script),str(config_path),str(before),str(after)]
                result=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
                first,last=json.loads(before.read_text()),json.loads(after.read_text())
                for field in ['pid','uid','start_ticks','pgrp','session']:
                    self.assertEqual(first[field],last[field])
                if old_B:
                    self.assertEqual([x for x in first['argv'] if x!='-B'],last['argv'])
                    with self.assertRaisesRegex(ValueError,'PID identity changed'):
                        cap.same_process(first,last)
                else:
                    self.assertEqual(first['argv'],last['argv'])
                    self.assertTrue(cap.same_process(first,last))

    def test_recycled_pid_identity_cannot_be_signaled(self):
        expected={'pid':123,'uid':os.getuid(),'start_ticks':100,'cmdline_sha256':'a',
                  'pgrp':123,'session':123,'state':'S'}
        for field in ['start_ticks','cmdline_sha256','uid','pgrp','session']:
            actual=copy.deepcopy(expected);actual[field]='different'
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'PID identity changed'):
                cap.same_process(expected,actual)

    def test_fuse_fallback_refuses_a_non_D_worker_without_any_signal(self):
        expected={'pid':123,'uid':os.getuid(),'start_ticks':100,'cmdline_sha256':'a',
                  'pgrp':123,'session':123,'state':'S'}
        record={'processes':{'worker':expected},'sshfs_fallback':{'process':{},'mount':{}}}
        with patch.object(cap,'inspect',return_value=expected),patch.object(cap.os,'kill') as kill:
            self.assertFalse(cap.release_blocked_fuse(record));kill.assert_not_called()

    def test_real_local_watchdog_caps_only_its_two_disposable_processes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='cap-fixture-') as folder:
            folder=Path(folder);pidfile=folder/'child.json';manifest=folder/'manifest.json';manifest.write_text('{}')
            runner=str(ROOT/'scripts/run_gr00t_continuation.py')
            code=("import json,subprocess,sys,time;from pathlib import Path;"
                  f"p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',{runner!r},'--worker',{str(manifest)!r}],start_new_session=True);"
                  f"Path({str(pidfile)!r}).write_text(json.dumps(p.pid));time.sleep(60)")
            launcher=subprocess.Popen([sys.executable,'-c',code,runner,'--run','cap-fixture','--timeout-seconds','1'],
                                      start_new_session=True)
            child=None
            try:
                until=time.monotonic()+5
                while not pidfile.exists() and time.monotonic()<until:time.sleep(.02)
                child=json.loads(pidfile.read_text());report=folder/'watchdog.json'
                result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/watch_training_cap.py'),
                    '--launcher-pid',str(launcher.pid),'--worker-pid',str(child),
                    '--manifest',str(manifest),'--report',str(report),'--run','cap-fixture','--cap-seconds','1'],
                    cwd=ROOT,capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
                proof=json.loads(report.read_text());self.assertEqual(proof['status'],'cap_enforced')
                initial=json.loads(report.with_name(report.stem+'-armed.json').read_text())
                self.assertEqual(initial['status'],'armed');self.assertEqual(initial['signals'],[])
                self.assertEqual(initial['watchdog_identity'],proof['watchdog_identity'])
                self.assertFalse(proof['external_storage_access_in_loop']);self.assertIsNone(proof['sshfs_fallback'])
                self.assertEqual([(x['role'],x['pid']) for x in proof['signals']],
                                 [('worker',child),('launcher',launcher.pid)])
                launcher.wait(timeout=5)
            finally:
                for pid in [child,launcher.pid]:
                    if pid is None:continue
                    try:os.killpg(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                launcher.wait(timeout=5)


if __name__=='__main__':unittest.main()
