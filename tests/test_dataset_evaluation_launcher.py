"""CPU-only checks for the additive evaluator launcher and its native guard."""
import argparse
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import launch_gr00t_evaluation as launcher


class LauncherTests(unittest.TestCase):
    def test_import_restores_exact_startup_paths(self):
        config = {'config_path': ROOT / 'machine.local.json'}
        def importing(name):
            self.assertEqual(name, 'cv2')
            os.environ['PATH'] = 'changed-path'
            os.environ['LD_LIBRARY_PATH'] = 'changed-library-path'
        with patch.dict(os.environ, {'PATH': 'native-path', 'LD_LIBRARY_PATH': 'native-libs'}), \
             patch.object(launcher.importlib, 'import_module', side_effect=importing), \
             patch.object(launcher, 'ensure_python', return_value='runtime') as guard:
            self.assertEqual(launcher.preload_native_video(config), 'runtime')
            self.assertEqual(os.environ['PATH'], 'native-path')
            self.assertEqual(os.environ['LD_LIBRARY_PATH'], 'native-libs')
            guard.assert_called_once_with(config, 'gr00t')

    def test_failed_import_restores_missing_and_present_values(self):
        def importing(name):
            os.environ['PATH'] = 'changed'
            os.environ['LD_LIBRARY_PATH'] = 'new-value'
            raise ImportError('fixture failure')
        with patch.dict(os.environ, {'PATH': 'native-path'}, clear=True), \
             patch.object(launcher.importlib, 'import_module', side_effect=importing), \
             patch.object(launcher, 'ensure_python') as guard:
            with self.assertRaisesRegex(ImportError, 'fixture failure'):
                launcher.preload_native_video({})
            self.assertEqual(os.environ['PATH'], 'native-path')
            self.assertNotIn('LD_LIBRARY_PATH', os.environ)
            guard.assert_not_called()

    def test_forwards_canonical_config_and_exact_arguments(self):
        config = {'config_path': ROOT / 'reports/machine-fixture.json'}
        arguments = ['evaluate', '--split', 'both', '--protocol', 'unchanged.json', '--step-zero']
        previous = sys.argv
        expected = [str(ROOT / 'scripts/evaluate_gr00t_dataset.py'), '--config',
                    str(config['config_path']), *arguments]
        def execute(path, run_name):
            self.assertEqual(sys.argv, expected)
            self.assertEqual(path, expected[0])
            self.assertEqual(run_name, '__main__')
        with patch.object(launcher.runpy, 'run_path', side_effect=execute):
            launcher.forward_evaluator(config, arguments)
        self.assertIs(sys.argv, previous)

    def test_main_keeps_runtime_guard_and_forwards_after_preload(self):
        config = {'config_path': ROOT / 'reports/machine-fixture.json'}
        arguments = ['evaluate', '--split', 'both', '--step-zero']
        with patch.object(launcher, 'load_config', return_value=config), \
             patch.object(launcher, 'ensure_python') as guard, \
             patch.object(launcher, 'preload_native_video') as preload, \
             patch.object(launcher, 'forward_evaluator') as forward:
            launcher.main(['--config', 'reports/machine-fixture.json', *arguments])
            guard.assert_called_once_with(config, 'gr00t')
            preload.assert_called_once_with(config)
            forward.assert_called_once_with(config, arguments)


def actual_runtime_preflight():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-preflight', action='store_true', required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = launcher.load_config(args.config)
    # This initial call may genuinely re-exec into the pinned interpreter/env.
    launcher.ensure_python(config, 'gr00t')
    saved = {name: os.environ.get(name) for name in ('PATH', 'LD_LIBRARY_PATH')}
    # PID is insufficient proof: execve preserves it. A restart attempt must fail.
    with patch.object(os, 'execve', side_effect=AssertionError('Unexpected runtime restart')):
        launcher.preload_native_video(config)
        launcher.ensure_python(config, 'gr00t')
        launcher.importlib.import_module('cv2')
        launcher.ensure_python(config, 'gr00t')
    assert {name: os.environ.get(name) for name in saved} == saved
    assert 'cv2' in sys.modules
    print(json.dumps({'passed': True, 'original_runtime_guard_repeated': True,
                      'restart_attempts_fail_test': True, 'cached_cv2_import_checked': True,
                      'scope': 'Actual pinned CPU/native runtime only; no model or GPU inference.'}))


if __name__ == '__main__':
    if '--runtime-preflight' in sys.argv:
        actual_runtime_preflight()
    else:
        unittest.main(verbosity=2)
