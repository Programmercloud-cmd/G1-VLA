#!/usr/bin/env python3
"""Preload OpenCV before invoking the unchanged GR00T dataset evaluator."""
import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

from convert_single_camera import ensure_python
from project_config import PROJECT_ROOT, load_config


def preload_native_video(config):
    """Cache cv2, restoring the native startup search paths even on failure."""
    saved = {name: os.environ.get(name) for name in ('PATH', 'LD_LIBRARY_PATH')}
    try:
        importlib.import_module('cv2')
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    # Keep the original validator active. It must accept the restored environment.
    return ensure_python(config, 'gr00t')


def forward_evaluator(config, arguments):
    target = PROJECT_ROOT / 'scripts/evaluate_gr00t_dataset.py'
    previous = sys.argv
    try:
        sys.argv = [str(target), '--config', str(config['config_path']), *arguments]
        return runpy.run_path(str(target), run_name='__main__')
    finally:
        sys.argv = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--preflight-only', action='store_true',
                        help='Check runtime and cached OpenCV import without model/GPU inference')
    parser.add_argument('evaluator_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_python(config, 'gr00t')
    preload_native_video(config)
    if args.preflight_only:
        if args.evaluator_arguments:
            parser.error('--preflight-only does not take evaluator arguments')
        print(json.dumps({'passed': True, 'cv2_cached': 'cv2' in sys.modules,
                          'scope': 'Native runtime/OpenCV preflight only; no model or GPU inference.'}))
        return
    if not args.evaluator_arguments:
        parser.error('Evaluator arguments are required')
    forward_evaluator(config, args.evaluator_arguments)


if __name__ == '__main__':
    main()
