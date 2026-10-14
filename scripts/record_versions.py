#!/usr/bin/env python3
"""Record reproducibility facts without network calls or credential inspection."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from project_config import PROJECT_ROOT as PROJECT, load_config, resolve_path, guard_derived, guard_path, runtime_environment


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "unavailable": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", action="append", default=[], type=Path,
                        help="Repeat for each pinned checkout/submodule root")
    args = parser.parse_args()
    config = load_config(args.config)
    output = resolve_path(args.output)
    if output.is_relative_to(config["paths"]["reports"]):
        output = guard_path(output, config, "reports")
    else:
        output = guard_derived(output, config)
    if output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing existing record: {output}")
    packages = {}
    for name in ["Pillow", "numpy", "torch", "torchvision", "torchcodec", "flash-attn",
                 "transformers", "lerobot", "gr00t", "huggingface-hub"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    repositories = []
    for root in args.repo:
        root = resolve_path(root)
        repositories.append({
            "path": str(root),
            "commit": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
            "submodules": command(["git", "-C", str(root), "submodule", "status", "--recursive"]),
            "status": command(["git", "-C", str(root), "status", "--porcelain"]),
        })
    usage = shutil.disk_usage(PROJECT)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "disk": {"path": str(PROJECT), "free_bytes": usage.free, "total_bytes": usage.total},
        "tools": {name: shutil.which(name) for name in ["git", "uv", "conda", "ffmpeg",
                                                       "nvidia-smi", "scp", "rsync"]},
        "ffmpeg": command(["ffmpeg", "-version"]),
        "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader"]),
        "repositories": repositories,
        "model_revisions": {key: value["revision"] for key, value in json.loads(
            (PROJECT / "locks/model-sources.json").read_text()).items()},
        "pin_status": "Immutable model/source locks; installed runtime is recorded by setup_pipeline.py",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"Version recording failed: {error}")
