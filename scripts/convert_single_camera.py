#!/usr/bin/env python3
"""Use pinned Unitree conversion on a verified, project-owned single-camera pilot."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from prepare_pilot import PROJECT, validate_snapshot
from project_config import load_config, resolve_path, guard_derived, load_runtime, runtime_environment


def git_output(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def guard_output(parent, repo_id, config=None, allow_v3=False):
    config = config or load_config()
    parent = guard_derived(parent, config)
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", repo_id):
        raise ValueError("repo-id must be namespace/name using letters, digits, _ or -")
    target = guard_derived(parent / repo_id, config)
    for path in [target.with_name(target.name + "_v3.0"),
                 target.with_name(target.name + "_v2.1")]:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing existing conversion destination: {path}")
    if allow_v3:
        if not target.is_dir() or target.is_symlink():
            raise ValueError("v2.1 conversion requires an existing ordinary derived v3 directory")
        if json.loads((target / "meta/info.json").read_text())["codebase_version"] != "v3.0":
            raise ValueError("Only v3.0 datasets can be converted to v2.1")
    elif target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing conversion destination: {target}")
    return target


def ensure_python(config, role):
    """Enter the recorded interpreter with native-library paths set at process startup."""
    runtime = load_runtime(config)
    python = Path(runtime["interpreters"][role]).absolute()
    env = runtime_environment(config, offline=True)
    # runtime_environment prepends tool directories; make repeated calls idempotent.
    for key in ["PATH", "LD_LIBRARY_PATH"]:
        env[key] = ":".join(dict.fromkeys(env.get(key, "").split(":")))
    startup_keys = ["PATH", "LD_LIBRARY_PATH", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"]
    identity = {"role": role, "python": str(python), "config": str(config["config_path"]),
                "startup_environment": {key: env.get(key) for key in startup_keys}}
    marker = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ready = (Path(sys.executable).absolute() == python.absolute()
             and os.environ.get("VLA_RUNTIME_STARTUP") == marker
             and all(os.environ.get(key) == env.get(key) for key in startup_keys))
    env["VLA_RUNTIME_STARTUP"] = marker
    if not ready:
        os.execve(str(python), [str(python), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]], env)
    # update() alone would retain credential keys intentionally omitted by the helper.
    os.environ.clear()
    os.environ.update(env)
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--unitree-root", type=Path)
    parser.add_argument("--unitree-commit", help="Full locked SHA; defaults to locks/runtime-sources.json")
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--to-v21", action="store_true", help="Convert the existing derived v3 dataset with NVIDIA upstream")
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    args.output_parent = resolve_path(args.output_parent or config["paths"]["derived"] / "lerobot")
    target = guard_output(args.output_parent, args.repo_id, config, args.to_v21)
    pilot = guard_derived(args.pilot_root, config)
    if args.to_v21:
        validate_snapshot(pilot)
        runtime = load_runtime(config)
        source = runtime["sources"]["gr00t"]
        locked = json.loads((PROJECT / "locks/runtime-sources.json").read_text())["Isaac-GR00T"]["commit"]
        if git_output(source, "rev-parse", "HEAD") != locked:
            raise ValueError("NVIDIA converter source differs from its lock")
        subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=True)
        guard_output(args.output_parent, args.repo_id, config, allow_v3=True)
        subprocess.run([str(runtime["interpreters"]["conversion"]),
                        str(source / "scripts/lerobot_conversion/convert_v3_to_v2.py"),
                        "--root", str(args.output_parent), "--repo-id", args.repo_id],
                       cwd=PROJECT, env=runtime_environment(config, offline=True), check=True)
        print(f"Converted to v2.1: {target}; original v3 retained beside it.")
        return
    ensure_python(config, "unitree")
    manifest = validate_snapshot(pilot)
    root = resolve_path(args.unitree_root or config["sources"]["unitree"])
    args.unitree_commit = args.unitree_commit or json.loads(
        (PROJECT / "locks/runtime-sources.json").read_text())["unitree_lerobot"]["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", args.unitree_commit):
        raise ValueError("An exact lowercase full commit SHA is required")
    actual_commit = git_output(root, "rev-parse", "HEAD")
    if actual_commit != args.unitree_commit:
        raise ValueError(f"Unitree commit differs from lock: {actual_commit}")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=True)
    submodules = git_output(root, "submodule", "status", "--recursive")
    if any(line[:1] in {"-", "+", "U"} for line in submodules.splitlines()):
        raise ValueError("Unitree submodules do not match their recorded gitlinks")

    # Set this before any LeRobot import. Upstream otherwise uses its default cache.
    os.environ["HF_LEROBOT_HOME"] = str(args.output_parent.resolve())
    sys.path.insert(0, str(root))
    import lerobot
    if not Path(lerobot.__file__).resolve().is_relative_to(root / "unitree_lerobot" / "lerobot"):
        raise ValueError("Install Unitree's locked embedded LeRobot with pip -e first")
    from lerobot.utils.constants import HF_LEROBOT_HOME
    if Path(HF_LEROBOT_HOME).resolve() != args.output_parent.resolve():
        raise ValueError("LeRobot cache root did not honor the guarded output parent")
    from unitree_lerobot.utils.constants import ROBOT_CONFIGS
    from unitree_lerobot.utils.convert_unitree_json_to_lerobot import (
        DatasetConfig, create_empty_dataset, populate_dataset,
    )
    original = ROBOT_CONFIGS["Unitree_G1_Inspire"]
    if len(original.motors) != 26:
        raise ValueError("Upstream motor configuration changed")
    expected = ["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"]
    if original.json_state_data_name != expected or original.json_action_data_name != expected:
        raise ValueError("Upstream joint ordering changed")
    ROBOT_CONFIGS["Unitree_G1_Inspire"] = dataclasses.replace(
        original, cameras=["cam_left_high"],
        camera_to_image_key={"color_0": "cam_left_high"},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Repeat the guard immediately before invoking upstream's destructive-on-rerun API.
    guard_output(args.output_parent, args.repo_id, config)
    dataset = create_empty_dataset(
        repo_id=args.repo_id, robot_type="Unitree_G1_Inspire",
        dataset_config=DatasetConfig(image_writer_processes=2, image_writer_threads=2),
    )
    try:
        populate_dataset(dataset, pilot / "raw", robot_type="Unitree_G1_Inspire")
    finally:
        dataset.finalize()
    (target / "conversion_provenance.json").write_text(json.dumps({
        "unitree_commit": actual_commit, "submodules": submodules,
        "pilot_manifest": str(pilot / "manifest.json"),
        "episodes": [{"episode": row["episode"], "content_sha256": row["content_sha256"]}
                     for row in manifest["episodes"]],
        "camera": {"color_0": "cam_left_high"}, "push_to_hub": False,
    }, indent=2) + "\n")
    print(f"Converted to {target}; validate v3 metadata before the v2.1 conversion.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Conversion refused or failed: {error}")
