"""Run the unchanged NVIDIA launcher for exactly two pilot optimization steps."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from project_config import (PROJECT_ROOT as ROOT, load_config, load_runtime, resolve_path,
                            guard_derived, guard_path, runtime_environment)
GIB = 1024**3


def prepare_offline_workdir(config=None, runtime=None):
    """Give the pinned Cosmos repo name a local path for Transformers 4.57.3.

    Its tokenizer's generic Mistral check otherwise calls model_info even offline.
    No downloaded files or vendor code are modified.
    """
    config = config or load_config()
    runtime = runtime or load_runtime(config)
    models = runtime["models"]
    record = models["nvidia/Cosmos-Reason2-2B"]
    snapshot = Path(record["snapshot"]).resolve()
    if snapshot.name != record["revision"] or not (snapshot / "model.safetensors").is_file():
        raise RuntimeError("Pinned Cosmos snapshot is incomplete")
    working = config["paths"]["model_aliases"]
    alias = working / "nvidia/Cosmos-Reason2-2B"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        if not alias.is_symlink() or alias.resolve() != snapshot:
            raise RuntimeError("Existing Cosmos alias differs from the pinned snapshot")
    else:
        alias.symlink_to(snapshot, target_is_directory=True)
    return working


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def existing_ancestor(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def smoke_command(config, runtime, dataset, output):
    """The validated two-step recipe; longer training is a separate milestone."""
    return [str(runtime["interpreters"]["gr00t"]),
        str(runtime["sources"]["gr00t"] / "gr00t/experiment/launch_finetune.py"),
        "--base-model-path", str(runtime["models"]["nvidia/GR00T-N1.7-3B"]["snapshot"]),
        "--dataset-path", str(dataset), "--embodiment-tag", "NEW_EMBODIMENT",
        "--modality-config-path", str(ROOT / "configs/g1_inspire_config.py"),
        "--output-dir", str(output), "--global-batch-size", "1", "--gradient-accumulation-steps", "1",
        "--num-gpus", "1", "--dataloader-num-workers", "0", "--max-steps", "2", "--save-steps", "2",
        "--save-total-limit", "1", "--save-only-model", "--no-use-wandb",
        "--num-shards-per-epoch", "1", "--shard-size", "256", "--episode-sampling-rate", "1.0"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", required=True, help="A new run name; existing outputs are never replaced")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        raise ValueError("Run must be a simple directory name")
    config = load_config(args.config)
    dataset = guard_derived(args.dataset, config)
    for filename in ("info.json", "modality.json", "stats.json", "relative_stats.json"):
        if not (dataset / "meta" / filename).is_file():
            raise FileNotFoundError(dataset / "meta" / filename)
    runtime = load_runtime(config)
    models = runtime["models"]
    if set(models) != {"nvidia/GR00T-N1.7-3B", "nvidia/Cosmos-Reason2-2B"}:
        raise RuntimeError("Both pinned model snapshots must be recorded by setup")
    for record in models.values():
        snapshot = Path(record["snapshot"])
        if snapshot.name != record["revision"] or not (snapshot / "config.json").is_file():
            raise RuntimeError("Recorded model snapshot is incomplete or not pinned")
    output = guard_path(config["paths"]["runs"] / args.run, config, "runs")
    log = guard_path(config["paths"]["reports"] / (args.run + "-train.log"), config, "reports")
    report_path = guard_path(config["paths"]["reports"] / (args.run + "-train.json"), config, "reports")
    if any(path.exists() or path.is_symlink() for path in [output, log, report_path]):
        raise FileExistsError("Choose a new run name; existing artifacts are preserved")
    reserve = config["reserve_gib"] * GIB
    if (shutil.disk_usage(ROOT).free < reserve
            or shutil.disk_usage(existing_ancestor(report_path)).free < reserve
            or shutil.disk_usage(existing_ancestor(output)).free < 35 * GIB
            or available_memory() < 60 * GIB):
        raise RuntimeError("Insufficient project disk, output storage, or host-memory headroom")
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    working = prepare_offline_workdir(config, runtime)
    command = smoke_command(config, runtime, dataset, output)
    record = {"command": command, "working_directory": str(working), "output": str(output), "dataset": str(dataset),
              "offline": True, "checkpoint_durability": "transient_until_archived", "model_revisions": {k:v["revision"] for k,v in models.items()}}
    start = time.monotonic()
    peak_gpu_mib = 0
    min_available = available_memory()
    aborted = None
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=working, env=runtime_environment(config, offline=True), stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            min_available = min(min_available, available_memory())
            if (shutil.disk_usage(ROOT).free < reserve
                    or shutil.disk_usage(report_path.parent).free < reserve
                    or shutil.disk_usage(output.parent).free < 3 * GIB or min_available < 16 * GIB):
                aborted = "resource_reserve"
            if time.monotonic() - start > 1800:
                aborted = "30_minute_timeout"
            if aborted:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            sample = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if sample.returncode == 0:
                peak_gpu_mib = max(peak_gpu_mib, max(int(x) for x in sample.stdout.split()))
            time.sleep(2)
    losses = []
    for match in re.finditer(r"\{[^{}]*'train_loss'[^{}]*\}", log.read_text()):
        try:
            losses.append(float(ast.literal_eval(match.group())["train_loss"]))
        except (ValueError, SyntaxError, KeyError):
            pass
    checkpoint = output / "checkpoint-2"
    state = json.loads((checkpoint / "trainer_state.json").read_text()) if (checkpoint / "trainer_state.json").exists() else {}
    record.update(exit_code=process.returncode, aborted=aborted, elapsed_seconds=time.monotonic()-start,
                  sampled_peak_gpu_mib=peak_gpu_mib, min_available_host_gib=min_available/GIB,
                  train_losses=losses, global_step=state.get("global_step"), checkpoint=str(checkpoint),
                  checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()),
                  total_output_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()))
    record["passed"] = process.returncode == 0 and state.get("global_step") == 2 and bool(losses) and all(math.isfinite(x) for x in losses)
    with report_path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
