"""Run a bounded pilot fit through NVIDIA's launcher with explicit finite-value auditing."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import time

from project_config import (PROJECT_ROOT as ROOT, load_config, load_runtime, resolve_path,
                            guard_derived, guard_path, runtime_environment)
from prepare_pilot import digest
from convert_single_camera import ensure_python
from run_gr00t_smoke import (GIB, available_memory, existing_ancestor,
                             prepare_offline_workdir, smoke_command)
import shutil


def require_finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite {label}: {value}")
    return value


def pilot_command(config, runtime, dataset, output, steps=512, accumulation=8):
    if not 1 <= steps <= 512 or not 1 <= accumulation <= 8:
        raise ValueError("This probe is bounded to 1–512 updates and accumulation 1–8")
    command = smoke_command(config, runtime, dataset, output)
    for flag, value in [("--max-steps", steps), ("--save-steps", steps),
                        ("--gradient-accumulation-steps", accumulation), ("--num-shards-per-epoch", 5)]:
        command[command.index(flag) + 1] = str(value)
    return command + ["--learning-rate", "0.0001", "--warmup-ratio", "0.05",
                      "--state-dropout-prob", "0.2"]


def finite_summary(records, expected_steps, accumulation):
    losses = [r for r in records if r["event"] == "micro_loss"]
    gradients = [r for r in records if r["event"] == "gradient"]
    steps = [r for r in records if r["event"] == "step_end"]
    valid = (len(losses) == expected_steps * accumulation
             and [r["microbatch"] for r in losses] == list(range(1, expected_steps * accumulation + 1))
             and [r["step_before"] for r in losses] == [i // accumulation for i in range(expected_steps * accumulation)]
             and [r["step"] for r in gradients] == list(range(1, expected_steps + 1))
             and [r["step"] for r in steps] == list(range(1, expected_steps + 1))
             and all(r.get("finite") is True for r in losses + gradients)
             and all(math.isfinite(r["loss"]) for r in losses)
             and all(math.isfinite(r["post_clip_norm"]) for r in gradients)
             and not any(r["event"] == "nonfinite" for r in records))
    values = [r["loss"] for r in losses if math.isfinite(r["loss"])]
    window = min(len(values), max(accumulation * 20, 1))
    return {"passed": valid, "micro_losses_observed": len(losses), "gradient_updates_observed": len(gradients),
            "optimizer_updates_observed": len(steps), "first_window_mean_loss": sum(values[:window])/window if window else None,
            "last_window_mean_loss": sum(values[-window:])/window if window else None,
            "loss_window_microbatches": window, "last_step": steps[-1]["step"] if steps else 0}


def install_finite_checks(audit_path):
    """Observe original losses/gradients without changing their values or vendor files."""
    import torch
    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer
    original = Gr00tTrainer.compute_loss
    stream = audit_path.open("x", buffering=1)
    started = time.monotonic()
    counter = 0

    def emit(record):
        record["elapsed_seconds"] = time.monotonic() - started
        stream.write(json.dumps(record, allow_nan=False) + "\n")

    def checked_value(value, label):
        try:
            return require_finite(value, label)
        except FloatingPointError:
            emit({"event": "nonfinite", "kind": label, "value": repr(float(value))})
            raise

    class FiniteGradients(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            if not gradients:
                raise RuntimeError("No parameter gradients at optimizer step")
            # Called after upstream clipping and before the update. Scalar norms add no model-sized copy.
            norms = torch.stack(list(torch._foreach_norm(gradients))).float()
            norm = checked_value(torch.linalg.vector_norm(norms).item(), "post-clip gradient norm")
            emit({"event": "gradient", "step": state.global_step + 1,
                  "post_clip_norm": norm, "finite": True})

        def on_step_end(self, args, state, control, optimizer=None, **kwargs):
            emit({"event": "step_end", "step": state.global_step,
                  "learning_rate": optimizer.param_groups[0]["lr"]})

    def checked_loss(self, *args, **kwargs):
        nonlocal counter
        if not getattr(self, "_pilot_finite_callback", False):
            self.add_callback(FiniteGradients())
            self.args.logging_nan_inf_filter = False
            self._pilot_finite_callback = True
        result = original(self, *args, **kwargs)
        loss = result[0] if isinstance(result, tuple) else result
        value = checked_value(loss.detach().float().item(), "microbatch loss")
        counter += 1
        emit({"event": "micro_loss", "microbatch": counter,
              "step_before": self.state.global_step, "loss": value, "finite": True})
        return result

    Gr00tTrainer.compute_loss = checked_loss
    return stream


def worker(config, manifest_path):
    ensure_python(config, "gr00t")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["script_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Execution source changed after manifest: " + relative)
    os.chdir(prepare_offline_workdir(config))
    command = manifest["upstream_command"]
    audit = guard_path(manifest["audit"], config, "reports")
    stream = install_finite_checks(audit)
    sys.argv = command[1:]
    try:
        runpy.run_path(command[1], run_name="__main__")
    finally:
        stream.close()


def read_audit(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break  # The writer may be between a JSON write and its newline.
    return records


def stop_process_group(process):
    """The new session belongs to this worker, including surviving descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        process.poll()  # Reap the group leader without assuming its children exited.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def verify_baseline(dataset, pilot_path, sample_path, baseline_path):
    from evaluate_gr00t_pilot import dataset_identity
    identity = dataset_identity(dataset, pilot_path)
    if json.loads(sample_path.read_text()) != identity:
        raise ValueError("Sample manifest does not match the frozen dataset and sampling recipe")
    baseline = json.loads(baseline_path.read_text())
    if (baseline.get("passed") is not True or baseline.get("role") != "two_step_reference"
            or baseline.get("sample_manifest_sha256") != digest(sample_path)
            or Path(baseline.get("dataset", "")).resolve() != dataset
            or baseline.get("sample_count") != 100 or baseline.get("inferences") != 300):
        raise ValueError("Complete the matching finite two-step baseline evaluation before training")
    arrays = baseline_path.with_suffix(".npz")
    if baseline.get("artifacts", {}).get(arrays.name) != digest(arrays):
        raise ValueError("Baseline prediction artifact differs from its recorded hash")
    return identity


def json_finite(value):
    """Keep diagnostic failures serializable without disguising nonfinite values."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--pilot-manifest", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--dry-run", action="store_true", help="Print exact configuration without launching or writing")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.worker:
        worker(config, resolve_path(args.worker))
        return
    for key in ["dataset", "pilot_manifest", "evaluation_manifest", "baseline_report", "run"]:
        if not getattr(args, key):
            parser.error("--" + key.replace("_", "-") + " is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run) or not 1 <= args.timeout_seconds <= 21600:
        raise ValueError("Use a simple run name and timeout within six hours")
    dataset = guard_derived(args.dataset, config)
    pilot_path = guard_derived(args.pilot_manifest, config)
    pilot = json.loads(pilot_path.read_text())
    if len(pilot["episodes"]) != 5:
        raise ValueError("This bounded run uses exactly the frozen five-episode pilot")
    evaluation_manifest = resolve_path(args.evaluation_manifest)
    baseline_path = resolve_path(args.baseline_report)
    identity = verify_baseline(dataset, pilot_path, evaluation_manifest, baseline_path)
    runtime = load_runtime(config)
    groot = runtime["sources"]["gr00t"]
    pins = json.loads((ROOT / "locks/runtime-sources.json").read_text())
    if subprocess.check_output(["git", "-C", str(groot), "rev-parse", "HEAD"], text=True).strip() != pins["Isaac-GR00T"]["commit"]:
        raise ValueError("GR00T checkout differs from immutable source lock")
    subprocess.run(["git", "-C", str(groot), "diff", "--quiet", "HEAD", "--"], check=True)
    output = guard_path(config["paths"]["runs"] / args.run, config, "runs")
    paths = {suffix: guard_path(config["paths"]["reports"] / (args.run + suffix), config, "reports")
             for suffix in ["-run.json", "-train.json", "-train.log", "-finite.jsonl"]}
    command = pilot_command(config, runtime, dataset, output, args.max_steps, args.accumulation)
    scripts = ["scripts/run_gr00t_pilot.py", "scripts/evaluate_gr00t_pilot.py",
               "scripts/run_gr00t_smoke.py", "scripts/convert_single_camera.py", "scripts/project_config.py",
               "configs/g1_inspire_config.py", "configs/modality.json"]
    files = identity["dataset_files_sha256"]
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "run": args.run,
        "repository_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": {name: digest(ROOT / name) for name in scripts},
        "source_lock_sha256": digest(ROOT / "locks/runtime-sources.json"),
        "model_lock_sha256": digest(ROOT / "locks/model-sources.json"),
        "model_revisions": {name: rec["revision"] for name, rec in runtime["models"].items()},
        "dataset": str(dataset), "dataset_files_sha256": files,
        "pilot_manifest": str(pilot_path), "pilot_manifest_sha256": digest(pilot_path),
        "evaluation_manifest": str(evaluation_manifest), "evaluation_manifest_sha256": digest(evaluation_manifest),
        "baseline_report": str(baseline_path), "baseline_report_sha256": digest(baseline_path),
        "output": str(output), "checkpoint": str(output / ("checkpoint-" + str(args.max_steps))), "audit": str(paths["-finite.jsonl"]), "upstream_command": command,
        "seed": 42, "microbatch": 1, "accumulation": args.accumulation, "max_steps": args.max_steps,
        "expected_training_examples": args.max_steps * args.accumulation,
        "timeout_seconds": args.timeout_seconds, "finite_checks": "Every microbatch loss and every post-clip/pre-update gradient norm",
        "scope": "Training-set fitting probe; no holdout, robot success, or instruction-selection claim"}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", *scripts], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "scripts", "configs"], check=True)
    if any(p.exists() or p.is_symlink() for p in [output, *paths.values()]):
        raise FileExistsError("Run outputs/evidence already exist; choose a new run name")
    reserve = config["reserve_gib"] * GIB
    if (shutil.disk_usage(ROOT).free < reserve or shutil.disk_usage(existing_ancestor(paths["-run.json"])).free < reserve
            or shutil.disk_usage(existing_ancestor(output)).free < 35 * GIB or available_memory() < 60 * GIB):
        raise RuntimeError("Insufficient project/report/output/host-memory reserve")
    output.parent.mkdir(parents=True, exist_ok=True)
    paths["-run.json"].parent.mkdir(parents=True, exist_ok=True)
    with paths["-run.json"].open("x") as stream:
        json.dump(manifest, stream, indent=2); stream.write("\n")
    process = None
    started, last_progress = time.monotonic(), 0
    peak_gpu, minimum_host, minimum_output = 0, available_memory(), shutil.disk_usage(output.parent).free
    aborted = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with paths["-train.log"].open("x") as log:
            process = subprocess.Popen([str(runtime["interpreters"]["gr00t"]), str(Path(__file__).resolve()),
                                        "--worker", str(paths["-run.json"])] ,
                                       cwd=ROOT, env=runtime_environment(config, offline=True),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                minimum_host = min(minimum_host, available_memory())
                minimum_output = min(minimum_output, shutil.disk_usage(output.parent).free)
                if (shutil.disk_usage(ROOT).free < reserve or shutil.disk_usage(paths["-run.json"].parent).free < reserve
                        or minimum_output < 8 * GIB or minimum_host < 16 * GIB):
                    aborted = "resource_reserve"
                if time.monotonic() - started > args.timeout_seconds:
                    aborted = "wall_time_limit"
                sample = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True)
                if sample.returncode == 0:
                    peak_gpu = max(peak_gpu, max(int(v) for v in sample.stdout.split()))
                if aborted:
                    stop_process_group(process)
                    break
                if time.monotonic() - last_progress >= 30:
                    audit = read_audit(paths["-finite.jsonl"])
                    steps = [r for r in audit if r["event"] == "step_end"]
                    print(json.dumps({"phase": "training", "step": steps[-1]["step"] if steps else 0,
                                      "elapsed_seconds": round(time.monotonic()-started, 1),
                                      "peak_gpu_mib": peak_gpu, "available_host_gib": available_memory()/GIB}), flush=True)
                    last_progress = time.monotonic()
                time.sleep(2)
    except KeyboardInterrupt:
        aborted = "interrupted"
    finally:
        if process is not None:
            stop_process_group(process)
        records = read_audit(paths["-finite.jsonl"])
        finite = finite_summary(records, args.max_steps, args.accumulation)
        checkpoint = output / ("checkpoint-" + str(args.max_steps))
        state_path = checkpoint / "trainer_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        report = {"passed": process is not None and process.returncode == 0 and not aborted
                  and finite["passed"] and state.get("global_step") == args.max_steps,
                  "checkpoint": str(checkpoint), "run_manifest": str(paths["-run.json"]),
                  "run_manifest_sha256": digest(paths["-run.json"]), "repository_commit": manifest["repository_commit"],
                  "exit_code": process.returncode if process else None, "aborted": aborted,
                  "elapsed_seconds": time.monotonic()-started, "sampled_peak_gpu_mib": peak_gpu,
                  "min_available_host_gib": minimum_host/GIB, "min_output_free_gib": minimum_output/GIB,
                  "finite_audit": finite, "finite_audit_sha256": digest(paths["-finite.jsonl"]) if paths["-finite.jsonl"].exists() else None, "trainer_log_history": json_finite(state.get("log_history", [])),
                  "checkpoint_bytes": sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()),
                  "total_output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}
        with paths["-train.json"].open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
