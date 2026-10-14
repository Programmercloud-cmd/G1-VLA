"""Archive one reload-validated checkpoint to local storage or compressed SSH transport."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import uuid

from project_config import PROJECT_ROOT as ROOT, load_config, resolve_path, guard_path


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def remote_python(ssh, code, *args):
    command = shlex.join(["python3", "-c", code, *map(str, args)])
    result = subprocess.run(ssh + [command], check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Remote storage operation failed: {result.stderr.strip()}")
    return result


def validate_source(source, evidence):
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Checkpoint source must be an ordinary directory")
    if (evidence.get("passed") is not True or evidence.get("fresh_process") is not True
            or resolve_path(evidence["checkpoint"]) != source):
        raise ValueError("Source must match a successful fresh-process reload report")
    paths = sorted(source.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Checkpoint archives do not follow symlinks")
    if not (source / "config.json").is_file() or not any(source.glob("*.safetensors")):
        raise ValueError("Checkpoint model files are incomplete")
    return [(path, source.name + "/" + str(path.relative_to(source)))
            for path in paths if path.is_file()]



def read_small_json(path, label):
    if path.is_symlink() or not path.is_file() or path.suffix != ".json":
        raise ValueError(f"{label} must be an ordinary JSON file")
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError(f"{label} exceeds the 4 MiB metadata limit")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return record


def pilot_evidence_sources(source, evaluation_report=None, run_config=None, pilot_manifest=None,
                           baseline_report=None, sample_manifest=None):
    """Bind replayable pilot evidence to the one checkpoint being archived."""
    supplied = [evaluation_report, run_config, pilot_manifest, baseline_report, sample_manifest]
    if not any(path is not None for path in supplied):
        return []
    if any(path is None for path in supplied):
        raise ValueError("Pilot archival requires evaluation, run config, snapshot, baseline, and sample manifests together")
    evaluation = read_small_json(evaluation_report, "Evaluation report")
    if (evaluation.get("passed") is not True or not evaluation.get("candidate_checkpoint")
            or resolve_path(evaluation["candidate_checkpoint"]) != source):
        raise ValueError("Pilot evaluation must pass and identify the checkpoint being archived")
    recipe = read_small_json(run_config, "Run config")
    if (not recipe.get("checkpoint") or resolve_path(recipe["checkpoint"]) != source
            or recipe.get("run") != source.parent.name):
        raise ValueError("Run config must identify the checkpoint and run being archived")
    snapshot = read_small_json(pilot_manifest, "Pilot manifest")
    if not isinstance(snapshot.get("episodes"), list) or not snapshot["episodes"]:
        raise ValueError("Pilot manifest must identify the snapshot episodes")
    samples = read_small_json(sample_manifest, "Sample manifest")
    if not isinstance(samples.get("samples"), list) or not samples["samples"]:
        raise ValueError("Sample manifest must identify the paired evaluation samples")
    baseline = read_small_json(baseline_report, "Baseline report")
    if baseline.get("passed") is not True or not baseline.get("candidate_checkpoint"):
        raise ValueError("Baseline evaluation must have passed")
    sample_hash = sha256(sample_manifest)
    for key, path in [("pilot_manifest_sha256", pilot_manifest),
                      ("evaluation_manifest_sha256", sample_manifest),
                      ("baseline_report_sha256", baseline_report)]:
        if recipe.get(key) != sha256(path):
            raise ValueError("Run config input hash mismatch: " + key)
    if any(record.get("sample_manifest_sha256") != sample_hash for record in [evaluation, baseline]):
        raise ValueError("Both evaluations must identify the exact sample manifest")
    if evaluation.get("baseline_report_sha256") != sha256(baseline_report):
        raise ValueError("Candidate evaluation must identify the exact baseline report")
    result = [(evaluation_report, "reports/evaluation.json"),
              (run_config, "reports/run-config.json"),
              (pilot_manifest, "dataset/snapshot-manifest.json"),
              (baseline_report, "reports/baseline.json"),
              (sample_manifest, "reports/sample-manifest.json")]
    for report, stem, record in [(evaluation_report, "evaluation", evaluation),
                                 (baseline_report, "baseline", baseline)]:
        if not report.with_suffix(".npz").is_file():
            raise ValueError("Both evaluations require saved NPZ predictions for replay")
        for suffix in [".npz", ".log", ".png"]:
            companion = report.with_suffix(suffix)
            if companion.exists():
                if companion.is_symlink() or not companion.is_file() or companion.stat().st_size > 32 * 1024**2:
                    raise ValueError("Evaluation companions must be ordinary files of at most 32 MiB")
                if suffix in {".npz", ".png"} and record.get("artifacts", {}).get(companion.name) != sha256(companion):
                    raise ValueError("Saved evaluation artifact hash mismatch: " + companion.name)
                result.append((companion, "reports/" + stem + suffix))
    return result



def temporal_evidence_sources(source, evaluation_report, baseline_report, sample_manifest,
                              declaration_path=None, comparison_path=None):
    """Bind optional target-change analysis to the already archived paired arrays."""
    if declaration_path is None and comparison_path is None:
        return []
    if any(path is None for path in [evaluation_report, baseline_report, sample_manifest,
                                    declaration_path, comparison_path]):
        raise ValueError("Temporal archival requires both reports and complete paired pilot evidence")
    declaration = read_small_json(declaration_path, "Temporal declaration")
    comparison = read_small_json(comparison_path, "Temporal comparison")
    if (comparison.get("passed") is not True or not comparison.get("candidate_checkpoint")
            or resolve_path(comparison["candidate_checkpoint"]) != source):
        raise ValueError("Temporal comparison must pass and identify the archived checkpoint")
    baseline = read_small_json(baseline_report, "Baseline report")
    shared = {"baseline_report_sha256": sha256(baseline_report),
              "baseline_npz_sha256": sha256(baseline_report.with_suffix(".npz")),
              "sample_manifest_sha256": sha256(sample_manifest)}
    for key, expected in shared.items():
        if declaration.get(key) != expected or comparison.get(key) != expected:
            raise ValueError("Temporal evidence input hash mismatch: " + key)
    final = {"candidate_report_sha256": sha256(evaluation_report),
             "candidate_npz_sha256": sha256(evaluation_report.with_suffix(".npz")),
             "declaration_sha256": sha256(declaration_path)}
    if any(comparison.get(key) != expected for key, expected in final.items()):
        raise ValueError("Temporal comparison must bind exact candidate and declaration hashes")
    if (declaration.get("processor_semantics_sha256") != baseline.get("processor_semantics_sha256")
            or any(declaration.get(key) != comparison.get(key) for key in ["method", "threshold"])):
        raise ValueError("Temporal comparison method or processor semantics differ from declaration")
    helper = "scripts/compare_pilot_temporal_change.py"
    if comparison.get("script_sha256", {}).get(helper) != sha256(ROOT / helper):
        raise ValueError("Temporal comparison helper changed after analysis")
    return [(declaration_path, "reports/temporal-declaration.json"),
            (comparison_path, "reports/temporal-comparison.json"), (ROOT / helper, helper)]


def followup_evidence_sources(source, evaluation_report=None, run_config=None, comparison_path=None,
                             protocol_path=None, previous_report=None):
    """Preserve the frozen warm-start protocol and checkpoint-512 replay inputs."""
    if protocol_path is None and previous_report is None:
        if run_config is not None and read_small_json(run_config, "Run config").get("evaluation_protocol_sha256"):
            raise ValueError("Follow-up archival requires evaluation protocol and previous report")
        return []
    if any(path is None for path in [evaluation_report, run_config, comparison_path,
                                    protocol_path, previous_report]):
        raise ValueError("Follow-up archival requires protocol, previous report, and complete pilot/temporal evidence")
    read_small_json(protocol_path, "Evaluation protocol")
    from evaluate_gr00t_pilot import load_evaluation_protocol
    protocol = load_evaluation_protocol(protocol_path)
    recipe = read_small_json(run_config, "Run config")
    evaluation = read_small_json(evaluation_report, "Evaluation report")
    comparison = read_small_json(comparison_path, "Temporal comparison")
    previous = read_small_json(previous_report, "Previous evaluation report")
    if (source.name != "checkpoint-1024" or source.parent.name != protocol.get("experiment")
            or recipe.get("run") != source.parent.name or recipe.get("max_steps") != 1024
            or recipe.get("state_dropout_prob") != 0.0
            or recipe.get("fresh_optimizer_and_scheduler") is not True):
        raise ValueError("Follow-up run must match the declared 1024-update warm start")
    for label, record, key in [("Run config", recipe, "checkpoint"),
                               ("Evaluation", evaluation, "candidate_checkpoint"),
                               ("Comparison", comparison, "candidate_checkpoint")]:
        if not record.get(key) or resolve_path(record[key]) != source:
            raise ValueError(label + " does not identify the archived follow-up checkpoint")
    if (evaluation.get("passed") is not True or comparison.get("passed") is not True
            or previous.get("passed") is not True or previous.get("checkpoint_global_step") != 512):
        raise ValueError("Follow-up evidence must pass integrity checks and reference checkpoint 512")
    protocol_sha = sha256(protocol_path)
    if any(record.get("evaluation_protocol_sha256") != protocol_sha
           for record in [recipe, evaluation, comparison]):
        raise ValueError("Follow-up evaluation protocol hash mismatch")
    arrays = previous_report.with_suffix(".npz")
    if arrays.is_symlink() or not arrays.is_file() or arrays.stat().st_size > 32 * 1024**2:
        raise ValueError("Previous predictions must be an ordinary NPZ file of at most 32 MiB")
    previous_sha, arrays_sha = sha256(previous_report), sha256(arrays)
    bindings = protocol.get("bindings", {})
    for name, expected in [("previous_report", previous_sha), ("previous_arrays", arrays_sha)]:
        if bindings.get(name, {}).get("sha256") != expected:
            raise ValueError("Follow-up protocol binding mismatch: " + name)
    if (comparison.get("previous_report_sha256") != previous_sha
            or comparison.get("previous_npz_sha256") != arrays_sha
            or previous.get("artifacts", {}).get(arrays.name) != arrays_sha):
        raise ValueError("Previous prediction evidence hash mismatch")
    for key, binding in [("baseline_report_sha256", "baseline_report"),
                         ("sample_manifest_sha256", "sample_manifest")]:
        if any(record.get(key) != bindings.get(binding, {}).get("sha256")
               for record in [previous, evaluation, comparison]):
            raise ValueError("Follow-up reference binding mismatch: " + binding)
    if comparison.get("candidate_report_sha256") != sha256(evaluation_report):
        raise ValueError("Follow-up comparison identifies different candidate evidence")
    # These two exact upstream paths are allowlisted and hash-checked by the protocol loader.
    implementation = [(ROOT / relative, relative)
                      for relative in sorted(protocol["training_only_implementation_bindings"])]
    for path, relative in implementation:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024**2:
            raise ValueError("Protocol implementation source must be an ordinary file of at most 4 MiB")
        if sha256(path) != protocol["training_only_implementation_bindings"][relative]:
            raise ValueError("Protocol implementation source changed: " + relative)
    # A failed fitting gate remains a valid result: acceptance_passed is not an archival gate.
    return [(protocol_path, "reports/evaluation-protocol.json"),
            (previous_report, "reports/previous-evaluation.json"),
            (arrays, "reports/previous-evaluation.npz"), *implementation]


def repository_provenance():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "tracked_changes_at_archive": git("diff", "HEAD", "--name-only").splitlines(),
            "file_hashes_record_exact_archived_content": True}


def local_archive(stage, final, sources, signatures, manifest_path, manifest):
    """Copy once, verify every destination hash, fsync, then publish a fresh directory."""
    if final.exists() or final.is_symlink():
        raise FileExistsError(final)
    stage.mkdir(parents=True)
    verified = []
    for (source, relative), record in zip(sources, manifest["files"]):
        if signature(source) != signatures[source]:
            raise RuntimeError(f"Source changed before transfer: {source}")
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=8 * 1024**2)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if signature(source) != signatures[source] or sha256(target) != record["sha256"]:
            raise RuntimeError(f"Source/destination verification failed: {relative}")
        verified.append({k: record[k] for k in ["path", "size", "sha256"]})
        print(json.dumps({"phase": "verified_local", "path": relative}), flush=True)
    with (stage / "archive_manifest.json").open("xb") as stream:
        stream.write(manifest_path.read_bytes()); stream.flush(); os.fsync(stream.fileno())
    report = {"verified_at_utc": now(), "passed": True, "mode": "local",
              "local_checkpoint_source": str(manifest["checkpoint_source"]),
              "destination_directory": str(final),
              "destination_checkpoint": str(final / manifest["checkpoint_directory"]),
              "file_count": len(verified), "total_bytes": manifest["total_bytes"],
              "checkpoint_bytes": manifest["checkpoint_bytes"], "checkpoint_count": 1,
              "manifest_sha256": sha256(manifest_path), "all_source_destination_sha256_match": True,
              "per_file_fsync_completed": True, "source_preserved": True, "files": verified}
    with (stage / "archive_verification.json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    if final.exists() or final.is_symlink():
        raise FileExistsError(final)
    stage.rename(final)
    return report


class ProgressWriter:
    def __init__(self, pipe):
        self.pipe = pipe
        self.bytes = 0
        self.start = self.last = time.monotonic()

    def write(self, data):
        result = self.pipe.write(data)
        self.bytes += result
        current = time.monotonic()
        if current - self.last >= 30:
            print(json.dumps({"phase": "transfer", "compressed_bytes_sent": self.bytes,
                              "elapsed_seconds": round(current-self.start, 1)}), flush=True)
            self.last = current
        return result

    def flush(self):
        self.pipe.flush()


REMOTE_VERIFY = r"""from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,sys
stage,final=map(Path,sys.argv[1:3])
expected_manifest_hash=sys.argv[3]
manifest_path=stage/'archive_manifest.json'
manifest_bytes=manifest_path.read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest()==expected_manifest_hash, 'Manifest mismatch'
manifest=json.loads(manifest_bytes)
expected={record['path'] for record in manifest['files']}|{'archive_manifest.json'}
actual={str(path.relative_to(stage)) for path in stage.rglob('*') if path.is_file()}
assert actual==expected, 'Remote inventory mismatch'
verified=[]
for record in manifest['files']:
 path=stage/record['path']
 assert not path.is_symlink() and path.is_file(), 'Unexpected remote file'
 assert path.stat().st_size==record['size'], 'Remote size mismatch'
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
  os.fsync(stream.fileno())
 assert h.hexdigest()==record['sha256'], 'Remote content mismatch: '+record['path']
 verified.append({'path':record['path'],'size':record['size'],'sha256':h.hexdigest()})
report={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
'local_checkpoint_source':manifest['checkpoint_source'],'remote_host':manifest['destination_host'],
'remote_directory':str(final),'remote_checkpoint':str(final/manifest['checkpoint_directory']),
'file_count':len(verified),'total_bytes':manifest['total_bytes'],
'checkpoint_bytes':manifest['checkpoint_bytes'],'checkpoint_count':1,
'manifest_sha256':expected_manifest_hash,'all_source_remote_sha256_match':True,
'per_file_fsync_completed':True,'source_preserved':True,
'compressed_transfer_bytes':int(sys.argv[4]),'files':verified,
'reload_dependency':manifest['reload_dependency']}
report_path=stage/'archive_verification.json'
with report_path.open('x') as stream:
 json.dump(report,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
if final.exists(): raise FileExistsError(str(final))
os.rename(stage,final)
print(json.dumps(report))
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path, required=True, help="Reload-validated checkpoint directory")
    parser.add_argument("--destination", required=True, help="Archive parent directory; a new run child is created")
    parser.add_argument("--run", required=True)
    parser.add_argument("--reload-report", type=Path, required=True)
    parser.add_argument("--train-report", type=Path)
    parser.add_argument("--evaluation-report", type=Path, help="Successful paired pilot evaluation JSON")
    parser.add_argument("--run-config", type=Path, help="Exact pilot run configuration JSON")
    parser.add_argument("--pilot-manifest", type=Path, help="Frozen snapshot manifest JSON; no raw files are copied")
    parser.add_argument("--baseline-report", type=Path, help="Successful paired baseline evaluation JSON and NPZ")
    parser.add_argument("--sample-manifest", type=Path, help="Frozen paired evaluation sample manifest JSON")
    parser.add_argument("--training-audit", type=Path, help="Optional bounded per-step finite audit JSONL for pilot evidence")
    parser.add_argument("--temporal-declaration", type=Path, help="Frozen ground-truth-only target-change declaration JSON")
    parser.add_argument("--temporal-comparison", type=Path, help="Completed target-change comparison JSON")
    parser.add_argument("--evaluation-protocol", type=Path, help="Frozen protocol for the zero-state-dropout warm start")
    parser.add_argument("--previous-report", type=Path, help="Paired checkpoint-512 evaluation JSON and NPZ; requires protocol")
    parser.add_argument("--mode", choices=["local", "ssh"], default="local")
    parser.add_argument("--ssh-host", help="SSH host/alias; defaults to configured storage.host")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        raise ValueError("Run must be a simple directory name")
    config = load_config(args.config)
    source = resolve_path(args.source)
    evidence_path = resolve_path(args.reload_report)
    evidence = json.loads(evidence_path.read_text())
    sources = validate_source(source, evidence)
    pilot_sources = pilot_evidence_sources(source,
        *(resolve_path(value) if value is not None else None for value in
          [args.evaluation_report, args.run_config, args.pilot_manifest, args.baseline_report, args.sample_manifest]))
    sources.extend(pilot_sources)
    temporal_sources = temporal_evidence_sources(source,
        *(resolve_path(value) if value is not None else None for value in
          [args.evaluation_report, args.baseline_report, args.sample_manifest,
           args.temporal_declaration, args.temporal_comparison]))
    sources.extend(temporal_sources)
    followup_sources = followup_evidence_sources(source,
        *(resolve_path(value) if value is not None else None for value in
          [args.evaluation_report, args.run_config, args.temporal_comparison,
           args.evaluation_protocol, args.previous_report]))
    if followup_sources and (not pilot_sources or not temporal_sources):
        raise ValueError("Follow-up archival requires complete pilot and temporal evidence")
    sources.extend(followup_sources)
    if args.training_audit is not None:
        audit = resolve_path(args.training_audit)
        if (not pilot_sources or audit.suffix != ".jsonl" or audit.is_symlink()
                or not audit.is_file() or audit.stat().st_size > 32 * 1024**2):
            raise ValueError("Training audit requires pilot evidence and an ordinary JSONL file of at most 32 MiB")
        sources.append((audit, "reports/training-audit.jsonl"))
    report_path = guard_path(config["paths"]["reports"] / (args.run + "-archive.json"), config, "reports")
    manifest_path = guard_path(config["paths"]["reports"] / (args.run + "-archive-manifest.json"), config, "reports")
    if any(path.exists() or path.is_symlink() for path in [report_path, manifest_path]):
        raise FileExistsError("Archive reports already exist; choose a new run")
    host = None
    if args.mode == "local":
        parent = guard_path(args.destination, config, "archive")
        final = guard_path(parent / args.run, config, "archive")
        if final == source or final.is_relative_to(source) or source.is_relative_to(final):
            raise ValueError("Archive destination must not overlap its source")
        if final.exists() or final.is_symlink():
            raise FileExistsError(final)
        stage = parent / ("." + args.run + ".incomplete-" + uuid.uuid4().hex)
    else:
        host = args.ssh_host or config["storage"]["host"]
        if not host or host.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]+", host):
            raise ValueError("A valid explicit/configured SSH host is required")
        parent = PurePosixPath(args.destination)
        if not parent.is_absolute() or str(parent) == "/" or ".." in parent.parts:
            raise ValueError("Remote destination must be a dedicated absolute archive directory")
        configured_remote = config["storage"]["remote_path"]
        if configured_remote:
            allowed = PurePosixPath(configured_remote) / "checkpoints"
            if not parent.is_relative_to(allowed):
                raise ValueError("Remote destination must be below configured remote_path/checkpoints")
        raw = config["paths"]["raw_data"]
        if parent == PurePosixPath(raw) or parent.is_relative_to(PurePosixPath(raw)):
            raise ValueError("Original recordings are read-only")
        final = parent / args.run
        stage = parent / ("." + args.run + ".incomplete-" + uuid.uuid4().hex)
    sources.append((evidence_path, "reports/reload.json"))
    for suffix in [".npz", ".log"]:
        candidate = evidence_path.with_suffix(suffix)
        if candidate.is_file():
            sources.append((candidate, "reports/reload" + suffix))
    if args.train_report:
        training = resolve_path(args.train_report)
        sources.append((training, "reports/train.json"))
        if training.with_suffix(".log").is_file():
            sources.append((training.with_suffix(".log"), "reports/train.log"))
    # Explicit non-secret evidence allowlist; no environment files or arbitrary reports.
    for relative in ["README.md", "docs/execution-plan.md", "configs/g1_inspire_config.py",
                     "configs/modality.json", "scripts/run_gr00t_smoke.py",
                     "scripts/reload_gr00t_checkpoint.py", "scripts/project_config.py",
                     "scripts/archive_smoke_checkpoint.py"]:
        if (ROOT / relative).is_file():
            sources.append((ROOT / relative, relative))
    if pilot_sources:
        for relative in ["scripts/run_gr00t_pilot.py", "scripts/evaluate_gr00t_pilot.py",
                         "scripts/convert_single_camera.py", "scripts/prepare_pilot.py"]:
            if not (ROOT / relative).is_file():
                raise FileNotFoundError("Missing pilot source provenance: " + relative)
            sources.append((ROOT / relative, relative))
    for path in sorted((ROOT / "locks").iterdir()):
        if path.is_file() and path.suffix in {".json", ".txt"}:
            sources.append((path, "locks/" + path.name))
    records, signatures = [], {}
    for path, relative in sources:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected regular source file: {path}")
        before = signature(path)
        digest = sha256(path)
        if signature(path) != before:
            raise RuntimeError(f"Source changed during hashing: {path}")
        signatures[path] = before
        records.append({"path": relative, "source": str(path), "size": before[0], "sha256": digest})
        print(json.dumps({"phase": "hashed", "path": relative, "bytes": before[0]}), flush=True)
    manifest = {"created_at_utc": now(), "checkpoint_source": str(source),
                "repository": repository_provenance(), "pilot_evidence_included": bool(pilot_sources),
                "temporal_evidence_included": bool(temporal_sources),
                "checkpoint_directory": source.name, "destination": str(final), "destination_host": host,
                "mode": args.mode, "checkpoint_count": 1, "source_checkpoint_preserved": True,
                "files": records, "total_bytes": sum(record["size"] for record in records),
                "checkpoint_bytes": sum(record["size"] for record in records if record["path"].startswith(source.name + "/")),
                "reload_dependency": "Pinned Cosmos cache and project runtime from included locks; no environments or credentials archived."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2); stream.write("\n")
    if args.mode == "local":
        report = local_archive(stage, final, sources, signatures, manifest_path, manifest)
    else:
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", host]
        remote_python(ssh, "from pathlib import Path; import sys; stage,final=map(Path,sys.argv[1:]); "
                      "assert not final.exists() and not final.is_symlink(), 'Destination exists'; "
                      "stage.parent.mkdir(parents=True,exist_ok=True); stage.mkdir()", stage, final)
        command = shlex.join(["tar", "--no-same-owner", "--no-same-permissions", "-xzf", "-", "-C", str(stage)])
        process = subprocess.Popen(ssh + [command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        writer = ProgressWriter(process.stdin)
        try:
            with tarfile.open(fileobj=writer, mode="w|gz", compresslevel=1, bufsize=1024**2) as archive:
                for path, relative in sources:
                    if signature(path) != signatures[path]:
                        raise RuntimeError(f"Source changed before transfer: {path}")
                    archive.add(path, arcname=relative, recursive=False)
                    if signature(path) != signatures[path]:
                        raise RuntimeError(f"Source changed during transfer: {path}")
                archive.add(manifest_path, arcname="archive_manifest.json", recursive=False)
            process.stdin.close(); process.stdin = None
            _, stderr = process.communicate(timeout=300)
            if process.returncode:
                raise RuntimeError(f"Remote archive extraction failed: {stderr.decode(errors='replace')}")
        except BaseException:
            process.kill(); process.wait(); raise
        result = remote_python(ssh, REMOTE_VERIFY, stage, final, sha256(manifest_path), writer.bytes)
        report = json.loads(result.stdout)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps({"phase": "complete", "report": str(report_path),
                      "destination": str(final), "files_verified": report["file_count"]}), flush=True)


if __name__ == "__main__":
    main()
