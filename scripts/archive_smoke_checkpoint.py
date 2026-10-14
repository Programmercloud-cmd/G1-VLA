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
                     "scripts/reload_gr00t_checkpoint.py", "scripts/project_config.py"]:
        if (ROOT / relative).is_file():
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
