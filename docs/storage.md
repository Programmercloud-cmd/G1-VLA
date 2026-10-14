# Storage and checkpoint archival

Configure `paths.runs` for active training output and `paths.archive` for durable artifacts in `machine.local.json`. Local directories and mounted filesystems are supported. Keep Python environments on a local executable Linux filesystem. SSH storage is optional; no particular host alias or external-drive path is required.

The default run directory is Linux shared memory. Its contents disappear after reboot. The two-step smoke requires 35 GiB free in the output filesystem and 60 GiB available host RAM; project disk retains an 8 GiB reserve. Upstream writes both the complete step checkpoint and duplicate final weights at the run root, so budget both transient copies. Archive only the complete step checkpoint, including its processor, statistics, config, and model index files.

## Optional SSHFS

To use an SSH-backed mount, set `storage.enabled` to `true`, provide your SSH alias and absolute remote project path, and set `paths.archive` to the dedicated local mountpoint. Existing SSH authentication must work noninteractively. Install SSHFS through your operating system if it is unavailable.

```bash
python3 scripts/project_storage.py --config machine.local.json status
python3 scripts/project_storage.py --config machine.local.json mount
```

The helper checks the configured source and refuses to hide files in a nonempty mountpoint. The mount is non-executable and must be re-established after reboot. On the original machine, remote file writes, fsync, rename, symlinks, and checksum transfers passed. Throughput was about 6 MiB/s through SSHFS and 9 MiB/s with SCP; those measurements are environment-specific.

## Archive a verified checkpoint

Run the fresh-process reload first and retain its successful JSON report. The archive helper accepts explicit source, destination, run name, and reload report; run `python3 scripts/archive_smoke_checkpoint.py --help` for the supported local/mounted and optional compressed SSH modes. It refuses existing destinations, preserves the source, and verifies the complete remote/file inventory with SHA-256 before publishing the result. No credentials or environment files belong in an archive.

For the default paths, archive a newly reloaded smoke checkpoint to local or mounted storage:

```bash
python3 scripts/archive_smoke_checkpoint.py --source /dev/shm/vla-groot-smoke/smoke-v1/checkpoint-2 \
  --destination storage/checkpoints --run smoke-v1 \
  --reload-report reports/smoke-v1-reload.json --train-report reports/smoke-v1-train.json
```

Add `--mode ssh --ssh-host your-storage-host` and set `--destination` to the absolute remote checkpoint parent for a compressed transfer over SSH. Configure that host and remote project root in your private machine configuration. This mode streams directly rather than creating a second local model archive.

The historical smoke's durable artifact reference is:

```text
archive:checkpoints/pilot-smoke-v2/checkpoint-2
```

Resolve `archive:` against your configured archive root. The tracked [artifact manifest](../experiments/pilot-smoke-v2/artifacts.json) records each checkpoint file's size and digest. Its sibling `archive_verification.json` is the original detailed durability record. The archived README and other documents are historical snapshots; the verification record establishes when publication completed.

A model checkpoint is not a standalone Python installation. Keep the pinned source/dependency locks and Cosmos model assets available for reload. Do not delete the source until archival verification has passed and the durable location is accessible.

## Archive the 512-step pilot

Pilot archival additionally binds the run configuration to the exact checkpoint, checks the frozen snapshot/sample/baseline hashes, and preserves both baseline and candidate predictions. It includes the finite-value audit, trajectory plots, and executed runner/evaluator source files. No baseline weights or duplicate root model are copied. `passed` in evaluation means evidence is complete and finite; an unimproved result can still be preserved.

```bash
python3 scripts/archive_smoke_checkpoint.py \
  --source /dev/shm/vla-groot-smoke/pilot-fit-v1/checkpoint-512 \
  --destination storage/checkpoints --run pilot-fit-v1 \
  --reload-report reports/pilot-fit-v1-reload.json \
  --train-report reports/pilot-fit-v1-train.json \
  --run-config reports/pilot-fit-v1-run.json \
  --evaluation-report reports/pilot-fit-v1-evaluation.json \
  --baseline-report reports/pilot-fit-v1-baseline.json \
  --sample-manifest reports/pilot-fit-v1-samples.json \
  --pilot-manifest derived/pilot-v1/manifest.json \
  --training-audit reports/pilot-fit-v1-finite.jsonl
```

For compressed SSH transfer, use the same evidence arguments with `--mode ssh`, your configured host, and an absolute remote checkpoint parent as described above. Pilot JSON metadata is limited to 4 MiB per file; prediction, plot, log, and audit companions are bounded to 32 MiB each. Sources remain preserved after successful verification.
