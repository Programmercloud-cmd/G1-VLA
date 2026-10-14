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
