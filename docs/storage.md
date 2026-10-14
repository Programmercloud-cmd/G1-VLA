# Storage and checkpoint archival

Configure `paths.runs` for active training output and `paths.archive` for durable artifacts in `machine.local.json`. Local directories and mounted filesystems are supported. Keep Python environments on a local executable Linux filesystem. SSH storage is optional; no particular host alias or external-drive path is required.

The default run directory is Linux shared memory. Its contents disappear after reboot. The two-step smoke requires 35 GiB free in the output filesystem and 60 GiB available host RAM; project disk retains an 8 GiB reserve. Upstream writes both the complete step checkpoint and duplicate final weights at the run root, so budget both transient copies. Archive only the complete step checkpoint, including its processor, statistics, config, and model index files.

Before the authorized `pilot-fit-no-state-dropout-v1` follow-up, all 32 files in the original smoke and 512-step checkpoints and all six duplicate root weight shards were checksum-verified. Those six root shard paths now hard-link to their respective saved checkpoint shards within `/dev/shm`; both names and exact contents are preserved, and durable archives are unchanged. The local record is `reports/pilot-fit-no-state-dropout-storage-dedup.json`. Afterward, output space was 39.45 GiB free, available RAM was 99.1 GiB, and project space was 8.88 GiB free. These are host measurements at preparation time; recheck capacity before launch. Treat both names of every linked shard as immutable: an in-place write through either name changes the shared file. Use a fresh run directory for new weights and archive only the new complete checkpoint. No mount or machine-configuration change was required.

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

Pilot archival additionally binds the run configuration to the exact checkpoint, checks the frozen snapshot/sample/baseline hashes, and preserves both baseline and candidate predictions. It includes the finite-value audit, trajectory plots, executed runner/evaluator source files, and the separately predeclared target-change analysis. The latter is bound to the exact declaration, baseline/candidate reports and arrays, sample manifest, and comparison helper source. No baseline weights or duplicate root model are copied. `passed` in evaluation means evidence is complete and finite; an unimproved result can still be preserved.

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
  --training-audit reports/pilot-fit-v1-finite.jsonl \
  --temporal-declaration reports/pilot-fit-v1-temporal-declaration.json \
  --temporal-comparison reports/pilot-fit-v1-temporal-comparison.json
```

For compressed SSH transfer, use the same evidence arguments with `--mode ssh`, your configured host, and an absolute remote checkpoint parent as described above. Pilot JSON metadata is limited to 4 MiB per file; prediction, plot, log, and audit companions are bounded to 32 MiB each. Sources remain preserved after successful verification.

## Replay the archived pilot comparison

The completed pilot's portable checkpoint reference is `archive:checkpoints/pilot-fit-v1/checkpoint-512`; [its manifest](../experiments/pilot-fit-v1/artifacts.json) records exact hashes. Archived source and documentation files are historical snapshots from before final publication; current results are in the repository's experiment summary.

The archive uses generic report filenames, while JSON artifact keys retain their original basenames. Restore the original names without rewriting JSON or copying weights. The example uses the default archive/reports directories; adjust them to your machine configuration and choose a fresh replay directory. Verify copied-file hashes against `archive_manifest.json` first.

```bash
mkdir -p reports/pilot-fit-v1-replay
cp storage/checkpoints/pilot-fit-v1/reports/baseline.json \
  reports/pilot-fit-v1-replay/pilot-fit-v1-baseline.json
cp storage/checkpoints/pilot-fit-v1/reports/baseline.npz \
  reports/pilot-fit-v1-replay/pilot-fit-v1-baseline.npz
cp storage/checkpoints/pilot-fit-v1/reports/evaluation.json \
  reports/pilot-fit-v1-replay/pilot-fit-v1-evaluation.json
cp storage/checkpoints/pilot-fit-v1/reports/evaluation.npz \
  reports/pilot-fit-v1-replay/pilot-fit-v1-evaluation.npz
cp storage/checkpoints/pilot-fit-v1/reports/temporal-declaration.json \
  reports/pilot-fit-v1-replay/pilot-fit-v1-temporal-declaration.json

python3 -B scripts/compare_pilot_temporal_change.py \
  --baseline reports/pilot-fit-v1-replay/pilot-fit-v1-baseline.json \
  --declaration reports/pilot-fit-v1-replay/pilot-fit-v1-temporal-declaration.json \
  --candidate reports/pilot-fit-v1-replay/pilot-fit-v1-evaluation.json \
  --report reports/pilot-fit-v1-replay/replayed-comparison.json
```

Compare the replayed JSON with archived `reports/temporal-comparison.json`: `groups`, `method`, `threshold`, and all baseline/candidate report/array, declaration, and sample-manifest hashes must match exactly. Timestamps and repository HEAD may differ. This replay uses saved arrays on CPU and needs no checkpoint weights or new inference. The completed pilot archive was replayed with these checks successfully.
