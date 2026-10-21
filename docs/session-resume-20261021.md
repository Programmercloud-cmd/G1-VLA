# Larger-dataset v1 resume — 21 October 2026

The user resumed the already approved 8,192-update experiment. The frozen split,
protocol, processor, source bindings and historical reports remain unchanged.
This note records the new session's execution locations and recovery work; the
[previous handoff](session-handoff-larger-dataset-v1.md) remains historical evidence.

Production restoration succeeded: all 48,204 files / 5,276,113,118 bytes passed
post-extraction SHA256 verification and were atomically published at the exact
`/dev/shm/vla-large-data-v1` paths. Six local project files were verified/reused,
and the earlier partial staging directory was preserved. The completed report
is `reports/larger-dataset-v1-restore-20261021.json`; its log is
`reports/larger-dataset-v1-restore-20261021.log`. The compressed archive matches
the original closeout SHA256. Actual-host GPU readiness also passed all 19 checks,
including the actual dataset loader, runtime/source locks, decoder, GPU capacity
and tiny CUDA kernels. Its report is
`storage/reports/larger-dataset-v1/larger-dataset-v1-next-session-readiness.json`.
Pretrained model baseline evaluation and training are still pending at this freeze;
readiness kernels do not constitute model baseline inference or training.

Storage maintenance also completed, reclaiming 405,536,768 allocated bytes while
preserving source, environment, model, dataset and checkpoint bytes. Its report
is `reports/larger-dataset-v1-disk-maintenance-final-2026-10-21.json`. The additive
`reports/larger-dataset-v1-git-pack-full-verification-2026-10-21.json` records both
successful full `git verify-pack` checks without `-s`; the original statistics-only
record remains preserved. The compact
[resume validation record](../experiments/larger-dataset-v1/resume-validation-20261021.json)
binds these reports, the preserved helper/tests, machine configuration and all
21 unchanged frozen source hashes.

The actual restore ran the preserved implementation from
`reports/restore_larger_dataset_v1.py`. Its byte-for-byte tracked copy is
[scripts/restore_larger_dataset_v1.py](../scripts/restore_larger_dataset_v1.py).
It is specifically the restorer for this experiment's archive and closeout
format. It checks archive/manifest/verification bindings, rejects unsafe or
unlisted members, verifies all extracted hashes, and publishes into an absent
destination with Linux `renameat2(RENAME_NOREPLACE)`. Failed staging directories
are preserved. Run Python normally, without `-O`, because its checks use assertions.

The five focused fixtures cover valid extraction, incorrect member SHA256,
path traversal, an occupied destination and incorrect compressed SHA256. They
mock only free-space capacity. Run them from the checkout root:

```bash
python3 -B -m unittest discover -s tests -p test_dataset_restore.py -v
```

For a future restoration into an absent data destination, after confirming that
the configured storage mount is the expected external drive:

```bash
python3 -B scripts/restore_larger_dataset_v1.py \
  --archive storage/derived/larger-dataset-v1/dataset.tar.gz \
  --manifest storage/derived/larger-dataset-v1/archive_manifest.json \
  --verification storage/derived/larger-dataset-v1/archive_verification.json \
  --closeout experiments/larger-dataset-v1/session-closeout.json \
  --destination /dev/shm/vla-large-data-v1 \
  --project /home/test2/dev/VLA \
  --report reports/larger-dataset-v1-restore-20261021.json
```

This command refuses an occupied destination or report. Inspect existing results
before choosing another execution; the current restored destination is occupied.
The machine's root filesystem must retain the existing 8 GiB reserve.

New reports use durable storage at
`storage/reports/larger-dataset-v1`. The machine-local configuration sets
`paths.reports` to that directory and explicitly retains
`paths.runtime_record="reports/machine-runtime.json"`. This changes only locations
for new execution evidence. Existing `reports/larger-dataset-v1-conversion.json`
and every frozen protocol input remain at their original paths. Shell log
redirections must also target the new reports directory. Do not write beneath
`storage` unless its configured SSHFS mount is present.

The ignored `reports/larger-dataset-v1-machine.json` was updated after saving
`reports/larger-dataset-v1-machine-before-storage-reports-2026-10-21.json`.
This session does not relocate model assets, existing conversion evidence or
frozen evaluation files.

Restoration and readiness have passed. Use these next commands from
`/home/test2/dev/VLA` with the updated machine-local configuration; the existing
readiness report is preserved rather than overwritten:

```bash
VLA_CFG=reports/larger-dataset-v1-machine.json
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_DATA=/dev/shm/vla-large-data-v1
VLA_REPORTS=storage/reports/larger-dataset-v1
VLA_PROTOCOL=experiments/larger-dataset-v1/evaluation-protocol.json
VLA_TRAIN="$VLA_DATA/lerobot/local/g1_inspire_larger_v1_train"
VLA_ZERO_TRAIN="$VLA_REPORTS/larger-dataset-v1-step-zero-train.json"
VLA_ZERO_VALIDATION="$VLA_REPORTS/larger-dataset-v1-step-zero-validation.json"

python3 scripts/project_storage.py --config "$VLA_CFG" status

"$VLA_PY" -B scripts/evaluate_gr00t_dataset.py --config "$VLA_CFG" evaluate \
  --protocol "$VLA_PROTOCOL" --split both --step-zero \
  --train-report "$VLA_ZERO_TRAIN" --validation-report "$VLA_ZERO_VALIDATION"

"$VLA_PY" -B scripts/run_gr00t_dataset.py --config "$VLA_CFG" \
  --train-dataset "$VLA_TRAIN" \
  --train-manifest "$VLA_DATA/snapshots/train/manifest.json" \
  --validation-manifest "$VLA_DATA/snapshots/validation/manifest.json" \
  --split-manifest experiments/larger-dataset-v1/split.json \
  --evaluation-protocol "$VLA_PROTOCOL" \
  --train-baseline-report "$VLA_ZERO_TRAIN" \
  --validation-baseline-report "$VLA_ZERO_VALIDATION" \
  --run larger-dataset-v1 --timeout-seconds 21600 --dry-run
```

Inspect both successful baseline reports and the dry-run bindings, then repeat
the training command without `--dry-run`. The resumed authorization covers this
run. Keep the six-hour limit and existing resource reserves. The launcher
automatically places its run report, training log, finite audit and coverage
files under the configured external reports directory.

After training exits, follow the
[checkpoint inventory, paired evaluation, archive and selection workflow](larger-dataset-execution-v1.md),
using `"$VLA_REPORTS/..."` for every new report/inventory argument, both step-zero
reference arguments, `--run-config` and `--train-report`. Checkpoint weights stay
under `storage/runs/larger-dataset-v1`; archival publishes their evidence in
place. No GPU evaluation may overlap training. Selection requires all four valid
pairs at 2,048, 4,096, 6,144 and 8,192 updates. Preserve and report an incomplete
run honestly if the cap is reached. Offline action errors do not establish robot
pickup success.
