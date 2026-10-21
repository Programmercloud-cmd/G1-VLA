# Larger-dataset v1 resume — 21 October 2026

Current outcome: [retry2](larger-dataset-v1-retry2.md) stopped at 5,231 of
8,192 updates when the unchanged 8 GiB local disk reserve guard fired. Its cap
was not reached. Checkpoints 2,048 and 4,096 have certified save events; their
paired train/validation evaluations are underway. The [partial outcome](../experiments/larger-dataset-v1/retry2-outcome.json)
preserves the finite prefix and final evidence. There is no automatic training
retry/resume or formal four-checkpoint selection. The [retry2 post-training record](post-training-larger-dataset-v1-retry2.md)
is the current evaluation/archive reference.

The launch and recovery commands below are historical execution records, not
instructions to start another run. The first startup stopped before any update
because upstream set iteration changed the JSON ordering of two identical
embodiment IDs. The [deterministic retry](larger-dataset-v1-retry1.md) used
`PYTHONHASHSEED=1`; retry1 later stopped at 1,364 completed updates after a storage
interruption, without a checkpoint. The user then explicitly approved retry2's
fresh six-hour cap. All attempts and their evidence remain preserved.

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
The readiness freeze preceded model inference. Both pretrained step-zero
baselines completed before training; the partial training outcome is recorded above.

Storage maintenance also completed, reclaiming 405,536,768 allocated bytes while
preserving source, environment, model, dataset and checkpoint bytes. Its report
is `reports/larger-dataset-v1-disk-maintenance-final-2026-10-21.json`. The additive
`reports/larger-dataset-v1-git-pack-full-verification-2026-10-21.json` records both
successful full `git verify-pack` checks without `-s`; the original statistics-only
record remains preserved. The compact
[resume validation record](../experiments/larger-dataset-v1/resume-validation-20261021.json)
binds these reports, the preserved helper/tests, machine configuration and all
21 unchanged frozen source hashes.

The first paired baseline completed the train split, then restarted because
OpenCV had changed `LD_LIBRARY_PATH` before the second runtime check. The original
no-overwrite guard stopped that restart, preserving the completed train evidence.
Validation then completed in a separate process using the unchanged evaluator,
model, processor, samples and seeds. Both reports passed CPU integrity checks;
train/validation model, processor, statistics and weight identities match. Their
group-balanced normalized MAEs are 0.759350 and 0.765781 respectively.
[The additive runtime incident record](../experiments/larger-dataset-v1/evaluation-runtime-incident-20261021.json)
binds the original failure log, recovered baseline artifacts and new launcher tests.

Use [scripts/launch_gr00t_evaluation.py](../scripts/launch_gr00t_evaluation.py) for
forthcoming paired checkpoint evaluations. It preloads OpenCV, restores the two
native startup path variables, and forwards arguments to the unchanged evaluator.
The original runtime validator and marker remain active. Four CPU fixtures and
an actual pinned-runtime OpenCV preflight passed; the latter makes any attempted
restart fail the test after startup. Full paired model inference through this
launcher also passed with fresh, nonauthoritative `step-zero-launcher-check`
reports: both prediction arrays are bitwise identical to their original baselines
(maximum absolute difference zero), and validation reused the one model in the
same process. The original authoritative baseline references remain unchanged.
Fine-tuned checkpoint evaluations are now underway for the two saved checkpoints.

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

Post-training reports use durable storage at
`storage/reports/larger-dataset-v1`. The machine-local configuration sets
`paths.reports` to that directory and explicitly retains
`paths.runtime_record="reports/machine-runtime.json"`. This changes only locations
for new execution evidence. Existing `reports/larger-dataset-v1-conversion.json`
and every frozen protocol input remain at their original paths. Post-training shell log
redirections also target that directory. Retry2 training instead used the separate
`reports/larger-dataset-v1-retry2-machine.json` configuration and local audit files;
those completed files are now durably copied with SHA verification. Do not write beneath
`storage` unless its configured SSHFS mount is present.

The ignored `reports/larger-dataset-v1-machine.json` was updated after saving
`reports/larger-dataset-v1-machine-before-storage-reports-2026-10-21.json`.
This session does not relocate model assets, existing conversion evidence or
frozen evaluation files.

Historical original-launch dry run, preserved below for provenance. It is not the
current retry2 command and must not be rerun into existing outputs. The existing
readiness and dry-run reports remain preserved:

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

The original launch command above was superseded first by the deterministic
retry1 and then by retry2. It is not authorization for another training run.
Retry2's actual run manifest records commit `2d681e4`, fixed hash seed 1, local
training audits and unchanged recipe/source bindings. The final report records
`aborted=resource_reserve` and worker exit -15; the independent cap watchdog ended
without sending signals. Final evidence is durably preserved under
`storage/reports/larger-dataset-v1-retry2-completion`.

Follow the [retry2 post-training record](post-training-larger-dataset-v1-retry2.md)
for the existing inventories and ongoing evaluations of checkpoints 2,048 and
4,096 only. The original step-zero baselines remain authoritative. Evaluation
uses a checked RAM-backed temporary directory to keep incidental temporary
allocation off the narrowly provisioned project filesystem. No GPU evaluation
may overlap training; training has exited and its GPU lease has been released.
The protocol's four-pair selection is unavailable for this incomplete run.
Offline action errors do not establish robot pickup success.
