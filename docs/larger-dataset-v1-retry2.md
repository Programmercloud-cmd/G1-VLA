# Larger-dataset v1 storage recovery and retry2

**Actual outcome:** retry2 stopped at 5,231 of 8,192 updates when the unchanged
8 GiB local disk reserve guard fired, before its six-hour cap. Checkpoints 2,048
and 4,096 are certified and preserved; their evaluations are underway. The
[partial outcome](../experiments/larger-dataset-v1/retry2-outcome.json) records
finite-prefix evidence. No automatic training retry/resume or four-pair selection
is authorized. The launch commands below are historical records; use the
[post-training record](post-training-larger-dataset-v1-retry2.md) for current scope.

`larger-dataset-v1-retry1` completed 1,364 optimizer updates before its SSHFS
connection failed. The preserved audit contains 10,915 finite micro-losses and
1,364 finite gradient/update records, ending three microbatches into the next
update. It saved no checkpoint. Its existing output paths and a separately
verified interruption snapshot remain preserved.

Direct SSH recovered after the user restored the PC/tunnel, but the original
SSHFS daemon had lost its transport and its open training file descriptors did
not recover. The pinned SSHFS 3.7.1 documentation says reconnection requires
reopening previously open files. The original worker was stopped, the broken
daemon was terminated to release its pending kernel I/O, and the same storage
source was normally unmounted and remounted. The local cap watchdog ended
without sending any cap signals. No source, dataset, processor, protocol,
environment or checkpoint bytes were changed by this recovery.

The user explicitly selected **“Fresh six-hour cap for the retry.”** This permits
`larger-dataset-v1-retry2` its own 21,600-second cap from the fresh launch and
supersedes retry1's original 22:37 UTC deadline for retry2 only. The answer was
observed by 2026-10-21 17:28:28 UTC; no exact platform approval timestamp was
provided. The [recovery record](../experiments/larger-dataset-v1/storage-interruption-recovery.json)
and [retry2 execution declaration](../experiments/larger-dataset-v1/retry2-execution.json)
bind this scope and the completed CPU checks.

Training reports now use the local project volume, so writing each finite audit
line does not depend on the storage connection. Checkpoints remain on the
configured external volume. Keep the PC, external drive and SSH tunnel available
through training, saves and evaluation. A noninteractive attempt to acquire a
sleep inhibitor on that PC was denied; no power settings were changed.

Historical configuration and launch dry-run record; existing outputs are occupied,
and these commands are not authorization to launch again:

```bash
VLA_TRAIN_CFG=reports/larger-dataset-v1-retry2-machine.json
VLA_EVAL_CFG=reports/larger-dataset-v1-machine.json
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_DATA=/dev/shm/vla-large-data-v1
VLA_PROTOCOL=experiments/larger-dataset-v1/evaluation-protocol.json
VLA_RUN=larger-dataset-v1-retry2
VLA_TRAIN_REPORTS=reports/larger-dataset-v1-retry2
VLA_EVAL_REPORTS=storage/reports/larger-dataset-v1
VLA_ZERO_TRAIN="$VLA_EVAL_REPORTS/larger-dataset-v1-step-zero-train.json"
VLA_ZERO_VALIDATION="$VLA_EVAL_REPORTS/larger-dataset-v1-step-zero-validation.json"

PYTHONHASHSEED=1 "$VLA_PY" -B scripts/run_gr00t_dataset.py --config "$VLA_TRAIN_CFG" \
  --train-dataset "$VLA_DATA/lerobot/local/g1_inspire_larger_v1_train" \
  --train-manifest "$VLA_DATA/snapshots/train/manifest.json" \
  --validation-manifest "$VLA_DATA/snapshots/validation/manifest.json" \
  --split-manifest experiments/larger-dataset-v1/split.json \
  --evaluation-protocol "$VLA_PROTOCOL" \
  --train-baseline-report "$VLA_ZERO_TRAIN" \
  --validation-baseline-report "$VLA_ZERO_VALIDATION" \
  --run "$VLA_RUN" --timeout-seconds 21600 --dry-run
```

The CPU dry run and twelve structure/environment/source/model checks passed,
including full model-asset hashes. Both earlier real CPU processor proofs remain
applicable: `PYTHONHASHSEED=1` reproduces all three frozen processor files, while
the training/data seed remains 42. All 21 frozen source hashes remain exact.

The unchanged pipeline was already published at `2d681e4` when retry2 launched.
The actual command omitted `--dry-run`, used `PYTHONHASHSEED=1` with normal GPU
visibility, and captured its launcher log locally. The execution declaration
records the selected environment and exact process identities. An independent
local watchdog used process-start plus 21,600 seconds; it ended without signals
when training stopped early. The first 32 actual consumed coordinates matched
retry1's preserved prefix, checking sampling identities rather than whole-model
numerical equivalence. This is a completed execution record, not an instruction
to repeat the command.

The executed training requirements were unchanged: 8,192 updates, microbatch one,
accumulation eight, pretrained initialization, fresh optimizer, zero state
dropout, four model-only checkpoints at 2,048/4,096/6,144/8,192, the mandatory
final root save, and existing finite/source/resource guards. Do not extend or
automatically resume if the cap or a required guard stops training.

Training has exited. Current post-training work uses `VLA_EVAL_CFG` for the
existing checkpoint inventories, the tested paired evaluation launcher and
in-place archival. Put each inventory and
train/validation evaluation report under `VLA_EVAL_REPORTS`. Pass
`--run-config "$VLA_TRAIN_REPORTS/$VLA_RUN-run.json"` and
`--train-report "$VLA_TRAIN_REPORTS/$VLA_RUN-train.json"` to archival; the run
manifest identifies the local finite and coverage audits. Keep the original
baseline paths above. Durably copy and SHA-verify completed local training
reports after they stop changing; do not relocate the active audit.
The [retry2 post-training note](post-training-larger-dataset-v1-retry2.md) gives
the exact checkpoint, evaluation and publication commands for these two configurations.

No GPU evaluation may overlap training; its lease was released after process exit.
The unchanged protocol requires all four valid checkpoint pairs for formal
selection. This run has only two saved checkpoints and will report a partial
curve without that selection. Offline errors alone do not establish robot
pickup success.
