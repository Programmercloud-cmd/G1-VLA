# Larger-dataset v1 continuation from checkpoint 4,096

Status: the controlled retry `larger-dataset-v1-continuation-v1-retry1` is running
and initializing under commit `84c1f7d`. The [execution receipt](../experiments/larger-dataset-v1-continuation-v1/execution.json)
binds its actual manifest, v2 protocol, armed watchdog and stable process identities.
Its conservative deadline is **22 October 2026 at 04:21:22 UTC**, earlier than the
original 04:21:24 UTC deadline; the 21,227-second cap consumes only the remaining
original budget. At the startup snapshot, zero new optimizer updates/checkpoints
are certified. Strict learned-weight loading, processor/runtime and fresh-optimizer
audits are pending; the startup record does not establish their success.

The first startup stopped after 23.4 seconds with zero
optimizer updates and no new checkpoint. The [failed-start receipt](../experiments/larger-dataset-v1-continuation-v1/startup-attempt-1.json)
preserves its manifest, final report, armed/final watchdog records and logs. The
watchdog rejected a worker command-line identity change after Python re-exec;
the runner then aborted. Strict weight loading and runtime initialization were
not proven. Its run, final report and occupied outputs remain preserved.
The first run was `larger-dataset-v1-continuation-v1`. Its source is the verified durable
`larger-dataset-v1-retry2/checkpoint-4096`, whose paired validation normalized MAE
was 0.149138. The earlier [partial run and results](larger-dataset-v1-results.md),
checkpoints, failed attempts, frozen protocol and original data remain preserved.

The user authorized **4,096 additional updates**, pretrained initialization is
not used for this segment. The source save contains learned model weights but
no optimizer/scheduler state, so this is a strict warm start with fresh AdamW
moments and a fresh schedule, not an exact restoration of the interrupted run.
The original six-hour cap includes model loading, updates, checkpoint
writes, full-state validation, hashes and fsync. No automatic extension or retry.

| Weight lineage | Native local Trainer step | Native output directory |
| --- | ---: | --- |
| Source 4,096 | 0 at new segment start | Preserved source checkpoint |
| Cumulative 6,144 | 2,048 | New run `checkpoint-2048` |
| Cumulative 8,192 | 4,096 | New run `checkpoint-4096` |

TrainerState and directory names remain truthful local steps. Each new checkpoint
contains `continuation.json` with the explicit source/segment/cumulative mapping.
The unsaved 1,135 updates after the old checkpoint are not part of this lineage.
New finite/coverage audits describe the fresh segment and separately bind the
source's certified 4,096-update finite prefix.

The new peak learning rate is **5e-5**, with native cosine decay across 4,096 local
updates and warmup ratio 0.025 (103 updates). It is about 7.6% below the parent
checkpoint's audited instantaneous rate, 5.413320221727027e-5, and half the earlier
peak 1e-4. This limits the restart jump while fresh moments form; it is a new
bounded schedule, not continuation of the lost scheduler. Microbatch one,
accumulation eight, train-only statistics, tuning flags and zero state dropout
are unchanged.

A narrow adapter sets native `config.data.seed=4138` before upstream `set_seed`:
42 + source checkpoint 4,096, following the pinned trainer's reseed convention.
The adapter verifies that it changes only this field. Frozen split seed 42 and
inference seeds 42/43/44 remain unchanged. Starting seed 42 again would repeat the
same initial 32,768 sample identities and 87 episodes. CPU reconstruction exactly
matches that actual parent prefix; the seed 4138 nominal schedule differs, exposing
nine previously unseen episodes. Its combined nominal lineage covers 96/98
training episodes and 36,545/37,281 unique windows. These are CPU sample-identity
projections, not actual continuation consumption or numerical-equivalence claims;
runtime audits will measure consumption.

Native `save_only_model=False` saves `optimizer.pt`, `scheduler.pt`,
`rng_state.pth`, TrainerState and model/processor files. A real pinned CPU Trainer
fixture saved then restored model weights, AdamW moments and scheduler progress.
However, the pinned iterable loader has no serialized iterator cursor and its
resume path sets `ignore_data_skip=True` and reseeds. Therefore these checkpoints
support native state restoration but **not exact sample-position replay**. No
custom dataloader or exact-restart claim is introduced.

Each save blocks the next update until full-state structure/local steps and the
frozen processor hashes pass, every file receives a durable SHA256/fsync inventory,
and checkpoint directories are fsynced. The two saves retain model, optimizer,
scheduler and RNG state. Startup requires at least 120 GiB free output, 60 GiB host
RAM and 40 GiB available VRAM; existing 8 GiB project/report/runtime output reserves
remain active. Root headroom was arranged separately before launch. Training
audits remain local; checkpoints stay on configured durable storage.

An independent local watchdog verifies launcher/worker PID, owner, start ticks,
command line, process group and session using `/proc`, with no SSHFS access in its
loop. At its conservative process-start-plus-six-hours deadline it sends TERM,
then bounded KILL. If the original worker remains blocked in FUSE D-state after
KILL, it may terminate only the captured, identity-verified daemon for this exact
project mount, preserving partial output and requiring explicit later mount
recovery. It never remounts or restarts training. Remote hash/fsync SSH commands
also have timeouts bounded by the original remaining deadline.

The commands below record the first attempt's official CPU dry run after its
source/protocol freeze. Its run name and outputs are now occupied and must stay
preserved; a controlled retry needs separately reviewed sources, protocol and
execution locations within the unchanged deadline. Use the
machine-local configuration and RAM-backed temporary directory prepared during
readiness. This command does not load a model or launch GPU training:

```bash
VLA_CFG=reports/larger-dataset-v1-continuation-v1-machine.json
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_PARENT=larger-dataset-v1-retry2
VLA_REPORTS=storage/reports/larger-dataset-v1
export PYTHONHASHSEED=1
export TMPDIR=/dev/shm/vla-continuation-tmp-20261022

"$VLA_PY" -B scripts/run_gr00t_continuation.py --config "$VLA_CFG" \
  --source-run-manifest "reports/$VLA_PARENT/$VLA_PARENT-run.json" \
  --source-train-report "reports/$VLA_PARENT/$VLA_PARENT-train.json" \
  --source-inventory "$VLA_REPORTS/$VLA_PARENT-checkpoint-4096-inventory.json" \
  --source-archive-report "$VLA_REPORTS/$VLA_PARENT-checkpoint-4096-archive.json" \
  --source-evaluation-train "$VLA_REPORTS/$VLA_PARENT-checkpoint-4096-train.json" \
  --source-evaluation-validation "$VLA_REPORTS/$VLA_PARENT-checkpoint-4096-validation.json" \
  --evaluation-protocol experiments/larger-dataset-v1/evaluation-protocol.json \
  --continuation-protocol experiments/larger-dataset-v1-continuation-v1/evaluation-protocol.json \
  --run larger-dataset-v1-continuation-v1 --timeout-seconds 21600 --dry-run
```

Before any controlled launch,
root reviews/publishes the applicable sources, freezes current resource evidence and
grants the sole training GPU lease. Only then omit `--dry-run`, preserving fresh
local logs with no overwrite. Retain normal CUDA visibility; the empty visibility
used by CPU tests must not enter training. Runtime must prove one strict load of
the bound learned 4,096 checkpoint, fresh optimizer state 0/local step 0/cumulative
base 4,096, the seed-only configuration change and unchanged frozen processor.

The [new experiment declaration](../experiments/larger-dataset-v1-continuation-v1/README.md)
records the separate comparison protocol. The [continuation evaluation/archive workflow](continuation-evaluation-v1.md)
uses separate model-only inference views while preserving canonical full-state
checkpoints. Evaluation runs only after training exits and the GPU lease is released,
using the same original samples/statistics
and paired seeds. Formal comparison to the source requires both new certified
pairs; incomplete prefixes are reported honestly. Offline scores do not establish
box selection, grasping, carrying or hardware deployment readiness.
