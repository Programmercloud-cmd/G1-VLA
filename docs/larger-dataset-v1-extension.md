# Bounded extension from cumulative 8,192 to 12,288

Status: running with [startup verified](../experiments/larger-dataset-v1-extension-v1/startup-complete.json)
under actual commit `f7ce2ffe`. The captured snapshot records 46 finite local
updates (cumulative weight lineage 8,238), all five runtime audits, strict
learned-8,192 loading, frozen processor/zero dropout, live cap and fresh optimizer
state. The first 32 consumed coordinates match seed 8,234. No new checkpoint
or final result is certified by this startup snapshot. The
[execution receipt](../experiments/larger-dataset-v1-extension-v1/execution.json)
binds the original actual-launch six-hour cap, ending **22 October 2026 at
13:23:44 UTC**, including loading and saves. This is a new 4,096-update segment
from the [selected cumulative-8,192 policy](../experiments/larger-dataset-v1-continuation-v1/selected-policy.json).
Earlier runs, full checkpoints, selected results and all frozen source/protocol
bytes remain preserved. The new run identity is `larger-dataset-v1-extension-v1`.

The source is the canonical **full** checkpoint
`archive:runs/larger-dataset-v1-continuation-v1-retry1/checkpoint-4096`, whose native
Trainer step is 4,096 and cumulative weight lineage is 8,192. The source contains
optimizer/scheduler/RNG state. This extension intentionally loads only its model
weights with a strict tensor-loading audit and starts fresh native AdamW and cosine
state. The prior cosine schedule ended at learning rate zero; reusing it for a new
cycle would require changing schedule semantics. A fresh bounded cycle makes the
reset explicit, preserves the old moments for future work, and avoids a manual
scheduler-state patch. It is not an exact optimizer or sample-cursor resume.

The new peak learning rate is **2.5e-5**, half the preceding segment's peak, with
native cosine decay over 4,096 local updates and warmup ratio 0.025 (103 updates).
The parent ended at zero; the new peak is a deliberate fresh cycle rather than a
claim to continue its instantaneous rate. Microbatch one, accumulation eight,
clipping, tuning flags, train-only normalization and zero state dropout retain the
pinned upstream configuration. No model or horizon changes are introduced.

| Weight lineage | Native local Trainer step | New checkpoint directory |
| --- | ---: | --- |
| Source cumulative 8,192 | 0 at fresh segment start | Preserved source `checkpoint-4096` |
| Cumulative 10,240 | 2,048 | `checkpoint-2048` |
| Cumulative 12,288 | 4,096 | `checkpoint-4096` |

Each new `continuation.json` sidecar records the local/cumulative mapping and
`larger_dataset_extension_checkpoint` identity. TrainerState and directory names
remain honest native local steps. Both saves must include `optimizer.pt`,
`scheduler.pt`, `rng_state.pth`, TrainerState, model and processor files. The save
callback blocks further updates until state structure, all three frozen processor
hashes, every-file remote SHA256/fsync and directory fsync pass. The native final
model-only save at the output root is retained within the same cap; it does not
replace either full-state checkpoint or become a resumable reference.

The source ancestry contains **two separately certified native 4,096-update
prefixes**: original cumulative 0–4,096 and the preceding fresh segment
4,096–8,192. Their loss coordinates supply 65,536 prior exposures, 36,545 unique
windows and 96 of 98 training episodes. The old unsaved 1,135 updates after the
original checkpoint are excluded. The new source-prefix schema binds both native
runs/reports/audits/coverage records; no fabricated native 8,192-step prefix or
TrainerState is created. Episode identities and order must match before coverage
is combined.

The native seed-only adapter uses **8,234 = 42 + 8,192** before upstream `set_seed`.
Frozen split seed 42 and evaluation seeds 42/43/44 remain unchanged. The dataset
retains its 98 native shards and configured shard size 256; an uncapped
window-count estimate does not replace native sharding. The
[fixed-seed CPU projection](../experiments/larger-dataset-v1-extension-v1/sampling-projection.json)
projects 32,768 unique new-segment coordinates across 86 episodes, with first-32
SHA256 `f289fd3c69ffb9c0fb77602f9dacac43474865a567238f342490f0cdf70ce9e6`.
Combined projected lineage is 98,304 exposures, 36,920 unique windows and 97 of 98
episodes. These are projections, not actual new consumption, augmentation/model
numerical equivalence or a requirement to extend training for complete coverage.
Runtime audits will capture actual loss-computed coordinates.

The source full inventory is checked through its completed storage-host SHA/fsync
certificate, fresh remote/mounted signatures and bounded small metadata. Its
approximately 13 GB optimizer is neither loaded nor locally rehashed for this
weights-only start. The original selected paired inference/view/archive chain is
bound separately to the canonical full checkpoint. A model-only inference view
must never be passed as a full-state resume reference.

The pinned Trainer supports native model/optimizer/scheduler/RNG state restoration,
but its iterable loader has no serialized cursor, uses `ignore_data_skip=True`
and reseeds on resume. **Exact sample-position replay remains unsupported.**
Production optimizer reload has not been exercised; the earlier pinned CPU
Trainer fixture is the reload evidence. The new run audits fresh optimizer state
zero and local step zero, not restoration of the source moments.

A fresh **21,600-second maximum** starts at actual launch and includes model
loading, updates, both full-state writes/certification and the native final root
save. The independent local cap uses the unchanged watchdog engine through a tiny
adapter that changes only the expected runner pathname. PID/UID/start ticks,
command line, group and session checks, immutable armed proof, local-only timing,
TERM→bounded KILL and exact-owned-SSHFS D-state fallback remain intact. It never
remounts or restarts training. The worker uses canonical argv without `-B`, while
`PYTHONDONTWRITEBYTECODE=1` remains in its environment. Remote SHA/fsync calls are
bounded by the same remaining deadline. No automatic retry, extension or resume.

Startup requires at least 120 GiB free durable output, 60 GiB available host RAM
and 40 GiB free VRAM. The existing 8 GiB filesystem reserves and runtime memory
guard remain unchanged. Checkpoints stay on configured durable storage; active
training audits stay local. This extension's full-state/view/archive workflow
requires configured SSH-backed storage, as did the preceding continuation; a
local-disk-only path remains unsupported.

The following is the planned official **CPU-only dry run**, after source and new
protocol publication. It uses new no-overwrite outputs and does not load a model:

```bash
export PYTHONHASHSEED=1
export TMPDIR=/dev/shm/vla-extension-tmp-20261022
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_CFG=reports/larger-dataset-v1-extension-v1-machine.json
VLA_SOURCE=larger-dataset-v1-continuation-v1-retry1
VLA_PARENT_REPORTS=reports/larger-dataset-v1-continuation-v1
VLA_EXTERNAL_REPORTS=storage/reports/larger-dataset-v1
VLA_PARENT_PREFIX=$VLA_SOURCE-local-4096-cumulative-8192

"$VLA_PY" -B scripts/run_gr00t_extension.py --config "$VLA_CFG" \
  --source-run-manifest "$VLA_PARENT_REPORTS/$VLA_SOURCE-run.json" \
  --source-train-report "$VLA_PARENT_REPORTS/$VLA_SOURCE-train.json" \
  --source-inventory "storage/reports/$VLA_SOURCE-checkpoint-local-4096-cumulative-8192/full-checkpoint-inventory.json" \
  --source-archive-report "$VLA_EXTERNAL_REPORTS/$VLA_PARENT_PREFIX-archive.json" \
  --source-evaluation-train "$VLA_EXTERNAL_REPORTS/$VLA_PARENT_PREFIX-train.json" \
  --source-evaluation-validation "$VLA_EXTERNAL_REPORTS/$VLA_PARENT_PREFIX-validation.json" \
  --source-comparison "$VLA_EXTERNAL_REPORTS/$VLA_PARENT_PREFIX-comparison.json" \
  --source-view-manifest "$VLA_EXTERNAL_REPORTS/$VLA_PARENT_PREFIX-view.json" \
  --source-view-inventory "$VLA_EXTERNAL_REPORTS/$VLA_PARENT_PREFIX-view-inventory.json" \
  --selected-policy experiments/larger-dataset-v1-continuation-v1/selected-policy.json \
  --previous-protocol experiments/larger-dataset-v1-continuation-v1/evaluation-protocol-startup-fix-v2.json \
  --evaluation-protocol experiments/larger-dataset-v1/evaluation-protocol.json \
  --extension-protocol experiments/larger-dataset-v1-extension-v1/evaluation-protocol.json \
  --run larger-dataset-v1-extension-v1 --timeout-seconds 21600 --dry-run
```

An actual launch requires source review/publication, a frozen declaration,
resource readiness and the exclusive training GPU lease. Only then omit
`--dry-run` and preserve launcher stdout at a fresh local path. Normal CUDA
visibility must be retained; the empty visibility used by CPU proofs must not
enter training. Keep the storage PC, drive and tunnel available.

After the final report and process exit, follow the separate
[evaluation/archive workflow](extension-evaluation-v1.md) on certified checkpoints.
It preserves raw native steps and labels cumulative lineage separately. Selection
uses the original frozen windows and compares source 8,192 with certified new
10,240/12,288 policies; any incomplete comparison is labeled explicitly. Persistence
is descriptive, not a mandatory gate. No training loss or offline selection
establishes physical box selection, grasp or carrying competence.

Focused CPU checks use the pinned interpreter and empty CUDA visibility:

```bash
CUDA_VISIBLE_DEVICES='' "$VLA_PY" -B -m unittest discover -s tests -p test_dataset_extension.py -v
CUDA_VISIBLE_DEVICES='' "$VLA_PY" -B -m unittest discover -s tests -p test_extension_cap.py -v
```

These cover the native seed/learning-rate cycle, strict loading information,
local/cumulative bounds, both finite source segments with the unsaved tail
excluded, and real disposable canonical-exec/watchdog identities. The separate
native processor preflight follows `set_seed` → `Config.validate()` → dataset
factory with `model=None`; it does not load model or optimizer weights or use CUDA.
