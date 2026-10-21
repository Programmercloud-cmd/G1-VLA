# Larger-dataset v1 continuation v1

Status: authorized; the first startup failed before any optimizer update. The
[startup receipt](startup-attempt-1.json) binds the immutable run manifest,
watchdog snapshots, final zero-update report and durable logs. The runner stopped
after 23.4 seconds following a watchdog command-line identity error; no strict
weight-load/runtime audit or new checkpoint was produced. A controlled retry is
being reviewed within the original deadline, **22 October 2026 at 04:21:24 UTC**;
no retry has launched at this snapshot. The segment starts
from the preserved, verified retry2 checkpoint 4,096 with fresh optimizer and
scheduler state, runs at most 4,096 additional local updates under a fresh six-hour
cap whose original deadline remains fixed, and targets cumulative weight-lineage
steps 6,144 and 8,192. Native checkpoints
remain `checkpoint-2048` and `checkpoint-4096`; `continuation.json` records the
cumulative mapping without changing TrainerState.

See [execution decisions and commands](../../docs/larger-dataset-v1-continuation.md)
for strict source binding, peak 5e-5/cosine/103-update warmup, native segment
seed 4138, synchronous full-state SHA/fsync certification and independent cap
protection. Original splitseed 42, evaluation seeds 42/43/44, train-only processor,
zero state dropout and all earlier experiment bytes remain unchanged.

The [evaluation/archive workflow](../../docs/continuation-evaluation-v1.md) describes
post-training inference views and canonical full-state evidence. The separate
[evaluation protocol](evaluation-protocol.json) binds original
frozen samples and actual source 4,096 reports. New comparisons add the certified
6,144/8,192 policies to that source while preserving truthful local training
steps. Model/optimizer/scheduler/RNG files are saved, but the pinned iterable
loader does not restore an exact sample cursor. No automatic training resume,
protocol relaxation or robot-success claim is made.

The prior [partial results](../../docs/larger-dataset-v1-results.md) remain a
completed two-checkpoint evaluation of an interrupted training run. They are not
reclassified as this new segment, and its unsaved post 4,096 updates do not enter
this segment's weight lineage. Generated data, reports and weights stay outside
Git; compact declarations and artifact references identify each execution.
