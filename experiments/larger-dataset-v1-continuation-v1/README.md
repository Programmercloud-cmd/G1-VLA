# Larger-dataset v1 continuation v1

Status: the controlled retry `larger-dataset-v1-continuation-v1-retry1` is running
with [startup verified](startup-complete.json). The captured snapshot records
35 finite local updates (cumulative weight lineage 4,131), all five runtime audits,
strict learned-parent loading, frozen processor/zero dropout, live cap and fresh
optimizer state. The first 32 actual sample coordinates match seed 4,138 and differ
from the parent's seed-42 prefix. No new checkpoint is certified by this snapshot.
Its earlier [immutable execution receipt](execution.json) binds actual
commit `84c1f7d`, the v2 protocol, run manifest and stable post-arm process identities.
The watchdog deadline is **22 October 2026 at 04:21:22 UTC**, slightly earlier than
the original deadline; its 21,227-second cap uses the remaining original budget.
That earlier receipt records the pre-audit initialization state and remains
unchanged history; the new startup proof supplies the completed runtime checks.

The [first new checkpoint](checkpoint-6144.json) is now certified at local step
2,048 / cumulative weight lineage 6,144: 20 files totaling 25,547,995,357 bytes,
including optimizer, scheduler and RNG state, passed complete remote SHA256 and
file/directory fsync checks. Training continues; this certificate establishes no
paired offline evaluation or whole-run result.

The first startup failed before any optimizer update. The
[startup receipt](startup-attempt-1.json) binds the immutable run manifest,
watchdog snapshots, final zero-update report and durable logs. The runner stopped
after 23.4 seconds following a watchdog command-line identity error; no strict
weight-load/runtime audit or new checkpoint was produced. That attempt and its
occupied outputs remain preserved. The segment starts
from the preserved, verified retry2 checkpoint 4,096 with fresh optimizer and
scheduler state, runs at most 4,096 additional local updates within the original
six-hour budget, and targets cumulative weight-lineage
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
[active v2 evaluation protocol](evaluation-protocol-startup-fix-v2.json) binds original
frozen samples and actual source 4,096 reports. New comparisons add the certified
6,144/8,192 policies to that source while preserving truthful local training
steps. Model/optimizer/scheduler/RNG files are saved, but the pinned iterable
loader does not restore an exact sample cursor. No automatic training resume,
protocol relaxation or robot-success claim is made.

The prior [partial results](../../docs/larger-dataset-v1-results.md) remain a
completed two-checkpoint evaluation of an interrupted training run. They are not
reclassified as this new segment, and its unsaved post 4,096 updates do not enter
this segment's weight lineage. Generated data, reports and weights stay outside
Git; compact declarations and artifact references identify each execution. The
[first-attempt protocol](evaluation-protocol.json) remains unchanged history.
