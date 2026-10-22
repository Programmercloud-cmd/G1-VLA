# Larger-dataset v1 extension to cumulative 12,288

Status: running with [startup verified](startup-complete.json). The captured
snapshot records 46 finite local updates (cumulative weight lineage 8,238),
strict learned-8,192 loading, all five runtime audits, frozen processor/zero
dropout, live cap and fresh optimizer state. The first 32 consumed coordinates
match seed 8,234. This is a historical startup snapshot, not a new checkpoint
or final result. The [execution receipt](execution.json) binds actual commit
`f7ce2ffe` and the six-hour deadline of **22 October 2026 at 13:23:44 UTC**.
The bounded segment targets 4,096 additional updates; source optimizer/scheduler/RNG
files remain preserved and were not restored.

| Native local save | Cumulative weight lineage |
| ---: | ---: |
| 2,048 | 10,240 |
| 4,096 | 12,288 |

Peak learning rate is 2.5e-5, with 103 warmup updates and native seed 8,234.
Microbatch one, accumulation eight, zero state dropout, train-only normalization,
the 98/24 split and 98 native training shards remain unchanged. Both checkpoints
must contain full native state and pass complete storage-host SHA/fsync before
training continues. A fresh six-hour cap includes loading and all writes; the
8 GiB reserves remain active. No automatic retry or extension.

The source lineage binds two native-4,096 finite prefixes and 65,536 exposures;
it excludes the original unsaved 1,135 updates. The
[sampling projection](sampling-projection.json) is fixed-seed CPU evidence, not
actual new consumption or a reason to alter sampling for complete coverage.
[Readiness](storage-readiness.json) and [source ancestry](ancestry.json) retain
the separately prepared machine/source evidence.

See [training scope and planned dry-run command](../../docs/larger-dataset-v1-extension.md)
and the [separate evaluation/archive workflow](../../docs/extension-evaluation-v1.md).
Model-only views are nonresumable. Canonical checkpoints preserve native state,
but exact iterable cursor restoration is unsupported and production optimizer
reload remains untested. Deployment and hardware competence are separate work.
