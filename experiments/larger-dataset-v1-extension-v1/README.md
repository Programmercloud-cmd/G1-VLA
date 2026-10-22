# Larger-dataset v1 extension to cumulative 12,288

Status: authorized and preparing; no new training has launched. The bounded new
segment strictly loads the verified selected cumulative-8,192 weights and starts
fresh native AdamW/cosine state for 4,096 additional updates. The prior full
optimizer/scheduler/RNG files remain preserved and are not restored.

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
