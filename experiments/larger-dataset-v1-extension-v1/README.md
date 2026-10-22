# Larger-dataset v1 extension to cumulative 12,288

Status: **completed and evaluated**. All 4,096 additional updates, both full-state
saves, paired train/validation evaluations and durable archives passed. The
[source-inclusive ranking](selected-policy.json) selected cumulative **12,288**:
validation four-group mean normalized MAE **0.104795**, 5.4% below the 8,192 source
(0.110770). The intermediate 10,240 checkpoint regressed to 0.133912. Gains are
not uniform: left-hand normalized MAE regressed slightly at 12,288.

See [final results and curves](../../docs/larger-dataset-v1-extension-results.md),
[complete evidence](results.json), the [sealed publication](results-publication.json)
and [training outcome](training-outcome.json). The [execution](execution.json),
[startup](startup-complete.json) and [checkpoint certificates](checkpoint-12288.json)
remain historical snapshots. Actual training executed commit `f7ce2ffe` and ended
within its original six-hour deadline, with all writers exited and no cap signal.
The prior optimizer/scheduler/RNG files remain preserved and were not restored.

| Native local save | Cumulative weight lineage |
| ---: | ---: |
| 2,048 | 10,240 |
| 4,096 | 12,288 |

Peak learning rate is 2.5e-5, with 103 warmup updates and native seed 8,234.
Microbatch one, accumulation eight, zero state dropout, train-only normalization,
the 98/24 split and 98 native training shards remain unchanged. Both checkpoints
contain full native state and passed complete storage-host SHA/fsync before
subsequent work. The six-hour cap included loading and all writes; the
8 GiB reserves stayed active. No automatic retry or extension.

The source lineage binds two native-4,096 finite prefixes and 65,536 exposures;
it excludes the original unsaved 1,135 updates. The
[sampling projection](sampling-projection.json) was fixed before training. Actual full-sequence coordinates matched it: 32,768
new unique windows/86 episodes; combined lineage 98,304 exposures, 36,920 unique
windows and 97/98 episodes. This does not establish numerical equivalence or
justify changing sampling for complete coverage.
[Readiness](storage-readiness.json) and [source ancestry](ancestry.json) retain
the separately prepared machine/source evidence.

See [training scope and historical dry-run command](../../docs/larger-dataset-v1-extension.md)
and the [separate evaluation/archive workflow](../../docs/extension-evaluation-v1.md).
Model-only views are nonresumable. Canonical checkpoints preserve native state,
but exact iterable cursor restoration is unsupported and production optimizer
reload remains untested. Deployment and hardware competence are separate work.
