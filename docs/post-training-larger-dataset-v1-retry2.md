# Larger-dataset v1 retry2 checkpoint evaluation

Retry2 stopped at **5,231 of 8,192 updates** after the unchanged local disk
reserve guard fired, before its six-hour cap. The worker exited and released the
GPU; the independent cap watchdog sent no signals. Only checkpoints **2,048 and
4,096** have certified saves. Their host-side inventories already exist, and
both paired evaluations, in-place archives and CPU analysis have passed.
Validation group-balanced normalized MAE is 0.196945982813923 at 2,048 and
0.14913777049933316 at 4,096, versus 0.7657805846075387 pretrained. See
[the partial results](larger-dataset-v1-results.md) and
[curated evidence](../experiments/larger-dataset-v1/partial-results.json). The [partial outcome](../experiments/larger-dataset-v1/retry2-outcome.json)
binds the final report, complete finite prefix and durable evidence.

This note records the current two-checkpoint workflow. Commands with existing
outputs must not be blindly rerun; inspect their reports and logs first. There
is no automatic training retry/resume and no formal four-pair selection. Earlier
attempts and original authoritative step-zero reports remain preserved.

Use a shell with `set -euo pipefail`, from `/home/test2/dev/VLA`. The fixed Python
hash seed is process-start configuration for the serialization recovery; it is
separate from the existing seed-42 training and seed-42/43/44 inference recipe.
Training uses `reports/larger-dataset-v1-retry2-machine.json` with local reports
under `reports/larger-dataset-v1-retry2`. Post-training inventory, evaluation and
archival use the original `reports/larger-dataset-v1-machine.json` configuration,
so all new reports and prediction artifacts go to external storage. Run weights
remain under `storage/runs`. The archive command receives the local training
run/final reports explicitly; the frozen protocol and original authoritative
baseline paths are unchanged.

```bash
set -euo pipefail
export PYTHONHASHSEED=1
export TMPDIR=/dev/shm/vla-larger-evaluation-tmp-20261021
VLA_EVAL_CFG=reports/larger-dataset-v1-machine.json
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_REPORTS=storage/reports/larger-dataset-v1
VLA_TRAIN_REPORTS=reports/larger-dataset-v1-retry2
VLA_RUN_ROOT=storage/runs
VLA_RUN=larger-dataset-v1-retry2
VLA_PROTOCOL=experiments/larger-dataset-v1/evaluation-protocol.json
VLA_ZERO_TRAIN="$VLA_REPORTS/larger-dataset-v1-step-zero-train.json"
VLA_ZERO_VALIDATION="$VLA_REPORTS/larger-dataset-v1-step-zero-validation.json"

# This RAM-backed directory was created mode 0700 before checkpoint evaluation.
# Verify the existing override in a fresh pinned Python process; do not fall back to /tmp.
"$VLA_PY" -B - <<'PYTMP'
import os, tempfile
from pathlib import Path
expected = Path(os.environ['TMPDIR']).resolve()
assert expected.is_dir() and os.access(expected, os.W_OK | os.X_OK)
assert expected.stat().st_mode & 0o777 == 0o700
assert Path(tempfile.gettempdir()).resolve() == expected
print('Verified temporary directory:', expected)
PYTMP
python3 scripts/project_storage.py --config "$VLA_EVAL_CFG" status
python3 - "$VLA_TRAIN_REPORTS/$VLA_RUN-run.json" "$VLA_TRAIN_REPORTS/$VLA_RUN-train.json" <<'PY'
from pathlib import Path
import hashlib, json, sys
sys.path.insert(0, str(Path('scripts').resolve()))
from archive_gr00t_dataset import prefix_integrity, checkpoint_files
from run_gr00t_pilot import read_audit
run_path, report_path = map(Path, sys.argv[1:])
run, report = json.loads(run_path.read_text()), json.loads(report_path.read_text())
assert run['run'] == report['run'] == 'larger-dataset-v1-retry2'
assert Path(report['run_manifest']).resolve() == run_path.resolve()
assert report['run_manifest_sha256'] == hashlib.sha256(run_path.read_bytes()).hexdigest()
assert run['evaluation_protocol_sha256'] == report['evaluation_protocol_sha256']
assert len(report['checkpoints']) == 4
assert {c['step'] for c in report['checkpoints']} == {2048, 4096, 6144, 8192}
audit = Path(run['audit'])
assert audit.is_file() and not audit.is_symlink() and audit.stat().st_size <= 64 * 1024**2
assert hashlib.sha256(audit.read_bytes()).hexdigest() == report['finite_audit_sha256']
records = read_audit(audit)
eligible = []
for checkpoint in report['checkpoints']:
    if checkpoint['passed'] is not True:
        continue
    step = checkpoint['step']
    prefix, event = prefix_integrity(records, step)
    checkpoint_files(Path(checkpoint['checkpoint']), step)
    coverage = Path(event['coverage_path'])
    assert hashlib.sha256(coverage.read_bytes()).hexdigest() == event['coverage_sha256']
    assert json.loads(coverage.read_text())['consumed_examples'] == step * 8
    eligible.append(step)
assert eligible == [2048, 4096], 'Expected only the two certified retry2 checkpoints'
print('Whole run completed:', report['passed'], 'aborted:', report['aborted'])
print('Certified checkpoint steps:', ' '.join(map(str, eligible)))
PY
```

The final report records a resource-reserve stop, not a time-cap stop. A certified
checkpoint prefix can still be evaluated and archived after root releases the
GPU; overall `training.passed=false` does not invalidate a complete finite
checkpoint prefix. Inspect the final bindings and printed certified steps with
root. No training extension or retry is being launched. This incomplete run cannot be
called a completed four-checkpoint curve or enter the historical selection command below.

Both durable host-side inventories have completed successfully, using remote
SHA256 and fsync. The completed execution evaluated 4,096, then 2,048, through the
tested additive launcher and published each pair in place. The following records
the commands used; it is not a command to repeat the completed evaluations. Output
names refuse occupied destinations. Inspect existing or partial results before
any execution, and preserve them. The fresh RAM temporary-directory preflight
also checked host memory, `/dev/shm`, root reserve and free VRAM; all existing
resource guards remain unchanged.

```bash
# Actual certified subset, in current evaluation order. Inventories already exist.
VLA_STEPS=(4096 2048)
for VLA_STEP in "${VLA_STEPS[@]}"; do
  VLA_CHECKPOINT="$VLA_RUN_ROOT/$VLA_RUN/checkpoint-$VLA_STEP"
  VLA_INVENTORY="$VLA_REPORTS/$VLA_RUN-checkpoint-$VLA_STEP-inventory.json"
  VLA_EVAL_TRAIN="$VLA_REPORTS/$VLA_RUN-checkpoint-$VLA_STEP-train.json"
  VLA_EVAL_VALIDATION="$VLA_REPORTS/$VLA_RUN-checkpoint-$VLA_STEP-validation.json"
  # Historical inventory command, already completed for both steps; do not rerun.
  # "$VLA_PY" -B scripts/archive_gr00t_dataset.py --config "$VLA_EVAL_CFG" \
  #   --source "$VLA_CHECKPOINT" --inventory-output "$VLA_INVENTORY" --hash-via-ssh
  "$VLA_PY" -B scripts/launch_gr00t_evaluation.py --config "$VLA_EVAL_CFG" evaluate \
    --protocol "$VLA_PROTOCOL" --split both --checkpoint "$VLA_CHECKPOINT" \
    --checkpoint-inventory "$VLA_INVENTORY" \
    --train-reference "$VLA_ZERO_TRAIN" --validation-reference "$VLA_ZERO_VALIDATION" \
    --train-report "$VLA_EVAL_TRAIN" --validation-report "$VLA_EVAL_VALIDATION"
  "$VLA_PY" -B scripts/archive_gr00t_dataset.py --config "$VLA_EVAL_CFG" \
    --source "$VLA_CHECKPOINT" --checkpoint-inventory "$VLA_INVENTORY" \
    --run-config "$VLA_TRAIN_REPORTS/$VLA_RUN-run.json" \
    --train-report "$VLA_TRAIN_REPORTS/$VLA_RUN-train.json" \
    --train-evaluation "$VLA_EVAL_TRAIN" --validation-evaluation "$VLA_EVAL_VALIDATION" \
    --protocol "$VLA_PROTOCOL" --record-name "$VLA_RUN-checkpoint-$VLA_STEP"
done
```

Each pair must prove one strict model load, matching model/processor/statistics,
finite arrays, one shared process and validation reuse of the same policy. The
inventory is computed on the storage host to avoid redundant weight hashing over
SSHFS. The checkpoint is already durable, so archive publication copies evidence
and retains the referenced run weights in place. Preserve those weights with
their archive records. The two passed archive summaries are under
`storage/reports/larger-dataset-v1/larger-dataset-v1-retry2-checkpoint-STEP-archive.json`.

Historical full-run selection command, preserved for the original protocol only.
**Do not run it for retry2:** checkpoints 6,144 and 8,192 do not exist. The current
result is a two-checkpoint partial curve, without formal protocol selection:

```bash
"$VLA_PY" -B scripts/launch_gr00t_evaluation.py --config "$VLA_EVAL_CFG" select \
  --protocol "$VLA_PROTOCOL" \
  --train-reports "$VLA_REPORTS/$VLA_RUN-checkpoint-2048-train.json" \
    "$VLA_REPORTS/$VLA_RUN-checkpoint-4096-train.json" \
    "$VLA_REPORTS/$VLA_RUN-checkpoint-6144-train.json" \
    "$VLA_REPORTS/$VLA_RUN-checkpoint-8192-train.json" \
  --validation-reports "$VLA_REPORTS/$VLA_RUN-checkpoint-2048-validation.json" \
    "$VLA_REPORTS/$VLA_RUN-checkpoint-4096-validation.json" \
    "$VLA_REPORTS/$VLA_RUN-checkpoint-6144-validation.json" \
    "$VLA_REPORTS/$VLA_RUN-checkpoint-8192-validation.json" \
  --report "$VLA_REPORTS/$VLA_RUN-checkpoint-selection.json"
```

For a completed future four-checkpoint experiment, selection minimizes validation
group-balanced normalized MAE, with earliest step breaking an exact tie. For this
run, report only the two observed checkpoint curves and changing-target subsets;
persistence remains descriptive. Validation is a selection set, and these offline
scores do not establish box selection, grasp or carry success on the robot.

The additive [summary helper](../scripts/summarize_larger_dataset.py) performs CPU
analysis completed after both checkpoint pairs passed, using the tracked source.
For reproduction, pass both train and validation
reports for each of 2,048 and 4,096 explicitly, and omit `--selection`. It does not
select a checkpoint. Loss-window means use only the 41,848 microbatches belonging
to 5,231 complete updates; four trailing micro-losses are retained separately.
Coverage describes sample exposure, so its 41,852 count includes those four and
also reports the complete-update count. Baseline-only helper validation outputs
are tests, not the production partial learning curve.
