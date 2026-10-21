# Larger-dataset v1 retry1 checkpoint evaluation

These commands are prepared for the fresh `larger-dataset-v1-retry1` run. They
are not authorization to share the GPU while training is active. Root must first
confirm the training process has exited and release its GPU lease. The final
training report must exist before checkpoint inventory, evaluation or archival.
The original failed zero-update run and authoritative step-zero reports remain
preserved at their original names.

Use a shell with `set -euo pipefail`, from `/home/test2/dev/VLA`. The fixed Python
hash seed is process-start configuration for the serialization recovery; it is
separate from the existing seed-42 training and seed-42/43/44 inference recipe.
The machine configuration continues to use external reports and durable runs.

```bash
set -euo pipefail
export PYTHONHASHSEED=1
VLA_CFG=reports/larger-dataset-v1-machine.json
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_REPORTS=storage/reports/larger-dataset-v1
VLA_RUN_ROOT=storage/runs
VLA_RUN=larger-dataset-v1-retry1
VLA_PROTOCOL=experiments/larger-dataset-v1/evaluation-protocol.json
VLA_ZERO_TRAIN="$VLA_REPORTS/larger-dataset-v1-step-zero-train.json"
VLA_ZERO_VALIDATION="$VLA_REPORTS/larger-dataset-v1-step-zero-validation.json"

python3 scripts/project_storage.py --config "$VLA_CFG" status
python3 - "$VLA_REPORTS/$VLA_RUN-run.json" "$VLA_REPORTS/$VLA_RUN-train.json" <<'PY'
from pathlib import Path
import hashlib, json, sys
sys.path.insert(0, str(Path('scripts').resolve()))
from archive_gr00t_dataset import prefix_integrity, checkpoint_files
from run_gr00t_pilot import read_audit
run_path, report_path = map(Path, sys.argv[1:])
run, report = json.loads(run_path.read_text()), json.loads(report_path.read_text())
assert run['run'] == report['run'] == 'larger-dataset-v1-retry1'
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
assert eligible, 'No certified checkpoint prefix is available'
print('Whole run completed:', report['passed'], 'aborted:', report['aborted'])
print('Certified checkpoint steps:', ' '.join(map(str, eligible)))
PY
```

The final report may describe a bounded run that stopped at its cap. A certified
checkpoint prefix can still be evaluated and archived after root releases the
GPU; overall `training.passed=false` does not invalidate a complete finite
checkpoint prefix. Inspect the final bindings and printed certified steps with
root. The cap is not extended. An incomplete run cannot be called a completed
four-checkpoint curve or enter the four-checkpoint selection command below.

For each checkpoint, create its durable host-side inventory, run paired evaluation
through the tested additive launcher, then publish its evidence in place. Every
output name is new and refuses occupied destinations. Inspect any occupied or
partial output before proceeding; do not overwrite or blindly repeat the loop.

```bash
# Use only the certified steps printed above. This example is the full-run case.
VLA_STEPS=(2048 4096 6144 8192)
for VLA_STEP in "${VLA_STEPS[@]}"; do
  VLA_CHECKPOINT="$VLA_RUN_ROOT/$VLA_RUN/checkpoint-$VLA_STEP"
  VLA_INVENTORY="$VLA_REPORTS/$VLA_RUN-checkpoint-$VLA_STEP-inventory.json"
  VLA_EVAL_TRAIN="$VLA_REPORTS/$VLA_RUN-checkpoint-$VLA_STEP-train.json"
  VLA_EVAL_VALIDATION="$VLA_REPORTS/$VLA_RUN-checkpoint-$VLA_STEP-validation.json"
  "$VLA_PY" -B scripts/archive_gr00t_dataset.py --config "$VLA_CFG" \
    --source "$VLA_CHECKPOINT" --inventory-output "$VLA_INVENTORY" --hash-via-ssh
  "$VLA_PY" -B scripts/launch_gr00t_evaluation.py --config "$VLA_CFG" evaluate \
    --protocol "$VLA_PROTOCOL" --split both --checkpoint "$VLA_CHECKPOINT" \
    --checkpoint-inventory "$VLA_INVENTORY" \
    --train-reference "$VLA_ZERO_TRAIN" --validation-reference "$VLA_ZERO_VALIDATION" \
    --train-report "$VLA_EVAL_TRAIN" --validation-report "$VLA_EVAL_VALIDATION"
  "$VLA_PY" -B scripts/archive_gr00t_dataset.py --config "$VLA_CFG" \
    --source "$VLA_CHECKPOINT" --checkpoint-inventory "$VLA_INVENTORY" \
    --run-config "$VLA_REPORTS/$VLA_RUN-run.json" \
    --train-report "$VLA_REPORTS/$VLA_RUN-train.json" \
    --train-evaluation "$VLA_EVAL_TRAIN" --validation-evaluation "$VLA_EVAL_VALIDATION" \
    --protocol "$VLA_PROTOCOL" --record-name "$VLA_RUN-checkpoint-$VLA_STEP"
done
```

Each pair must prove one strict model load, matching model/processor/statistics,
finite arrays, one shared process and validation reuse of the same policy. The
inventory is computed on the storage host to avoid redundant weight hashing over
SSHFS. The checkpoint is already durable, so archive publication copies evidence
and retains the referenced run weights in place. Preserve those weights with
their archive records. The four archive summaries appear under
`storage/reports/larger-dataset-v1/larger-dataset-v1-retry1-checkpoint-STEP-archive.json`.

Only after all four pairs and archives pass, select the checkpoint:

```bash
"$VLA_PY" -B scripts/launch_gr00t_evaluation.py --config "$VLA_CFG" select \
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

Selection minimizes validation group-balanced normalized MAE, with earliest step
breaking an exact tie. Report all four group curves and changing-target subsets;
persistence remains descriptive. Validation is a selection set, and these offline
scores do not establish box selection, grasp or carry success on the robot.
