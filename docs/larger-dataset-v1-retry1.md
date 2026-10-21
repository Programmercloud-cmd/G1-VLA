# Larger-dataset v1 deterministic startup retry

The first `larger-dataset-v1` launch stopped during processor validation after
37.19 seconds, before any loss or optimizer update. Its output and reports remain
at their original durable paths. The [incident record](../experiments/larger-dataset-v1/startup-order-incident.json)
binds that evidence and the CPU recovery proof. This attempt did not train a policy.

The frozen processor configuration and statistics matched exactly. Only two
adjacent entries in `embodiment_id.json` changed JSON order; their keys and numeric
IDs were unchanged. The pinned upstream processor constructs missing mapping
entries by iterating sets, then saves their insertion order. `PYTHONHASHSEED=1`,
set before the interpreter starts, reproduces the frozen order without source
changes or serialization patches. The training and data seed remains 42.

Two fresh CPU processes each ran both the real bootstrap processor and the pinned
training pipeline's `_create_dataset` with its real `DatasetFactory`, using the
preserved failed runtime configuration. All three saved processor files matched
the frozen hashes in both processes, and processor semantics matched the frozen
protocol. Neither process loaded model weights or initialized CUDA. An independent
CPU schedule simulation found identical nominal coordinates across hash seeds;
actual consumed coverage still requires the runner audit.

The fresh retry is `larger-dataset-v1-retry1`, declared in
[retry1-execution.json](../experiments/larger-dataset-v1/retry1-execution.json).
Retain the authoritative `larger-dataset-v1-step-zero-{train,validation}` baseline
reports. Keep the frozen dataset, protocol, sources, 8,192 updates, six-hour cap,
zero state dropout, fresh optimizer, four checkpoints, mandatory final root save,
and all finite/source/resource guards. The retry uses new output and report names;
existing failed-run files must remain intact.

After the CPU proof, publication and actual-host resource checks, use:

```bash
VLA_CFG=reports/larger-dataset-v1-machine.json
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_DATA=/dev/shm/vla-large-data-v1
VLA_REPORTS=storage/reports/larger-dataset-v1

PYTHONHASHSEED=1 "$VLA_PY" -B scripts/run_gr00t_dataset.py --config "$VLA_CFG" \
  --train-dataset "$VLA_DATA/lerobot/local/g1_inspire_larger_v1_train" \
  --train-manifest "$VLA_DATA/snapshots/train/manifest.json" \
  --validation-manifest "$VLA_DATA/snapshots/validation/manifest.json" \
  --split-manifest experiments/larger-dataset-v1/split.json \
  --evaluation-protocol experiments/larger-dataset-v1/evaluation-protocol.json \
  --train-baseline-report "$VLA_REPORTS/larger-dataset-v1-step-zero-train.json" \
  --validation-baseline-report "$VLA_REPORTS/larger-dataset-v1-step-zero-validation.json" \
  --run larger-dataset-v1-retry1 --timeout-seconds 21600 --dry-run
```

Inspect the dry run, then repeat without `--dry-run` and capture launcher output
to a fresh external `larger-dataset-v1-retry1-launcher.log` with overwrite disabled.
The empty `CUDA_VISIBLE_DEVICES` used for CPU proof subprocesses must not enter the
GPU launch. Record only selected `PYTHONHASHSEED` and `CUDA_VISIBLE_DEVICES` values
from the launched worker, together with its run-manifest hash and repository
commit, in the external execution-environment record. Never dump full process
environments.

After training exits, use the existing inventory, paired checkpoint evaluation,
in-place archive, and selection workflow with `VLA_RUN=larger-dataset-v1-retry1`.
Keep both baseline reference paths above unchanged. Do not evaluate checkpoints
while training is active. Preserve any completed prefix if the cap or a mandatory
integrity/resource guard stops the run; do not extend or resume automatically.
