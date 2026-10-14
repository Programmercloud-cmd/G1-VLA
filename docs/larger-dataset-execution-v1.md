# Larger-dataset execution v1

The user approved the [larger-dataset plan](larger-dataset-training-plan-v1.md). Preparation is in progress; these commands describe the bounded workflow and do not claim completed training or evaluation. The [experiment record](../experiments/larger-dataset-v1/README.md) states the current scope. Historical pilot runs and their recorded results remain intact.

Run from the repository root with the pinned environments and configured storage available. The paths below are examples from the current host, not portable defaults. `VLA_CFG` must set `paths.derived`, `paths.runs`, and the existing durable storage connection consistently with these variables; use [machine.example.json](../machine.example.json) as the configuration template. Keep machine configuration, full inventories, recordings, generated datasets, predictions and weights outside Git.

```bash
VLA_CFG=reports/larger-dataset-v1-machine.json
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_DATA=/dev/shm/vla-large-data-v1
VLA_RUN_ROOT=storage/runs
VLA_RUN=larger-dataset-v1
VLA_SPLIT=experiments/larger-dataset-v1/split.json
VLA_PROTOCOL=experiments/larger-dataset-v1/evaluation-protocol.json
VLA_CONVERSION=reports/larger-dataset-v1-conversion.json
VLA_TRAIN="$VLA_DATA/lerobot/local/g1_inspire_larger_v1_train"
VLA_VALIDATION="$VLA_DATA/lerobot/local/g1_inspire_larger_v1_validation"
VLA_TRAIN_MANIFEST="$VLA_DATA/snapshots/train/manifest.json"
VLA_VALIDATION_MANIFEST="$VLA_DATA/snapshots/validation/manifest.json"
VLA_ZERO_TRAIN="reports/$VLA_RUN-step-zero-train.json"
VLA_ZERO_VALIDATION="reports/$VLA_RUN-step-zero-validation.json"
```

The original freeze used the commands below. Its output already exists: reuse it for this experiment. Helpers refuse existing outputs; a revised inventory or split requires a new declared version, not an overwrite. The inventory checks all 125 reviewed source JSON bindings and all referenced images in the 122 accepted episodes. It does not establish completion of the original copying process. The three held episodes stay excluded, and later arrivals cannot enter the frozen split.

```bash
"$VLA_PY" -B scripts/prepare_dataset_split.py --config "$VLA_CFG" inventory \
  --label-overrides annotations/goal-overrides-workspace-v1.json \
  --output reports/larger-dataset-v1-inventory.json
"$VLA_PY" -B scripts/prepare_dataset_split.py --config "$VLA_CFG" freeze \
  --inventory reports/larger-dataset-v1-inventory.json --output "$VLA_SPLIT"
```

The frozen seed-42 allocation contains 98 training and 24 validation episodes. The five historical pilot episodes remain in training. No genuine recorded session or layout identifiers were found; shared dates and folder order are not treated as groups. Validation has 15 left/front, eight right/front and one front-only episode. Training has 61, 36 and one respectively. This is an episode split, not evidence of generalization to new sessions or layouts.

Create the two new snapshots sequentially, checking successful completion before conversion. Only accepted `text.goal` values change in the copied JSON; original JSON provenance, images, states and commanded actions are preserved and verified. The old pilot preparation limit remains unchanged.

```bash
"$VLA_PY" -B scripts/prepare_dataset_split.py --config "$VLA_CFG" snapshot \
  --split-manifest "$VLA_SPLIT" --split train \
  --output "$VLA_DATA/snapshots/train" --max-mib 8192
"$VLA_PY" -B scripts/prepare_dataset_split.py --config "$VLA_CFG" snapshot \
  --split-manifest "$VLA_SPLIT" --split validation \
  --output "$VLA_DATA/snapshots/validation" --max-mib 8192
"$VLA_PY" -B scripts/prepare_dataset_split.py --config "$VLA_CFG" convert \
  --split-manifest "$VLA_SPLIT" \
  --train-snapshot "$VLA_DATA/snapshots/train" \
  --validation-snapshot "$VLA_DATA/snapshots/validation" \
  --output-parent "$VLA_DATA/lerobot" --report "$VLA_CONVERSION"
```

Conversion retains separate v3.0 copies beside the final v2.1 datasets. It checks exact float32 states/actions, task text and row order, decodes final video frame counts, and compares three source-image anchors per episode. GR00T absolute and relative statistics are computed only for training. Validation contributes no processor statistics. The conversion report binds the split, snapshots, final file inventories and training statistics.

Before bulk work, measure capacity on the actual host. The dedicated RAM-data budget is at most 8 GiB including estimated conversion temporary storage, while retaining at least 8 GiB free there and on the project volume. These are preparation preflight estimates, not a continuous conversion disk monitor. Preserve every existing output. The training run needs a separate durable output volume with at least 80 GiB free, at least 60 GiB available host RAM and 40 GiB free VRAM at launch. Its six-hour cap includes initialization and checkpoint writes; there is no automatic extension.

After successful conversion, freeze the processor and evaluation protocol on CPU. This step checks the conversion-report handoff, initializes `NEW_EMBODIMENT` from training-only statistics, and saves an independent processor. It selects 100 full 16-row windows per split before inference, covering all episodes. The source-hash-bound protocol fixes inference seeds 42/43/44, four denoising steps, target-change subsets and the validation selection rule.

```bash
"$VLA_PY" -B scripts/evaluate_gr00t_dataset.py --config "$VLA_CFG" prepare \
  --train-dataset "$VLA_TRAIN" --validation-dataset "$VLA_VALIDATION" \
  --train-manifest "$VLA_TRAIN_MANIFEST" \
  --validation-manifest "$VLA_VALIDATION_MANIFEST" \
  --split-manifest "$VLA_SPLIT" --conversion-manifest "$VLA_CONVERSION" \
  --processor-output "$VLA_DATA/processor" --protocol "$VLA_PROTOCOL"
```

Finish CPU preparation before archiving the dedicated data root: its processor files must be included, and no further writes may enter that root during hashing or transfer. The archive helper sends one compressed stream to the configured remote `derived/larger-dataset-v1` directory. The receiver verifies every member hash, fsyncs the archive and metadata, and publishes a new directory atomically. It refuses existing final or partial destinations and leaves the RAM source unchanged. Require a passing archive report before training. Paired step-zero evaluation can overlap the transfer because it reads the frozen data and writes reports elsewhere.

```bash
"$VLA_PY" -B scripts/prepare_dataset_split.py --config "$VLA_CFG" archive \
  --data-root "$VLA_DATA" --split-manifest "$VLA_SPLIT" \
  --conversion-manifest "$VLA_CONVERSION" \
  --manifest reports/larger-dataset-v1-archive-manifest.json \
  --report reports/larger-dataset-v1-archive.json
"$VLA_PY" -B scripts/evaluate_gr00t_dataset.py --config "$VLA_CFG" evaluate \
  --protocol "$VLA_PROTOCOL" --split both --step-zero \
  --train-report "$VLA_ZERO_TRAIN" --validation-report "$VLA_ZERO_VALIDATION"
```

A tiny real-host receiver fixture passed member-hash verification, file and directory fsync, and atomic publication on the configured storage. That fixture validates the transfer mechanism; it is not the production dataset archive. Its local evidence is `reports/larger-dataset-v1-receiver-fixture.json`.

The durable dataset artifact is `dataset.tar.gz`, with an adjacent `archive_manifest.json` and `archive_verification.json`; it is not an extracted dataset. For restoration, verify the compressed SHA256 from the archive report, extract only into a new empty directory, and verify the restored member sizes/hashes against the manifest before use. The archived processor must accompany the datasets. `/dev/shm` is temporary and cannot serve as the durable copy. Absolute paths inside historical evidence identify the original execution; relocation needs explicit new execution bindings rather than edits to frozen records.

After both step-zero reports and the durable dataset archive pass, freeze the implementation in Git and run the bounded training wrapper. It starts from pinned pretrained GR00T weights with a fresh optimizer/scheduler and training-only normalization. Both processor and action-head state dropout must be zero at runtime. The wrapper preserves four model-only checkpoints at steps 2048, 4096, 6144 and 8192 and records actual consumed/unique window coverage. A dry run validates inputs and prints the command; the real launch additionally enforces resource and committed-source checks.

```bash
"$VLA_PY" -B scripts/run_gr00t_dataset.py --config "$VLA_CFG" \
  --train-dataset "$VLA_TRAIN" --train-manifest "$VLA_TRAIN_MANIFEST" \
  --validation-manifest "$VLA_VALIDATION_MANIFEST" --split-manifest "$VLA_SPLIT" \
  --evaluation-protocol "$VLA_PROTOCOL" \
  --train-baseline-report "$VLA_ZERO_TRAIN" \
  --validation-baseline-report "$VLA_ZERO_VALIDATION" \
  --run "$VLA_RUN" --timeout-seconds 21600 --dry-run
```

Run the same command without `--dry-run` only after those prerequisites pass. Do not run GPU evaluation concurrently with training. Once training has exited, process the four checkpoints sequentially. Each inventory hashes/fsyncs weights on the configured storage host, avoiding repeated weight reads over SSHFS. Each paired evaluation performs one fresh strict model load for both splits and records finite action chunks; its train report also supplies the reload proof for archive publication.

```bash
for VLA_STEP in 2048 4096 6144 8192; do
  VLA_CHECKPOINT="$VLA_RUN_ROOT/$VLA_RUN/checkpoint-$VLA_STEP"
  VLA_INVENTORY="reports/$VLA_RUN-checkpoint-$VLA_STEP-inventory.json"
  VLA_EVAL_TRAIN="reports/$VLA_RUN-checkpoint-$VLA_STEP-train.json"
  VLA_EVAL_VALIDATION="reports/$VLA_RUN-checkpoint-$VLA_STEP-validation.json"
  "$VLA_PY" -B scripts/archive_gr00t_dataset.py --config "$VLA_CFG" \
    --source "$VLA_CHECKPOINT" --inventory-output "$VLA_INVENTORY" --hash-via-ssh || break
  "$VLA_PY" -B scripts/evaluate_gr00t_dataset.py --config "$VLA_CFG" evaluate \
    --protocol "$VLA_PROTOCOL" --split both --checkpoint "$VLA_CHECKPOINT" \
    --checkpoint-inventory "$VLA_INVENTORY" \
    --train-reference "$VLA_ZERO_TRAIN" --validation-reference "$VLA_ZERO_VALIDATION" \
    --train-report "$VLA_EVAL_TRAIN" --validation-report "$VLA_EVAL_VALIDATION" || break
  "$VLA_PY" -B scripts/archive_gr00t_dataset.py --config "$VLA_CFG" \
    --source "$VLA_CHECKPOINT" --checkpoint-inventory "$VLA_INVENTORY" \
    --run-config "reports/$VLA_RUN-run.json" --train-report "reports/$VLA_RUN-train.json" \
    --train-evaluation "$VLA_EVAL_TRAIN" --validation-evaluation "$VLA_EVAL_VALIDATION" \
    --protocol "$VLA_PROTOCOL" --record-name "$VLA_RUN-checkpoint-$VLA_STEP"
  if [ "$?" -ne 0 ]; then break; fi
done
```

Checkpoint publication is in place: it stores bound evidence under the configured archive and refers to the checkpoint already on durable run storage. It does not duplicate the weights. Retain the referenced run checkpoint permanently with its archive record; publishing the record does not permit deletion of the run directory. Check every command/report before proceeding; partial or failed runs cannot be reported as a completed four-checkpoint learning curve.

Select only after all four train/validation evaluation pairs pass:

```bash
"$VLA_PY" -B scripts/evaluate_gr00t_dataset.py --config "$VLA_CFG" select \
  --protocol "$VLA_PROTOCOL" \
  --train-reports "reports/$VLA_RUN-checkpoint-2048-train.json" \
    "reports/$VLA_RUN-checkpoint-4096-train.json" \
    "reports/$VLA_RUN-checkpoint-6144-train.json" \
    "reports/$VLA_RUN-checkpoint-8192-train.json" \
  --validation-reports "reports/$VLA_RUN-checkpoint-2048-validation.json" \
    "reports/$VLA_RUN-checkpoint-4096-validation.json" \
    "reports/$VLA_RUN-checkpoint-6144-validation.json" \
    "reports/$VLA_RUN-checkpoint-8192-validation.json" \
  --report "reports/$VLA_RUN-checkpoint-selection.json"
```

Validation group-balanced normalized MAE determines selection, with the earliest checkpoint breaking exact ties. Report all four arm/hand curves, native and normalized errors, changing-target subsets, per-episode metrics and coverage. Persistence remains a descriptive comparison; the historical pilot threshold is not reused as a mandatory barrier. Integrity checks remain required. This validation set is used for selection and is not a final untouched test set; offline action error does not establish successful robot behavior.
