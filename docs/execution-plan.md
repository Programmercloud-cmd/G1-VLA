# Completed smoke, pilot fitting results, and next proposal

The first milestone is complete: five episodes, both reference conversions, raw-value and image checks, GR00T loading/statistics, two optimizer steps, fresh-process reload, and one checksum-verified durable checkpoint. See the [recorded results](../experiments/pilot-smoke-v2/summary.json), [pilot content identity](../experiments/pilot-smoke-v2/dataset_manifest.json), and [external artifact references](../experiments/pilot-smoke-v2/artifacts.json).

## What is established

The converted state/action arrays match the original float32 values. Camera mapping, task text, timestamps, episode ordering, and 26-dimensional joint ordering were checked. The saved processor returns finite absolute action chunks of shape `(16, 26)`. Vision/language remained frozen while the projector and action head were trainable.

On the validation machine, batch 1 reached 38,608 MiB sampled GPU memory. The 12,583,318,231-byte checkpoint was saved model-only and reloaded offline. Upstream also saved a duplicate model at the run root, making transient output roughly 25.17 GB. A single complete checkpoint was archived; all 33 archived checkpoint/provenance files matched their source hashes.

These checks do not establish pickup competence, closed-loop stability, or generalization. No robot commands have been sent.

## Completed 512-update fitting probe

Run `pilot-fit-v1` completed with implementation commit `4685d0dd04611cce5f1dbfbab4ef2e40caf7f67d`. All 4,096 microbatch losses and 512 gradient checks were finite, and fresh-process reload plus paired evaluation passed. Training took 865.1 seconds and sampled peak GPU usage was 38,764 MiB. The complete checkpoint and comparison evidence were durably archived; original smoke and pilot outputs remain preserved.

| Setting | Executed value |
| --- | --- |
| Initialization | Pinned pretrained GR00T model |
| Microbatch | 1 |
| Gradient accumulation | 8, giving effective batch 8 |
| Loader workers | 0 |
| Optimizer updates | 512 |
| Checkpoints | One model-only save at step 512 |
| Shard size | 256 |
| Episode sampling rate | 1.0 |
| Shards per epoch | 5, covering the five whole-episode shards |
| Learning rate / warmup | 1e-4 / 0.05 |
| State dropout | 0.2 |

The run used the configured run root plus `pilot-fit-v1/checkpoint-512`. Its checkpoint and upstream duplicate occupy 23.44 GiB; minimum free output space was 16.02 GiB and minimum available host RAM was 69.24 GiB. Accumulation 8 is now validated for this recipe; larger microbatches remain untested. Setup and readiness checks do not start training.

The candidate and two-step reference used the same 100 pilot starts, full 16-frame targets, three paired inference seeds, and four denoising steps. Candidate MAE improved by 11–45% across groups, but every group remained worse than current-state persistence. The gap also remains on the predeclared changing-target subsets below. These checks establish finite, comparable training-set execution, not task competence. See [the full results and trajectory plot](../experiments/pilot-fit-v1/README.md).

## Commands used for the completed run

These commands use the default paths. Choose fresh run/report names to reproduce the experiment; existing outputs are preserved. The baseline freezes 100 eligible starts across the five episodes, then evaluates all three inference seeds before training begins. The candidate reuses that exact manifest and baseline evidence.

```bash
python3 scripts/evaluate_gr00t_pilot.py \
  --checkpoint /dev/shm/vla-groot-smoke/pilot-smoke-v2/checkpoint-2 \
  --dataset derived/lerobot/local/g1_inspire_pilot_v1 \
  --pilot-manifest derived/pilot-v1/manifest.json \
  --manifest reports/pilot-fit-v1-samples.json \
  --report reports/pilot-fit-v1-baseline.json
python3 scripts/run_gr00t_pilot.py \
  --dataset derived/lerobot/local/g1_inspire_pilot_v1 \
  --pilot-manifest derived/pilot-v1/manifest.json \
  --evaluation-manifest reports/pilot-fit-v1-samples.json \
  --baseline-report reports/pilot-fit-v1-baseline.json \
  --run pilot-fit-v1 --max-steps 512 --accumulation 8 --timeout-seconds 21600
python3 scripts/reload_gr00t_checkpoint.py \
  --checkpoint /dev/shm/vla-groot-smoke/pilot-fit-v1/checkpoint-512 \
  --dataset derived/lerobot/local/g1_inspire_pilot_v1 \
  --report reports/pilot-fit-v1-reload.json
python3 scripts/evaluate_gr00t_pilot.py \
  --checkpoint /dev/shm/vla-groot-smoke/pilot-fit-v1/checkpoint-512 \
  --dataset derived/lerobot/local/g1_inspire_pilot_v1 \
  --pilot-manifest derived/pilot-v1/manifest.json \
  --manifest reports/pilot-fit-v1-samples.json \
  --reference reports/pilot-fit-v1-baseline.json \
  --report reports/pilot-fit-v1-evaluation.json
```

The runner writes an immutable configuration with the implementation commit, source/config/model/data hashes, exact upstream arguments, and bounded run settings. It audits every microbatch loss and every post-clip gradient norm before the optimizer update, aborting on nonfinite values, reserve violations, or the six-hour limit. Evaluation `passed` means finite, complete, comparable evidence; fitting improvement is reported separately. The saved arrays and trajectory plots support inspection of all 100 starts and three paired seeds. [Archive the final checkpoint and comparison evidence](storage.md) after reload and evaluation pass.

## Predeclared changing-target analysis

The original evaluator's `commanded_motion` mask measures target-to-observed-state offset. That offset can remain large while targets are nearly constant, so it is retained with its original meaning. A separate target-change protocol was frozen during training, before candidate predictions or metrics existed. The exact portable declaration is tracked as [evaluation-protocol.json](../experiments/pilot-fit-v1/evaluation-protocol.json); its bytes and SHA-256 match the frozen machine-local declaration.

For each joint, take peak-to-peak ground-truth target variation within the 16-frame horizon and divide by the maximum saved action-normalization span across that horizon. Exclude constant normalization dimensions. A group is selected when its largest joint ratio exceeds 0.05. This ground-truth-only rule selects 75 left-arm, 88 right-arm, 67 left-hand, and 70 right-hand windows out of the same 100 starts. These are changing target commands, not inferred task phases or physical-motion labels. The threshold and selection mask are fixed before candidate comparison.

```bash
python3 scripts/compare_pilot_temporal_change.py \
  --baseline reports/pilot-fit-v1-baseline.json \
  --declaration reports/pilot-fit-v1-temporal-declaration.json
python3 scripts/compare_pilot_temporal_change.py \
  --baseline reports/pilot-fit-v1-baseline.json \
  --declaration reports/pilot-fit-v1-temporal-declaration.json \
  --candidate reports/pilot-fit-v1-evaluation.json \
  --report reports/pilot-fit-v1-temporal-comparison.json
```

The first command freezes or verifies the declaration; the second requires that exact declaration and compares saved prediction arrays without new model inference. It reports candidate, two-step reference, and current-state persistence errors for both changing-target and small-change subsets. Report hashes bind the declaration, frozen sample manifest, and both evaluation reports/NPZ files. Original evaluator reports and predictions remain unchanged.

## Proposed next fitting diagnostic — not launched

The next proposal is a bounded warm start from checkpoint 512 for 1,024 additional updates on the same five episodes, keeping microbatch 1, accumulation 8, seed 42, learning rate 1e-4, warmup 0.05, tuning flags, and augmentation unchanged, while setting state dropout to zero. Because the checkpoint is model-only, this would use a fresh optimizer and scheduler. Estimated training time is approximately 27 minutes.

The proposed acceptance gate requires every group to beat persistence by at least 10% in both native and normalized MAE on the frozen changing-target subsets, with no group worse than checkpoint 512 across all 100 windows. Stop at the update cap regardless; a failed gate calls for diagnosis rather than automatic extension.

Preparing this experiment requires a small runner update: the current runner enforces the 512-update bound and fixes dropout at 0.2. The comparator also needs a narrow allowance for the training-only dropout configuration difference while keeping normalization, joint ordering, horizon, and evaluation sampling identical. No follow-up training has been launched.

Resolve output capacity before that run: only about 16 GiB remains in the current shared-memory filesystem, while another checkpoint and upstream duplicate require about 23.44 GiB and the runner requires 35 GiB free at startup. Configure sufficient output storage while preserving the existing checkpoints; no artifacts were removed for this pilot.

Source inspection found state dropout applied independently during [processor state zeroing](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py#L648) and [action-head feature zeroing](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L220), both training-only. At probability 0.2, approximately 36% of training samples lose state through at least one path. This is a diagnostic hypothesis, not a confirmed cause of the observed error; changing duration and regularization together does not isolate either factor causally. The training loss is masked flow-matching velocity MSE over the active 16×26 outputs, not action MAE. Inspection found no obvious action-order, padding-mask, or normalization mismatch.

## Dataset decisions before longer training

The source copy was still in progress when the pilot was frozen. Confirm completion before making a final inventory. Episode 0136 was excluded from the pilot pending review of a longer frozen-image segment; shorter duplicate runs were recorded as informational findings rather than broadly re-cleaning the dataset.

The [task-label review](dataset-labels.md) records 125 provisional source JSON files with identical original goals and 337 image evidence references; one episode lacks local images, and 0092, 0136, and 0165 remain on review hold. Episode 0141's right/front correction passed snapshot and conversion validation in a new derived copy only. Original recordings and the completed pilot remain preserved. The global spatial reference is still pending, so there is no accepted full-dataset overlay; accepted per-episode overrides require a new snapshot and conversion. Current recordings demonstrate the same target box, and instruction-dependent selection among different targets remains a separate capability requiring suitable demonstrations and evaluation.

Once copying is complete, freeze a content manifest and reserve about 15–20 percent of whole episodes. Prefer collection-session or layout groups when genuine metadata supports them. Copy timestamps do not identify collection sessions; otherwise use a seeded episode split and record that limitation.

Create separate train and validation datasets before normalization. Calculate absolute and relative statistics from training episodes only; use the checkpoint's saved statistics at evaluation. Regenerate relative statistics if the action horizon changes. Report performance on moving segments separately so static tails do not dominate.

## Deployment follows offline validation

The initial deployment candidate is desktop GPU inference over Ethernet, with the Jetson handling observations and local command execution. Size the actual desktop GPU and Jetson variant before implementation. Hardware execution, control timing, command limits, and robot trials are subsequent work, outside this repository milestone.

Pinned references: [fine-tuning configuration](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/gr00t/configs/finetune_config.py), [offline evaluator](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/gr00t/eval/open_loop_eval.py), [statistics implementation](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/gr00t/data/stats.py).
