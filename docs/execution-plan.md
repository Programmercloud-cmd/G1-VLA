# Completed smoke and next bounded experiment

The first milestone is complete: five episodes, both reference conversions, raw-value and image checks, GR00T loading/statistics, two optimizer steps, fresh-process reload, and one checksum-verified durable checkpoint. See the [recorded results](../experiments/pilot-smoke-v2/summary.json), [pilot content identity](../experiments/pilot-smoke-v2/dataset_manifest.json), and [external artifact references](../experiments/pilot-smoke-v2/artifacts.json).

## What is established

The converted state/action arrays match the original float32 values. Camera mapping, task text, timestamps, episode ordering, and 26-dimensional joint ordering were checked. The saved processor returns finite absolute action chunks of shape `(16, 26)`. Vision/language remained frozen while the projector and action head were trainable.

On the validation machine, batch 1 reached 38,608 MiB sampled GPU memory. The 12,583,318,231-byte checkpoint was saved model-only and reloaded offline. Upstream also saved a duplicate model at the run root, making transient output roughly 25.17 GB. A single complete checkpoint was archived; all 33 archived checkpoint/provenance files matched their source hashes.

These checks do not establish pickup competence, closed-loop stability, or generalization. No robot commands have been sent.

## Proposed 512-step fitting probe — not executed

After the portable repository is verified, the next proposed task is a bounded fit/evaluation experiment on the same five pilot episodes:

| Setting | Proposed value |
| --- | --- |
| Microbatch | 1 |
| Gradient accumulation | 8, giving effective batch 8 |
| Loader workers | 0 |
| Optimizer updates | At most 512 |
| Checkpoints | One model-only save at step 512 |
| Shard size | 256 |
| Episode sampling rate | 1.0 |
| Shards per epoch | 5, covering the five whole-episode shards |

Accumulation 8 remains unvalidated. Measure memory during initial updates before continuing. Do not assume batch 4 fits. The completed smoke uses accumulation 1 and is implemented separately; no longer experiment is launched by setup or readiness verification.

Compare the two-step checkpoint and candidate on the same 100 pilot starts, full 16-frame targets, and paired inference seeds 42/43/44. Report arm and hand errors separately and compare against a current-state persistence baseline. Inspect approach, grasp, lift, and hold trajectories for collapsed constant predictions. This is a training-set fitting measurement, not held-out evaluation. Extend the run only when the learning trend justifies it.

## Dataset decisions before longer training

The source copy was still in progress when the pilot was frozen. Confirm completion before making a final inventory. Episode 0136 was excluded from the pilot pending review of a longer frozen-image segment; shorter duplicate runs were recorded as informational findings rather than broadly re-cleaning the dataset.

Preserve original task labels in conversion. Current recordings demonstrate the same target box; the instruction's left/front wording needs checking against varied positions. Do not silently invent replacement labels. Instruction-dependent selection among different targets is a separate capability requiring suitable demonstrations and evaluation.

Once copying is complete, freeze a content manifest and reserve about 15–20 percent of whole episodes. Prefer collection-session or layout groups when genuine metadata supports them. Copy timestamps do not identify collection sessions; otherwise use a seeded episode split and record that limitation.

Create separate train and validation datasets before normalization. Calculate absolute and relative statistics from training episodes only; use the checkpoint's saved statistics at evaluation. Regenerate relative statistics if the action horizon changes. Report performance on moving segments separately so static tails do not dominate.

## Deployment follows offline validation

The initial deployment candidate is desktop GPU inference over Ethernet, with the Jetson handling observations and local command execution. Size the actual desktop GPU and Jetson variant before implementation. Hardware execution, control timing, command limits, and robot trials are subsequent work, outside this repository milestone.

Pinned references: [fine-tuning configuration](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/gr00t/configs/finetune_config.py), [offline evaluator](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/gr00t/eval/open_loop_eval.py), [statistics implementation](https://github.com/NVIDIA/Isaac-GR00T/blob/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/gr00t/data/stats.py).
