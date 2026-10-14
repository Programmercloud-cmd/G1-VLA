# Larger-dataset learning curve v1

Status: approved; preparation in progress. No result from this experiment is claimed yet. The completed five-episode fitting tests and their failed persistence criterion remain unchanged.

The [frozen split](split.json) contains 98 training episodes (38,751 frames) and 24 validation episodes (8,811 frames). The five historical pilot episodes stay in training. Episodes 0092, 0136 and 0165 remain on hold. All 122 accepted source JSONs and their 47,562 referenced images passed structural, value, decode and checksum checks. This verifies the selected local contents; completion of the original copying process is unconfirmed. Later arrivals cannot enter this frozen experiment.

No genuine session or layout identifiers were recorded. Seed 42 determines the whole-episode split, stratified by the accepted workspace position. Training has 61 left/front, 36 right/front and one front-only episode; validation has 15, eight and one respectively. This split cannot establish generalization to new sessions or layouts.

The bounded recipe starts from the pinned pretrained GR00T weights with a fresh optimizer and training-only normalization. It uses 8,192 updates, microbatch one, accumulation eight, learning rate 1e-4, warmup 0.05 and zero state dropout in both processor and action head. Four model-only checkpoints are scheduled at 2,048-update intervals. The six-hour training cap includes startup and checkpoint writes; there is no automatic extension. There are 37,281 valid training window starts, so 65,536 consumed examples would equal about 1.76 passes by example count. The runtime audit will separately report actual unique windows and episode coverage.

The [evaluation design](evaluation-design.json) fixes the comparison rules before inference. Separate sets of 100 target-selected windows cover all training and validation episodes. Step zero and each checkpoint use the same training processor, three inference seeds and four denoising steps. Validation group-balanced normalized MAE selects among the four checkpoints; persistence is descriptive. These offline comparisons do not establish successful box selection, grasping or carrying.

Storage is configured locally. Working snapshots and conversions use a new RAM directory, with a durable dataset archive required before training. The external run volume holds all four checkpoints and the pinned trainer's final duplicate model. The output guard requires 80 GiB free, preserves an 8 GiB project reserve, and does not remove previous artifacts. The measured external write rate was about 8.4 MiB/s, making checkpoint writes a substantial part of the six-hour budget.

See the [approved proposal](../../docs/larger-dataset-training-plan-v1.md) for the rationale. Full source inventories, machine paths, generated datasets, predictions and weights remain outside Git; compact manifests and hashes identify them.
