# Zero-state-dropout follow-up results

The 1,024-update follow-up completed, reloaded successfully, and reduced errors in every joint group compared with checkpoint 512. It failed the predeclared fitting gate: all eight full-window checks passed, but all eight changing-target checks against persistence failed. This completed training-set experiment is not a deployment candidate.

The model started from the verified `pilot-fit-v1/checkpoint-512` weights and performed 1,024 **additional** updates: **1,536 cumulative updates** across the two runs. The optimizer and scheduler started fresh, so the new saved checkpoint is named `checkpoint-1024`. Durable archival verified all 62 checkpoint/provenance files against their source SHA-256 hashes, with per-file fsync completed and source outputs preserved. See [artifact references and checksums](artifacts.json).

## Paired prediction errors

The same 100 starts from five training episodes were evaluated with seeds 42/43/44, four denoising steps, and all 16 targets in each action chunk. Native-unit mean absolute errors are shown below; lower is better.

| Group | Two-step reference | Checkpoint 512 | Follow-up | Persistence | Improvement vs 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Left arm, radians | 0.2008 | 0.1782 | 0.1488 | 0.0769 | 16.5% |
| Right arm, radians | 0.1824 | 0.1528 | 0.1305 | 0.0788 | 14.6% |
| Left hand, 0–1 command | 0.1068 | 0.0586 | 0.0488 | 0.0111 | 16.8% |
| Right hand, 0–1 command | 0.1225 | 0.0843 | 0.0725 | 0.0119 | 14.1% |

The [trajectory plot](trajectories.png) shows the most target-varying joint in each episode/group at horizon +15, including the two-step reference and the mean and standard deviation across the follow-up's three seeds. It does not plot checkpoint 512. The table evaluates every joint across the full horizon. Persistence is an offline comparison baseline, not a proposed robot controller.

## Frozen changing-target gate

The [protocol](evaluation-protocol.json) was frozen before training and candidate inference. It reused the original ground-truth-only masks without reselection. Every group had to beat persistence by at least 10% in both native and normalized MAE on its changing-target subset, while never worsening against checkpoint 512 in either metric across all 100 windows. All 16 checks were required.

| Group | Windows | Checkpoint 512 MAE | Follow-up MAE | Persistence MAE |
| --- | ---: | ---: | ---: | ---: |
| Left arm, radians | 75 | 0.1782 | 0.1485 | 0.0837 |
| Right arm, radians | 88 | 0.1520 | 0.1298 | 0.0778 |
| Left hand, 0–1 command | 67 | 0.0590 | 0.0493 | 0.0142 |
| Right hand, 0–1 command | 70 | 0.0846 | 0.0726 | 0.0149 |

| Group | All-window 512 normalized MAE | All-window follow-up normalized MAE | Changing-target follow-up normalized MAE | Changing-target persistence normalized MAE |
| --- | ---: | ---: | ---: | ---: |
| Left arm | 0.6289 | 0.5295 | 0.5290 | 0.3034 |
| Right arm | 0.5838 | 0.5044 | 0.5015 | 0.3103 |
| Left hand | 0.4804 | 0.3945 | 0.4001 | 0.1252 |
| Right hand | 0.5582 | 0.4778 | 0.4779 | 0.1096 |

The follow-up remains worse than persistence in every group and metric. The comparison's `passed=true` means complete, finite, comparable evidence; `acceptance_passed=false` records the failed fitting gate. [summary.json](summary.json) includes all 16 decisions and comparisons with the two-step reference, checkpoint 512, and persistence for all-window, changing-target, and small-change subsets. The bounded run stopped after 1,024 additional updates; no automatic extension was performed.

## Execution and provenance

Training took 1678.3 seconds with sampled peak GPU use of 38,762 MiB. All 8,192 microbatch losses and 1,024 post-clip/pre-update gradient checks were finite. The first and last 160-microbatch mean losses were 0.9833 and 0.5689; this flow-matching velocity loss is not action MAE.

Microbatch 1, accumulation 8, seed 42, learning rate 1e-4, warmup 0.05, zero loader workers, tuning flags, and augmentation retained the original recipe. Exact tuning flags were `tune_llm=false`, `tune_visual=false`, `tune_projector=true`, `tune_diffusion_model=true`, and `tune_vlln=true`; the last flag tunes the action-head feature LayerNorm. Processor and action-head state dropout changed from 0.2 to 0.0. Before the first update, runtime audits verified all 16 initialization files, zero dropout in the training configuration, model configuration, action head and processor, global step 0, empty optimizer state, and scheduler epoch 0.

Fresh-process offline reload returned finite float32 absolute actions shaped `(16, 26)`. Paired evaluation produced 300 action chunks and took 57.3 seconds. The checkpoint is model-only, requires the pinned Cosmos assets, and cannot exactly resume optimizer/scheduler state.

The five-episode dataset, saved statistics, original predictions, sample manifest, and temporal declaration retained their frozen hashes. No raw data, labels, conversion, or normalization statistics changed. Storage preparation verified the original checkpoints and six duplicate root weight shards before sharing their immutable inodes, preserving paths and content while reclaiming 25,152,413,696 bytes. All 96 prelaunch tests passed in 13.656 seconds, and all 22 focused archive-support tests passed in 0.063 seconds; none were skipped.

Implementation commit: `7256f2f7a930ce73bbd3aeaf5d597ab491ef7437`. Frozen protocol SHA-256: `fd5fec3f0db608ad18dc43fc95501a6fdf39a0f006bac6db05fb2f5a8d0b8cc3`. [summary.json](summary.json) binds the reports, test logs, original inputs, source files, and protocol by checksum.

This combined continuation/dropout diagnostic cannot attribute the improvement to either change separately. It provides no held-out generalization, language-grounded choice among target boxes, deployment, or robot-success evidence. See the [execution plan](../../docs/execution-plan.md) for the next decision.
