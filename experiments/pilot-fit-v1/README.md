# Pilot-fit-v1 results

The 512-update pilot completed and its checkpoint reloaded successfully. Errors improved over the two-step reference in all four joint groups, but remained worse than holding the observed current state. This completed training-set experiment is not a deployment candidate.

Durable archival verified: all 56 checkpoint/provenance files matched their source hashes and completed file fsync. The saved comparison was replayed successfully from archived JSON/NPZ files on CPU. See [artifact references and checksums](artifacts.json).

## Paired prediction errors

Both checkpoints used the same 100 starts from five training episodes, seeds 42/43/44, four denoising steps, and all 16 targets in each action chunk. These are native-unit mean absolute errors; lower is better.

| Group | Two-step reference | 512-step candidate | Persistence | Improvement vs reference |
| --- | ---: | ---: | ---: | ---: |
| Left arm, radians | 0.2008 | 0.1782 | 0.0769 | 11.3% |
| Right arm, radians | 0.1824 | 0.1528 | 0.0788 | 16.2% |
| Left hand, 0–1 command | 0.1068 | 0.0586 | 0.0111 | 45.1% |
| Right hand, 0–1 command | 0.1225 | 0.0843 | 0.0119 | 31.2% |

Persistence is an offline comparison baseline, not a proposed robot controller. The [trajectory plot](trajectories.png) shows the most target-varying joint in each episode/group at horizon +15, with mean and standard deviation across the three paired seeds. The table evaluates every joint across the full horizon. Full native and normalized metrics are in [summary.json](summary.json).

## Predeclared changing-target subsets

The separately committed [protocol](evaluation-protocol.json) measures ground-truth target variation within the horizon. It was frozen before candidate outputs existed and uses no prediction-based selection. The original evaluator's target-to-observed-state offset mask remains unchanged and measures something different.

| Group | Windows | Reference MAE | Candidate MAE | Persistence MAE |
| --- | ---: | ---: | ---: | ---: |
| Left arm, radians | 75 | 0.2025 | 0.1782 | 0.0837 |
| Right arm, radians | 88 | 0.1833 | 0.1520 | 0.0778 |
| Left hand, 0–1 command | 67 | 0.1066 | 0.0590 | 0.0142 |
| Right hand, 0–1 command | 70 | 0.1227 | 0.0846 | 0.0149 |

The persistence gap remains in every changing-target group. These labels identify target variation, not task phases or measured robot motion.

## Execution and provenance

Training used microbatch 1, accumulation 8, zero loader workers, and 512 optimizer updates, starting from the pinned pretrained model. All 4,096 microbatch losses and 512 post-clip/pre-update gradient checks were finite. Training took 865.1 seconds; sampled peak GPU usage was 38,764 MiB. The first and last 160-microbatch mean losses were 1.1923 and 0.9927.

Exact tuning flags were `tune_llm=false`, `tune_visual=false`, `tune_projector=true`, `tune_diffusion_model=true`, and `tune_vlln=true`. The last flag controls the action-head feature LayerNorm; vision/language backbones remained frozen.

Fresh-process reload returned finite float32 absolute actions shaped `(16, 26)`. The 12,583,325,249-byte checkpoint is model-only and requires the pinned Cosmos assets. It cannot resume optimizer/scheduler state exactly.

Training implementation: `4685d0dd04611cce5f1dbfbab4ef2e40caf7f67d`. Predeclared temporal protocol: `8caaf9fbf88ee5e9da2bbe3012680b0b98578132`. Protocol SHA-256: `cd8cd85ae3b18e2d258a7baafce62bcbc4e188f9d8d36a64364cbec9dff004d3`.

The baseline report records the preceding repository HEAD because it ran before the implementation commit. All four recorded baseline helper hashes exactly match the committed implementation. The reload helper's metadata-only update is recorded by its exact source hash and included in the results commit.

All 65 focused lightweight tests passed, including real CPU Trainer fixtures; no tests were skipped. See [summary.json](summary.json) for source and evidence hashes, and [the execution plan](../../docs/execution-plan.md) for the proposed next diagnostic. No held-out evaluation or robot trial was performed. Demonstrations concern the same target box; instruction-dependent choice among different boxes remains untested.
