# Bounded pilot interval review

Retain all ten reviewed intervals. This review found no systematic defect that justifies trimming, shifting, smoothing, repairing, or excluding them. That finding applies to the bounded evidence below; it does not certify every recording. Recorded commands remain the training targets, and no dataset, statistics, configuration, checkpoint, or training run was changed.

The earlier persistence fitting gate remains a preserved diagnostic result. Passing that threshold is not an upstream GR00T requirement or a prerequisite for further work. Its failure does not by itself establish corrupted demonstrations or unsuccessful box handling.

## Selection and visual evidence

Selection reused the 100 saved evaluation windows, with no new inference: the highest equal-group normalized MAE window in each episode and the lower median of the remaining non-overlapping evaluated windows. Each window contains 16 target rows; context extends by 30 rows on either side, clipped at the episode boundary. Selection used prediction error, not visual phase or suspected data quality. These ten windows are a targeted diagnostic sample, not an estimate of defect prevalence.

Visual inspection covered 12 explicitly recorded stills per context and one 12-frame overview per episode: **180 frame placements, 178 unique source frames, and zero extra frames**. No continuous video playback or exhaustive visual inspection was performed. The [visual findings](../experiments/pilot-interval-review-v1/visual-findings.json) list every viewed raw row, nominal time, contact-sheet hash, and source-image hash references. Contact sheets remain in ignored local `reports/`; the links below require that local evidence. The 700-row context hash inventory is quantitative evidence, not a count of visually reviewed frames.

Raw row and source `idx` are equal in these five episodes. Seconds below are nominal row/30 values, not measured capture times. “Side hold,” “lift,” and “carry-like” describe visible postures and motion across sampled stills; they do not establish grip force, exact phase boundaries, physical success, or closed-loop stability.

| Episode / selection | Evaluated raw rows (nominal time) | Context rows | Sampled visual evidence | Equal-group normalized MAE | Decision |
| --- | --- | --- | --- | ---: | --- |
| 0037 / highest error | 40–55 (1.33–1.83 s) | 10–85 | [Open-hand setup/approach](../reports/pilot-interval-review-v1/episode_0037-worst-sampled.jpg) | 0.551 | Retain |
| 0037 / median comparison | 141–156 (4.70–5.20 s) | 111–186 | [Side placement; pre-lift hold](../reports/pilot-interval-review-v1/episode_0037-median-sampled.jpg) | 0.472 | Retain |
| 0040 / highest error | 424–439 (14.13–14.63 s) | 394–469 | [Elevated hold/carry-like posture](../reports/pilot-interval-review-v1/episode_0040-worst-sampled.jpg) | 0.513 | Retain |
| 0040 / median comparison | 274–289 (9.13–9.63 s) | 244–319 | [Side hold; lift later in context](../reports/pilot-interval-review-v1/episode_0040-median-sampled.jpg) | 0.473 | Retain |
| 0042 / highest error | 430–445 (14.33–14.83 s) | 400–475 | [Elevated hold with reorientation](../reports/pilot-interval-review-v1/episode_0042-worst-sampled.jpg) | 0.544 | Retain |
| 0042 / median comparison | 278–293 (9.27–9.77 s) | 248–323 | [Side hold; lift later in context](../reports/pilot-interval-review-v1/episode_0042-median-sampled.jpg) | 0.466 | Retain |
| 0046 / highest error | 329–344 (10.97–11.47 s) | 299–344 | [Late elevated hold](../reports/pilot-interval-review-v1/episode_0046-worst-sampled.jpg) | 0.558 | Retain |
| 0046 / median comparison | 190–205 (6.33–6.83 s) | 160–235 | [Side hold transitioning to lift](../reports/pilot-interval-review-v1/episode_0046-median-sampled.jpg) | 0.464 | Retain |
| 0047 / highest error | 0–15 (0.00–0.50 s) | 0–45 | [Initial open-hand setup](../reports/pilot-interval-review-v1/episode_0047-worst-sampled.jpg) | 0.523 | Retain |
| 0047 / median comparison | 135–150 (4.50–5.00 s) | 105–180 | [Side placement; pre-lift hold](../reports/pilot-interval-review-v1/episode_0047-median-sampled.jpg) | 0.462 | Retain |

All five overviews show open-hand approach, bilateral side placement, subsequent lifting, and later elevated held-box postures. The quiet tails in 0037, 0046, and 0047 still show a box held between the hands. Nothing in the permitted sample supports discarding those tails as irrelevant idle. No sampled scene reset, missing target, or visible discontinuity supplied a repair or exclusion rationale. Events between sampled frames remain unobserved.

## Numerical and timing checks

The [quantitative summary](../experiments/pilot-interval-review-v1/summary.json) and [CPU replay](../experiments/pilot-interval-review-v1/replay.py) bind the source arrays, selection and metrics. The equal-group interval score improves by 5.4–19.4% against checkpoint 512 in all ten intervals. This average hides a local exception: in 0037 rows 40–55, left-hand normalized MAE increases from 0.595 to 0.647 and right-hand MAE from 0.552 to 0.579. The images place this interval in open-hand setup/approach. Command and measured-state variation differ there, but that does not identify a bad target or demonstrate latency.

The [timing and integrity summary](../experiments/pilot-interval-review-v1/timing-summary.json) checks the entire five-episode pilot numerically: 2,121 original JSON/JPEG hashes match the frozen pilot; all 2,116 source images and converted video frames decode; state/action values and row ordering match through both existing conversion formats. No nonfinite values or flags under the documented heuristic adjacent-step screen were found. Those screens are not hardware limits or proof that no subtler defect exists.

There are 216 repeated-image transitions among 2,111 adjacent pairs (10.23%), with a longest identical run of five stored frames. Its first-to-last nominal span is 0.133 seconds; its physical capture duration and cause are unknown. These short repetitions are recorded as quality observations, with row alignment preserved. A strict 31-row low-motion screen covers only 54 rows (2.55%), all in 0037's held-box tail; a relaxed screen covers 133 rows (6.29%). Neither numerical quietness nor repeated images establishes disposable idle.

The raw recordings provide a date, nominal frame rate, row indices and filenames, but no independent camera-capture, state-acquisition, command-issue, or actuator-execution clocks. Converted timestamps are synthesized from row order. Their numerical agreement cannot exclude hidden asynchronous latency. No timing shift is justified by the available evidence, and measured state must not replace the recorded command targets. Future collection should log those monotonic clocks and their clock domains so alignment can be measured directly.

## Scoped decision and next step

The ten decisions are **retain**: zero trims, repairs or exclusions. The full-dataset holds for **0092, 0136 and 0165 remain unchanged**; these episodes were not revalidated by this five-episode review. The accepted workspace labels and all frozen pilot artifacts remain preserved.

The [larger-dataset training plan](larger-dataset-training-plan-v1.md) proposes the next bounded experiment with explicit split, normalization, implementation and storage work. It is a proposal; this review created no larger dataset and launched no training. The next decision should use that plan and the task-level evidence it proposes, without turning the earlier diagnostic threshold into a universal requirement.
