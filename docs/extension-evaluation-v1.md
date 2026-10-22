# Evaluation of the extension from 8,192 to 12,288

The new declaration is `experiments/larger-dataset-v1-extension-v1/evaluation-protocol.json`. It preserves the original 21 runtime sources plus the previous continuation's six sources, original 100 frozen windows per split, normalization, evaluation seeds 42/43/44, four denoising steps and zero state dropout. The original evaluator and its tested OpenCV-preloading launcher run unchanged. New records compare each new policy both to the original pretrained baseline and to the actual independently verified 8,192 predictions.

Native checkpoint names and TrainerState remain local: `checkpoint-2048` maps to cumulative 10,240, and `checkpoint-4096` maps to cumulative 12,288. The learned 8,192 source retains its canonical native checkpoint 4,096 and complete optimizer/scheduler/RNG files. This extension deliberately loads its model weights with a fresh AdamW optimizer and a cosine schedule for 4,096 additional updates: peak 2.5e-5, 103 warmup updates, native seed 8,234. It does not restore the previous optimizer or an exact iterable cursor. Source lineage remains two separately certified native 4,096 segments, with 65,536 consumed examples; the old unsaved 1,135 updates remain excluded.

After training exits and a stable final report exists, evaluate every available certified new checkpoint under a fresh GPU lease. A passed finite prefix and full save-time SHA/fsync inventory are required even if the overall training run stopped at its cap or a reserve guard. The synchronous training save callback certifies all model/optimizer/scheduler/RNG files before another update. Do not evaluate incomplete checkpoints or claim absent checkpoints were evaluated.

These helpers currently require configured **SSH-backed storage**. Local-disk-only full-state verification, inference views and archival are unsupported future portability work. Hardlink inference views share the certified canonical model/processor/configuration files; `--method verified_copy` is a verified server-local copy fallback on that SSH host. Both are explicitly **NONRESUMABLE**, excluding optimizer/scheduler/RNG. The canonical full checkpoint remains the resume reference; exact iterable sample-position restoration is unsupported.

Use the locked GR00T Python for every outer helper and CPU array check. Set `TMPDIR` to an existing readiness-verified RAM directory outside frozen datasets. The following paths describe this machine; the evaluation configuration keeps all new reports and logs on the configured external storage while training evidence stays local.

```bash
set -o noclobber
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
export PYTHONHASHSEED=1
export TMPDIR=/dev/shm/vla-extension-tmp-20261022
VLA_CFG=reports/larger-dataset-v1-machine.json
VLA_DECL=experiments/larger-dataset-v1-extension-v1/evaluation-protocol.json
VLA_RUN=larger-dataset-v1-extension-v1
VLA_LOCAL_REPORTS=reports/larger-dataset-v1-extension-v1
VLA_REPORTS=storage/reports/larger-dataset-v1
VLA_LOCAL_STEP=2048
VLA_CUMULATIVE_STEP=10240
VLA_PREFIX=$VLA_RUN-local-$VLA_LOCAL_STEP-cumulative-$VLA_CUMULATIVE_STEP

"$VLA_PY" -B scripts/archive_gr00t_extension.py --config "$VLA_CFG" view \
  --checkpoint "storage/runs/$VLA_RUN/checkpoint-$VLA_LOCAL_STEP" \
  --full-inventory "$VLA_LOCAL_REPORTS/$VLA_RUN-coverage-checkpoint-$VLA_LOCAL_STEP-inventory.json" \
  --run-manifest "$VLA_LOCAL_REPORTS/$VLA_RUN-run.json" \
  --train-report "$VLA_LOCAL_REPORTS/$VLA_RUN-train.json" \
  --view "storage/runs/$VLA_RUN-inference/checkpoint-$VLA_LOCAL_STEP" \
  --view-manifest "$VLA_REPORTS/$VLA_PREFIX-view.json" \
  --view-inventory "$VLA_REPORTS/$VLA_PREFIX-view-inventory.json"

"$VLA_PY" -B scripts/evaluate_gr00t_extension.py --config "$VLA_CFG" evaluate \
  --extension-protocol "$VLA_DECL" \
  --view-manifest "$VLA_REPORTS/$VLA_PREFIX-view.json" \
  --view-inventory "$VLA_REPORTS/$VLA_PREFIX-view-inventory.json" \
  --train-report "$VLA_REPORTS/$VLA_PREFIX-train.json" \
  --validation-report "$VLA_REPORTS/$VLA_PREFIX-validation.json" \
  --output "$VLA_REPORTS/$VLA_PREFIX-comparison.json" \
  > "$VLA_REPORTS/$VLA_PREFIX-evaluation.log" 2>&1

"$VLA_PY" -B scripts/archive_gr00t_extension.py --config "$VLA_CFG" archive \
  --extension-protocol "$VLA_DECL" \
  --comparison "$VLA_REPORTS/$VLA_PREFIX-comparison.json" \
  --record-name "$VLA_PREFIX"
```

Use local 4,096/cumulative 12,288 for the other checkpoint, if certified. All output paths must be fresh, including logs; retain failed attempts and choose distinct new attempt names rather than overwrite them. Loading each 12.6 GB model through the current SSHFS link can take about 24 minutes, followed by a short paired inference phase. Keep the original 8 GiB filesystem reserve and other resource guards. Archive records preserve small hashed evidence and reference the complete canonical checkpoint in place; no weights or optimizer are duplicated by the archive command.

The new rule always selects the minimum validation mean of four group normalized MAEs among the source 8,192 and available certified new pairs. An exact tie chooses the earliest cumulative step. Therefore 8,192 remains selected if neither new policy improves on it. A cap may leave one or both planned checkpoints missing: return a real best available policy, retain the missing-step list and `selection_status=partial_best_available`, and keep `planned_comparisons_complete=false` and `formal_extension_selection_complete=false`. Missing points must not be interpolated or presented as completed evaluations.

```bash
VLA_COMPARISONS=(
  "$VLA_REPORTS/$VLA_RUN-local-2048-cumulative-10240-comparison.json"
  "$VLA_REPORTS/$VLA_RUN-local-4096-cumulative-12288-comparison.json"
)
# Include only the certified, completed paired comparison files above.
# Use VLA_COMPARISONS=() if no new checkpoint could be certified.
"$VLA_PY" -B scripts/evaluate_gr00t_extension.py --config "$VLA_CFG" select \
  --extension-protocol "$VLA_DECL" \
  --comparisons "${VLA_COMPARISONS[@]}" \
  --output "$VLA_REPORTS/$VLA_RUN-selection.json"

"$VLA_PY" -B -m unittest discover -s tests -p test_extension_evaluation.py -v
```

Selection uses the same held-out validation windows and is not an untouched final test. Persistence and improvement thresholds remain descriptive; no automatic hardware deployment follows from offline scores. Preserve all previous declarations, raw predictions, selected policies, full checkpoints and result histories. New-checkpoint full saves/views/inference are pending until this extension produces actual checkpoints; native full-state reload has only the previously recorded tiny CPU Trainer fixture as a reload test.
