# Continuation evaluation and full-state preservation

This additive experiment starts from the certified original checkpoint4096 with a fresh optimizer and a new 4096-update schedule. Native checkpoint2048 represents cumulative weight lineage6144; native checkpoint4096 represents cumulative lineage8192. TrainerState and raw evaluator reports retain honest local steps. The original frozen protocol, samples, processor, training statistics, evaluator and four-checkpoint selector remain unchanged.

The declaration is `experiments/larger-dataset-v1-continuation-v1/evaluation-protocol.json`. It binds the original protocol, original step-zero predictions, actual source4096 predictions and durable archive, the new implementation, full-state saving requirements, and the separate selection rule. Freeze it only after the new implementation is finalized. Both new checkpoints use fresh paired train/validation inference on the original100 windows per split, seeds42/43/44, four denoising steps and zero state dropout. Each comparison retains the original evaluator's step-zero deltas and adds native/normalized group and per-episode changes versus actual source4096 predictions.

Training saves canonical full Hugging Face checkpoints, including optimizer, scheduler and RNG. Its synchronous save callback validates state structure, hashes/fsyncs every file on the storage host, fsyncs checkpoint directories, and only then permits another update. Native full-state optimizer/scheduler/RNG reload is supported; exact iterable dataset cursor restoration is **not** supported. The initial historical source4096 had model weights only, so this run must not be described as an exact optimizer or sample-cursor resume.

After training has exited and its final report exists, create a separate model-only inference view for each certified checkpoint. A view contains hardlinks to the canonical model, processor, configuration, TrainerState and continuation metadata; it excludes optimizer/scheduler/RNG and is explicitly **not resumable**. Canonical full checkpoints remain the resume references. A receipt outside the view binds its complete inventory to the full save-time inventory and verifies file hashes and shared device/inode. If hardlinks are unavailable, `--method verified_copy` performs a server-local copy with independent complete hash verification. Neither mode changes canonical checkpoint bytes. Views prevent the unchanged original evaluator's metadata checks from repeatedly reading large optimizer files through SSHFS.

The following uses this machine's optional external-storage configuration. The training configuration must keep its reports local; evaluation uses the original external report configuration. Paths are passed explicitly and all outputs must be absent. Obtain a fresh GPU lease and resource readiness check before inference. Do not evaluate or archive while training still owns the GPU or before the final training report exists. No further training is launched by these commands.

```bash
export PYTHONHASHSEED=1
export TMPDIR=/dev/shm/vla-larger-evaluation-tmp-20261021
VLA_EVAL_CFG=reports/larger-dataset-v1-machine.json
VLA_DECL=experiments/larger-dataset-v1-continuation-v1/evaluation-protocol.json
VLA_RUN=larger-dataset-v1-continuation-v1
VLA_LOCAL_REPORTS=reports/$VLA_RUN
VLA_EXTERNAL_REPORTS=storage/reports/larger-dataset-v1
VLA_LOCAL_STEP=2048
VLA_CUMULATIVE_STEP=6144
VLA_PREFIX=$VLA_RUN-local-$VLA_LOCAL_STEP-cumulative-$VLA_CUMULATIVE_STEP
VLA_INVENTORY=$VLA_LOCAL_REPORTS/$VLA_RUN-coverage-checkpoint-$VLA_LOCAL_STEP-inventory.json

python3 -B scripts/archive_gr00t_continuation.py --config "$VLA_EVAL_CFG" view \
  --checkpoint "storage/runs/$VLA_RUN/checkpoint-$VLA_LOCAL_STEP" \
  --full-inventory "$VLA_INVENTORY" \
  --run-manifest "$VLA_LOCAL_REPORTS/$VLA_RUN-run.json" \
  --train-report "$VLA_LOCAL_REPORTS/$VLA_RUN-train.json" \
  --view "storage/runs/$VLA_RUN-inference/checkpoint-$VLA_LOCAL_STEP" \
  --view-manifest "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-view.json" \
  --view-inventory "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-view-inventory.json"

python3 -B scripts/evaluate_gr00t_continuation.py --config "$VLA_EVAL_CFG" evaluate \
  --continuation-protocol "$VLA_DECL" \
  --view-manifest "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-view.json" \
  --view-inventory "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-view-inventory.json" \
  --train-report "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-train.json" \
  --validation-report "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-validation.json" \
  --output "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-comparison.json" \
  > "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-evaluation.log" 2>&1

python3 -B scripts/archive_gr00t_continuation.py --config "$VLA_EVAL_CFG" archive \
  --continuation-protocol "$VLA_DECL" \
  --comparison "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-comparison.json" \
  --record-name "$VLA_PREFIX"
```

Run the same explicit commands for local4096/cumulative8192 only after its full checkpoint is certified. Existing output paths are rejected; retain failure artifacts and choose a fresh attempt identity instead of overwriting them. Loading model weights over the current SSHFS link can take tens of minutes per checkpoint; train and validation share one strict model load. The original8GiB filesystem reserve and other runtime guards still apply. `TMPDIR` is a process-local setting outside the frozen dataset, not a protocol or source change.

The new archive publishes small evidence with SHA/fsync verification and references the canonical full checkpoint in place. It does not duplicate weights/optimizer or turn the view into a resume checkpoint. Preserve both canonical checkpoint files and archive records. A checkpoint from a bounded interrupted run may be evaluated/archived if its complete finite prefix, coverage, save-time inventory and final training report bindings pass; the whole run is still reported as interrupted.

When both new pairs exist, run the separate continuation ranking:

```bash
python3 -B scripts/evaluate_gr00t_continuation.py --config "$VLA_EVAL_CFG" select \
  --continuation-protocol "$VLA_DECL" \
  --comparisons \
    "$VLA_EXTERNAL_REPORTS/$VLA_RUN-local-2048-cumulative-6144-comparison.json" \
    "$VLA_EXTERNAL_REPORTS/$VLA_RUN-local-4096-cumulative-8192-comparison.json" \
  --output "$VLA_EXTERNAL_REPORTS/$VLA_RUN-selection.json"
```

This selects the minimum validation mean of four group normalized MAEs across cumulative4096,6144,8192; an exact tie chooses the earlier cumulative step. The source4096 can remain best. Missing new pairs produce descriptive ranking only, with no formal selected checkpoint. There is no mandatory improvement or persistence threshold, and offline accuracy does not establish box selection, grasp or carrying competence on hardware.

CPU checks:

```bash
python3 -B -m unittest discover -s tests -p test_continuation_evaluation.py -v
```

These validate tiny hardlink/copy views, complete-file identity, full-state exclusions, bounded optimizer verification, fixed protocol/source bindings and early rejection of uncertified training. Production full-state saving, views and new-checkpoint inference remain untested until the authorized continuation produces its actual checkpoints. The separate native Trainer CPU test covers optimizer/scheduler/RNG save/reload and demonstrates the iterable-loader reseeding limitation; the cap test uses disposable processes, not a live model or storage failure.
