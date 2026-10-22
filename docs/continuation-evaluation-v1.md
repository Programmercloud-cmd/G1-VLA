# Continuation evaluation and full-state preservation

This additive experiment starts from the certified original checkpoint4096 with a fresh optimizer and a new 4096-update schedule. Native checkpoint2048 represents cumulative weight lineage6144; native checkpoint4096 represents cumulative lineage8192. TrainerState and raw evaluator reports retain honest local steps. The original frozen protocol, samples, processor, training statistics, evaluator and four-checkpoint selector remain unchanged.

The active declaration is `experiments/larger-dataset-v1-continuation-v1/evaluation-protocol-startup-fix-v2.json`; the first `evaluation-protocol.json` remains preserved as historical startup-attempt evidence. The active declaration binds the original protocol, original step-zero predictions, actual source4096 predictions and durable archive, the new implementation, full-state saving requirements, and the separate selection rule. Both new checkpoints use fresh paired train/validation inference on the original100 windows per split, seeds42/43/44, four denoising steps and zero state dropout. Each comparison retains the original evaluator's step-zero deltas and adds native/normalized group and per-episode changes versus actual source4096 predictions.

Training saves canonical full Hugging Face checkpoints, including optimizer, scheduler and RNG. Its synchronous save callback validates state structure, hashes/fsyncs every file on the storage host, fsyncs checkpoint directories, and only then permits another update. Native full-state optimizer/scheduler/RNG reload is supported; exact iterable dataset cursor restoration is **not** supported. The initial historical source4096 had model weights only, so this run must not be described as an exact optimizer or sample-cursor resume.

After training has exited and its final report exists, create a separate model-only inference view for each certified checkpoint. A view contains hardlinks to the canonical model, processor, configuration, TrainerState and continuation metadata; it excludes optimizer/scheduler/RNG and is explicitly **not resumable**. Canonical full checkpoints remain the resume references. A receipt outside the view binds its complete inventory to the full save-time inventory and verifies file hashes and shared device/inode. If hardlinks are unavailable, `--method verified_copy` performs a server-local copy with independent complete hash verification. Neither mode changes canonical checkpoint bytes. Views prevent the unchanged original evaluator's metadata checks from repeatedly reading large optimizer files through SSHFS.

The current continuation full-state verification, inference-view and archive helpers require configured SSH-backed storage, including storage-host SHA/fsync certification; another SSH storage host can replace this machine's host. The base pipeline and durable training paths support local disk, but this continuation view/evaluation/archive workflow has no supported or validated local-disk-only path. Adding that path is future portability work. `--method verified_copy` remains a server-local copy on the configured SSH host; it does not provide a local-disk-only fallback.

The following uses this machine's configured SSH storage and active retry1 run. The training configuration keeps its reports in the original local parent directory; evaluation uses the original external report configuration. Use the pinned GR00T interpreter for the outer helpers and CPU tests: their array verification requires NumPy from the locked environment. Paths are passed explicitly and all outputs must be absent. Obtain a fresh GPU lease and resource readiness check before inference. Do not evaluate or archive while training still owns the GPU or before the final training report exists. No further training is launched by these commands.

```bash
export PYTHONHASHSEED=1
export TMPDIR=/dev/shm/vla-continuation-tmp-20261022
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_EVAL_CFG=reports/larger-dataset-v1-machine.json
VLA_DECL=experiments/larger-dataset-v1-continuation-v1/evaluation-protocol-startup-fix-v2.json
VLA_RUN=larger-dataset-v1-continuation-v1-retry1
VLA_LOCAL_REPORTS=reports/larger-dataset-v1-continuation-v1
VLA_EXTERNAL_REPORTS=storage/reports/larger-dataset-v1
VLA_LOCAL_STEP=2048
VLA_CUMULATIVE_STEP=6144
VLA_PREFIX=$VLA_RUN-local-$VLA_LOCAL_STEP-cumulative-$VLA_CUMULATIVE_STEP
VLA_INVENTORY=$VLA_LOCAL_REPORTS/$VLA_RUN-coverage-checkpoint-$VLA_LOCAL_STEP-inventory.json

"$VLA_PY" -B scripts/archive_gr00t_continuation.py --config "$VLA_EVAL_CFG" view \
  --checkpoint "storage/runs/$VLA_RUN/checkpoint-$VLA_LOCAL_STEP" \
  --full-inventory "$VLA_INVENTORY" \
  --run-manifest "$VLA_LOCAL_REPORTS/$VLA_RUN-run.json" \
  --train-report "$VLA_LOCAL_REPORTS/$VLA_RUN-train.json" \
  --view "storage/runs/$VLA_RUN-inference/checkpoint-$VLA_LOCAL_STEP" \
  --view-manifest "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-view.json" \
  --view-inventory "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-view-inventory.json"

"$VLA_PY" -B scripts/evaluate_gr00t_continuation.py --config "$VLA_EVAL_CFG" evaluate \
  --continuation-protocol "$VLA_DECL" \
  --view-manifest "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-view.json" \
  --view-inventory "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-view-inventory.json" \
  --train-report "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-train.json" \
  --validation-report "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-validation.json" \
  --output "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-comparison.json" \
  > "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-evaluation.log" 2>&1

"$VLA_PY" -B scripts/archive_gr00t_continuation.py --config "$VLA_EVAL_CFG" archive \
  --continuation-protocol "$VLA_DECL" \
  --comparison "$VLA_EXTERNAL_REPORTS/$VLA_PREFIX-comparison.json" \
  --record-name "$VLA_PREFIX"
```

Run the same explicit commands for local4096/cumulative8192 only after its full checkpoint is certified. Existing output paths are rejected; retain failure artifacts and choose a fresh attempt identity instead of overwriting them. Loading model weights over the current SSHFS link can take tens of minutes per checkpoint; train and validation share one strict model load. The original8GiB filesystem reserve and other runtime guards still apply. `TMPDIR` is a process-local setting outside the frozen dataset, not a protocol or source change.

The new archive publishes small evidence with SHA/fsync verification and references the canonical full checkpoint in place. It does not duplicate weights/optimizer or turn the view into a resume checkpoint. Preserve both canonical checkpoint files and archive records. A checkpoint from a bounded interrupted run may be evaluated/archived if its complete finite prefix, coverage, save-time inventory and final training report bindings pass; the whole run is still reported as interrupted.

When both new pairs exist, run the separate continuation ranking:

```bash
"$VLA_PY" -B scripts/evaluate_gr00t_continuation.py --config "$VLA_EVAL_CFG" select \
  --continuation-protocol "$VLA_DECL" \
  --comparisons \
    "$VLA_EXTERNAL_REPORTS/$VLA_RUN-local-2048-cumulative-6144-comparison.json" \
    "$VLA_EXTERNAL_REPORTS/$VLA_RUN-local-4096-cumulative-8192-comparison.json" \
  --output "$VLA_EXTERNAL_REPORTS/$VLA_RUN-selection.json"
```

This selects the minimum validation mean of four group normalized MAEs across cumulative4096,6144,8192; an exact tie chooses the earlier cumulative step. The source4096 can remain best. Missing new pairs produce descriptive ranking only, with no formal selected checkpoint. There is no mandatory improvement or persistence threshold, and offline accuracy does not establish box selection, grasp or carrying competence on hardware.

CPU checks:

```bash
"$VLA_PY" -B -m unittest discover -s tests -p test_continuation_evaluation.py -v
```

These validate tiny hardlink/copy views, complete-file identity, full-state exclusions,
bounded optimizer verification, fixed protocol/source bindings and early rejection
of uncertified training. Production full-state saving and certification now passed
at both native steps 2,048 and 4,096. Both separate inference views, strict paired
train/validation inferences, comparisons and durable archives passed; formal selection
chose cumulative 8,192, confirmed by independent CPU recomputation of all four new
prediction arrays. See [final results](larger-dataset-v1-continuation-results.md).

Production optimizer reload has not been exercised. The separate native Trainer
CPU fixture covers optimizer/scheduler/RNG save/reload and demonstrates the pinned
iterable-loader reseeding limitation; exact sample-cursor replay remains unsupported.
The cap test uses disposable processes, not a live model or storage failure.
The SSH-backed continuation storage requirement described above remains in force;
a local-disk-only continuation view/evaluation/archive path is unsupported.
