# Larger-dataset v1 session handoff

The user deferred the long run to the next session. **Do not start GPU evaluation or training automatically.** When the user asks to resume, check the saved state and current resources, then continue the already approved workflow. There is no step-zero baseline, larger-dataset checkpoint, or training result yet. The pending run is 8,192 updates with a six-hour cap, microbatch one, accumulation eight, pretrained initialization, fresh optimizer, and zero state dropout in both model and processor.

The implementation was committed at `1382dec`; use its frozen helpers and pinned submodules. The [split](../experiments/larger-dataset-v1/split.json) remains SHA256 `9a721d4bb14157a119650b4848c62e0cdd6134a843dc0e6d27b84d36e20e040b`: 98 training episodes / 38,751 frames and 24 validation episodes / 8,811 frames. The five historical pilot episodes stay in training; 0092, 0136 and 0165 remain excluded. [Copy completion](../experiments/larger-dataset-v1/copy-completion.json) is recorded separately without changing this split. Original recordings and all completed pilot artifacts remain preserved.

**CPU preparation and durable archival are complete.** Both splits passed exact row/state/action/task checks and final video checks, with statistics computed only from training data. The frozen protocol has 100 windows per split covering all episodes. The archive passed all 48,204 member hashes and fsync checks: 5,276,113,118 uncompressed bytes, stored as 4,986,798,573 compressed bytes. The [session closeout](../experiments/larger-dataset-v1/session-closeout.json) binds the reports, protocol and archive hashes. A final actual-host check found no preparation/evaluation/training or GPU compute processes. Existing complete v3 files and the failed partial are preserved; do not rerun preparation over occupied outputs. The original [path incident](../experiments/larger-dataset-v1/conversion-path-incident.json) remains a historical record.

Run from `/home/test2/dev/VLA`. The ignored machine configuration is `reports/larger-dataset-v1-machine.json`, with `paths.derived=/dev/shm/vla-large-data-v1`, `paths.runs=storage/runs`, and default `paths.archive=storage`. SSHFS uses alias `local-pc` and remote project root `/media/faps/External HDD/dev/VLA`; its existing noninteractive SSH authentication must work. These are current-host bindings, not portable defaults.

| Artifact | Current path | Persistence |
| --- | --- | --- |
| Snapshots | `/dev/shm/vla-large-data-v1/snapshots/{train,validation}` | Lost on reboot |
| Final v2.1 datasets | `/dev/shm/vla-large-data-v1/lerobot/local/g1_inspire_larger_v1_{train,validation}` | Lost on reboot |
| Bootstrap processor | `/dev/shm/vla-large-data-v1/processor` | Frozen and archived; RAM copy lost on reboot |
| External conversion staging | `storage/derived/larger-dataset-v1-conversion` | Durable; preserve |
| Data archive | `storage/derived/larger-dataset-v1/dataset.tar.gz` | Verified durable copy |
| Conversion report | `reports/larger-dataset-v1-conversion.json` | Complete; ignored local evidence |
| Protocol and sample JSON | `experiments/larger-dataset-v1/{evaluation-protocol,train-samples,validation-samples}.json` | Frozen and archived |
| Sample target arrays | `experiments/larger-dataset-v1/{train,validation}-samples.npz` | Archived; local files ignored by Git |
| New run | `storage/runs/larger-dataset-v1` | Not started; durable output |

The archive includes the processor and `evaluation-evidence/` copies of the protocol, both sample JSON files and both ignored NPZ files. `evaluation-evidence/manifest.json` maps those five files back to their original project paths and hashes. The production report is `reports/larger-dataset-v1-archive.json`; its adjacent durable verification record is `storage/derived/larger-dataset-v1/archive_verification.json`. The [execution guide](larger-dataset-execution-v1.md) records the completed preparation commands; they are not next-session prerequisites to repeat.

After a reboot, re-establish the configured SSHFS mount with `python3 scripts/project_storage.py --config reports/larger-dataset-v1-machine.json mount`, then check `status`. Never write into an unmounted `storage` directory or hide existing files beneath a mount. Verify the compressed archive SHA256 and manifest SHA256 against its successful verification record. Restore only into a fresh empty location, rejecting absolute/traversal paths, links and unlisted members, then verify every restored member's size and SHA256. A full reboot/extract/restore rehearsal has not been performed, and no general restoration helper is implemented; archive integrity checks do not establish a tested restore.

To reuse this protocol, restore its data and processor bytes to **the exact `/dev/shm/vla-large-data-v1` paths**, recover the conversion report from archived `provenance/conversion.json` to its original report path, and restore the five evaluation files to the paths recorded by `evaluation-evidence/`. Existing matching files may be verified and reused; differing files must be preserved for review. Archive-only `provenance/` and `evaluation-evidence/` files need not enter either dataset directory. Preserve the ignored machine configuration or reconstruct the same settings above. A different checkout, data root, model cache, or changed helper source requires explicit new execution bindings and a new declared protocol version; never edit frozen paths or hashes to make validation pass.

For the next session, confirm the archive and protocol are complete, the source bindings still match, and no larger-run output or step-zero reports already exist. If they do, inspect them before choosing a continuation. Capacity checks must run on the actual host: the sandbox can show a different `/dev/shm` mount. Retain at least 80 GiB free durable output space, 60 GiB available host RAM, 40 GiB free VRAM, and 8 GiB project reserve. Prior readiness measurements are historical. Four approximately 12.58 GB checkpoints plus the pinned final root copy are retained; measured SSHFS writes put compute plus saves near 5.6 hours, so the six-hour cap may produce an incomplete run.

The commands below are **next-session commands**, after the user asks to resume. They do not repair missing preparation outputs.

```bash
VLA_CFG=reports/larger-dataset-v1-machine.json
VLA_PY=vendor/Isaac-GR00T/.venv/bin/python
VLA_DATA=/dev/shm/vla-large-data-v1
VLA_PROTOCOL=experiments/larger-dataset-v1/evaluation-protocol.json
VLA_TRAIN="$VLA_DATA/lerobot/local/g1_inspire_larger_v1_train"
VLA_ZERO_TRAIN=reports/larger-dataset-v1-step-zero-train.json
VLA_ZERO_VALIDATION=reports/larger-dataset-v1-step-zero-validation.json

python3 scripts/project_storage.py --config "$VLA_CFG" status
"$VLA_PY" -B scripts/verify_pipeline.py --config "$VLA_CFG" \
  --level gpu --profile training --dataset "$VLA_TRAIN" \
  --report reports/larger-dataset-v1-next-session-readiness.json
```

That general readiness helper retains the pilot's 35 GiB output check; the larger runner separately enforces **80 GiB**. Once current resources and frozen bindings pass, establish the missing baseline with one fresh model load shared by both splits:

```bash
"$VLA_PY" -B scripts/evaluate_gr00t_dataset.py --config "$VLA_CFG" evaluate \
  --protocol "$VLA_PROTOCOL" --split both --step-zero \
  --train-report "$VLA_ZERO_TRAIN" --validation-report "$VLA_ZERO_VALIDATION"

"$VLA_PY" -B scripts/run_gr00t_dataset.py --config "$VLA_CFG" \
  --train-dataset "$VLA_TRAIN" \
  --train-manifest "$VLA_DATA/snapshots/train/manifest.json" \
  --validation-manifest "$VLA_DATA/snapshots/validation/manifest.json" \
  --split-manifest experiments/larger-dataset-v1/split.json \
  --evaluation-protocol "$VLA_PROTOCOL" \
  --train-baseline-report "$VLA_ZERO_TRAIN" \
  --validation-baseline-report "$VLA_ZERO_VALIDATION" \
  --run larger-dataset-v1 --timeout-seconds 21600 --dry-run
```

Check both successful baseline reports and the dry-run output. Then run the same training command without `--dry-run` under the user's resumed authorization. The launcher checks committed sources and current reserves; there is no automatic timeout extension or optimizer resume. Do not run evaluation concurrently with training.

After training exits, follow the [execution guide's checkpoint loop and selection commands](larger-dataset-execution-v1.md): inventory/hash/fsync each complete checkpoint on the storage host, run paired train/validation evaluation once per checkpoint, publish its evidence in place, then select only after all four pairs pass integrity checks. Keep the referenced durable checkpoint paths immutable. Validation group-balanced normalized MAE determines selection; persistence remains descriptive, and offline scores do not establish robot task success.
