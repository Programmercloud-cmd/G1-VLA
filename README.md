# G1 VLA fine-tuning pipeline

Fine-tune GR00T N1.7 on Unitree G1/Inspire recordings from XR teleoperation. The pipeline uses pinned Unitree and NVIDIA converters, one RGB camera, and 26 arm/hand state and action values.

**Completed:** the zero-state-dropout follow-up ran 1,024 additional updates from checkpoint 512 in 28 minutes, then passed fresh-process reload, paired offline evaluation, and checksum-verified archival. Native MAE improved by 14.1–16.8% across joint groups versus checkpoint 512, but all eight persistence checks failed: only 8 of the 16 required acceptance checks passed. The policy is not a deployment candidate. That five-episode pilot had no held-out evaluation; no robot trial has run.

See [follow-up results and trajectories](experiments/pilot-fit-no-state-dropout-v1/README.md), [the frozen evaluation protocol](experiments/pilot-fit-no-state-dropout-v1/evaluation-protocol.json), and [the execution plan](docs/execution-plan.md). All [62 checkpoint and evidence files](experiments/pilot-fit-no-state-dropout-v1/artifacts.json) were durably verified with source outputs preserved. The earlier [512-update pilot](experiments/pilot-fit-v1/README.md) and [two-step smoke](experiments/pilot-smoke-v2/summary.json) remain preserved.

**Larger-dataset partial run:** the approved [experiment](experiments/larger-dataset-v1/README.md) has 98 training and 24 validation episodes, a frozen processor/evaluation protocol, and a verified durable archive of 48,204 files. Restoration and pretrained step-zero evaluations passed. The [fresh six-hour-capped retry](docs/larger-dataset-v1-retry2.md) stopped at 5,231 of 8,192 updates when the unchanged local disk reserve guard fired, before its time cap. Checkpoints at 2,048 and 4,096 updates and all final evidence are preserved; both paired train/validation evaluations, archives and CPU analysis passed. The [partial outcome](experiments/larger-dataset-v1/retry2-outcome.json) records the finite prefix. Validation group-balanced normalized MAE fell from 0.765781 pretrained to 0.196946 at 2,048 and 0.149138 at 4,096 (80.5% lower). See [the results](docs/larger-dataset-v1-results.md) and [curated evidence](experiments/larger-dataset-v1/partial-results.json). No automatic training retry or four-checkpoint selection is authorized. Persistence remains descriptive; offline errors do not establish robot success.

**Continuation completed and evaluated:** [4,096 additional updates](experiments/larger-dataset-v1-continuation-v1/training-outcome.json) completed successfully, yielding cumulative weight lineage 8,192 with fresh optimizer state at 4,096. Both new full-state checkpoints, paired offline evaluations and durable archives passed. The formal comparison selected [cumulative 8,192](experiments/larger-dataset-v1-continuation-v1/selected-policy.json): validation group-balanced normalized MAE **0.110770**, 25.7% below the parent 4,096 checkpoint (0.149138). The intermediate 6,144 checkpoint regressed slightly to 0.152467. See [final results and curves](docs/larger-dataset-v1-continuation-results.md), [curated evidence](experiments/larger-dataset-v1-continuation-v1/results.json), and [execution history](docs/larger-dataset-v1-continuation.md). The model-only inference view is nonresumable; its separate canonical checkpoint retains optimizer/scheduler/RNG state. No robot trial has run, and offline selection does not establish hardware task competence.

## Clone and configure

The supported setup target is Ubuntu 22.04 on x86-64 Linux with a working NVIDIA driver and the build/FFmpeg prerequisites below. The validated training machine had an RTX A6000 with 48 GiB VRAM and 125 GiB host RAM. The pilot used microbatch 1 with accumulation 8 and peaked at 37.9 GiB VRAM; larger microbatches are unvalidated.

```bash
git clone --recurse-submodules git@github.com:Programmercloud-cmd/G1-VLA.git
cd G1-VLA
cp machine.example.json machine.local.json
cp .env.example .env.local
```

Edit `machine.local.json` for the recording location, cache, derived datasets, run output, and archive destination. Relative paths resolve against the checkout, regardless of the calling directory. Set your read token privately in `.env.local` if model downloads require it; obtain access to the pinned [Cosmos model](https://huggingface.co/nvidia/Cosmos-Reason2-2B). Machine configuration, credentials, models, datasets, environments, and generated reports are ignored by Git.

The default run directory is Linux RAM storage, `/dev/shm/vla-groot-smoke`. It must have at least 35 GiB free for the smoke, and host available RAM must exceed 60 GiB. It is transient and must be archived before shutdown. Alternatively configure a sufficiently large disk directory. Setup and execution preserve at least 8 GiB on the project filesystem. Keep executable environments on a Linux filesystem; a separate disk or optional SSH storage can hold bulk artifacts.

## Set up and verify

On a fresh supported host, install the complete system prerequisite set before `--apply`:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential python3-dev ffmpeg ca-certificates
```

This includes FFmpeg's shared-library dependencies. The historical project-local FFmpeg package list is not a complete fresh-host installation; [bootstrap-tools.json](locks/bootstrap-tools.json) records the prerequisite contract.

```bash
python3 scripts/setup_pipeline.py --plan
python3 scripts/setup_pipeline.py --apply
python3 scripts/verify_pipeline.py --level structure
python3 scripts/verify_pipeline.py --level imports
python3 scripts/verify_pipeline.py --level gpu
```

Setup reads the source, package, and model locks, prepares the separate environments, and writes a machine-local runtime inventory. The plan command shows the work before downloads. Verification checks source pins, exact package versions, model assets, encoded-video decoding, and optional GPU kernels without starting training. The default setup profile checks the runtime without requiring a prepared dataset or training capacity; every reported check must pass before proceeding. Use `--config path/to/config.json` on any entry point, or set `VLA_CONFIG`, to select another machine configuration.

Source checkouts are pinned submodules under `vendor/`: Isaac-GR00T, Unitree LeRobot and its nested sources, and the separate LeRobot revision used for NVIDIA format conversion. Exact references are in [locks/runtime-sources.json](locks/runtime-sources.json); model revisions are in [locks/model-sources.json](locks/model-sources.json) and required-file hashes in [locks/model-assets.json](locks/model-assets.json). Do not update these independently during an experiment. Use the configured interpreters directly rather than an automatic `uv run` resync.

## Convert a small pilot

Original recordings remain read-only. These commands create a new five-episode snapshot with reviewed workspace labels, convert it to LeRobot v3 and then v2.1, check raw values and decoded images, compute pilot statistics, and exercise GR00T's actual loader. The override manifest is bound to these recordings; other recordings need their own reviewed manifest. Choose new run names when outputs already exist.

```bash
python3 scripts/prepare_pilot.py \
  --count 5 --run workspace-label-pilot-v1 --max-mib 512 \
  --label-overrides annotations/goal-overrides-workspace-v1.json
python3 scripts/convert_single_camera.py \
  --pilot-root derived/workspace-label-pilot-v1 --repo-id local/g1_workspace_label_pilot_v1
python3 scripts/convert_single_camera.py --to-v21 \
  --pilot-root derived/workspace-label-pilot-v1 --repo-id local/g1_workspace_label_pilot_v1
python3 scripts/prepare_gr00t_dataset.py \
  --dataset derived/lerobot/local/g1_workspace_label_pilot_v1 \
  --pilot-root derived/workspace-label-pilot-v1 --report derived/workspace-label-pilot-v1-validation.json \
  --apply-metadata --compute-stats
python3 scripts/check_gr00t_loader.py \
  --dataset derived/lerobot/local/g1_workspace_label_pilot_v1 \
  --pilot-root derived/workspace-label-pilot-v1 --report derived/workspace-label-pilot-v1-loader.json
```

After conversion, check dataset and training capacity together:

```bash
python3 scripts/verify_pipeline.py --level gpu --profile training \
  --dataset derived/lerobot/local/g1_workspace_label_pilot_v1 \
  --report reports/workspace-label-pilot-v1-training-readiness.json
```

The training profile requires the dataset metadata, an actual GR00T loader sample, at least 40 GiB total and currently free VRAM on selected CUDA device 0, 60 GiB available host RAM, and 35 GiB free run-output space. It also checks the configured disk reserve. A passing check is a readiness snapshot, not a memory guarantee for larger batches.

The examples use the default relative directories; adjust explicit artifact paths when changing `paths.derived`. The converter preserves the snapshot's instructions and absolute joint targets. GR00T's processor makes arm targets relative to the observed state; hands remain absolute. Joint order is left arm 7, right arm 7, left hand 6, right hand 6. The one camera maps `color_0` to `cam_left_high`.

The [task-label review and override workflow](docs/dataset-labels.md) covers 125 provisional source JSON files with identical original goals and 337 image evidence references. The selected reference is the workspace as seen by the camera, not another box or a hard image-center split. The [accepted workspace overlay](annotations/goal-overrides-workspace-v1.json) has 122 accepted labels: 76 left/front, 44 right/front, and two front-only (0169, 0170); 0092, 0136, and 0165 remain on hold; 0165 lacked local images during the label review. [Annotation validation](annotations/workspace-label-validation-v1.json) checks the manifest and source bindings. All 122 accepted episodes are converted, validated and archived in the larger experiment's train/validation split; pretrained GPU evaluations passed, and a partial training run saved checkpoints at 2,048 and 4,096 updates; their paired evaluations and archives passed; see [the partial results](docs/larger-dataset-v1-results.md). The earlier episode 0141 integration test, original recordings and completed pilot are preserved. Selecting different boxes from changing instructions still needs separate data/evaluation; see [the experiment plan](docs/execution-plan.md).

## Run and reload the two-step smoke

```bash
python3 scripts/run_gr00t_smoke.py \
  --dataset derived/lerobot/local/g1_workspace_label_pilot_v1 --run smoke-v1
python3 scripts/reload_gr00t_checkpoint.py \
  --checkpoint /dev/shm/vla-groot-smoke/smoke-v1/checkpoint-2 \
  --dataset derived/lerobot/local/g1_workspace_label_pilot_v1 \
  --report reports/smoke-v1-reload.json
```

Use the configured run path for `--checkpoint`. The training helper fixes batch 1, accumulation 1, zero loader workers, exactly two optimizer steps, and one model-only checkpoint. Both commands use pinned local model assets offline; an isolated Cosmos path alias handles the pinned Transformers tokenizer behavior without editing upstream code. A saved checkpoint still needs the pinned Cosmos assets for reload. Model-only checkpoints cannot resume optimizer/scheduler state exactly.

See [storage and archival](docs/storage.md) for durable publication. Upstream also writes a duplicate final model at the run root; archive the complete `checkpoint-2` directory, not both copies.

## Evidence and repository scope

The historical two-step smoke used episodes 0037, 0040, 0042, 0046, and 0047: 2,116 frames. Its aggregate training loss was 1.4560; loss filtering means this does not prove every individual step loss was finite. Fresh reload took 10.84 s and the first prediction took 0.490 s on one observation, which is not a deployment benchmark.

The [portability validation record](docs/reproducibility-validation.md) separates checks performed on this host from untested fresh-host installation. Small sanitized records are tracked in [experiments/pilot-smoke-v2](experiments/pilot-smoke-v2). Artifact references use `archive:` paths relative to the configured archive root; recordings and weights are not distributed in this repository. Original machine-local reports remain unchanged outside Git. Historical file hashes identify that experiment's files, including documentation at archival time; they are not checksums of today's repository files.

The existing Apache 2.0 [license](LICENSE) is preserved. Each pinned upstream project and downloaded model retains its own license.
