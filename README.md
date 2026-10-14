# G1 VLA fine-tuning pipeline

Fine-tune GR00T N1.7 on Unitree G1/Inspire recordings from XR teleoperation. The pipeline uses pinned Unitree and NVIDIA converters, one RGB camera, and 26 arm/hand state and action values.

**Completed:** a five-episode pilot was converted and validated, two optimizer steps ran on an RTX A6000, and a fresh process reloaded the saved checkpoint and produced finite absolute actions of shape `(16, 26)`. The complete checkpoint was archived with matching SHA-256 hashes. This establishes pipeline operation; task success, held-out generalization, and robot deployment have not been demonstrated.

The approved 512-step fitting experiment has **not yet** run. See [the next experiment](docs/execution-plan.md) and the portable [smoke summary](experiments/pilot-smoke-v2/summary.json).

## Clone and configure

The supported setup target is Ubuntu 22.04 on x86-64 Linux with a working NVIDIA driver and the build/FFmpeg prerequisites below. The validated training machine had an RTX A6000 with 48 GiB VRAM and 125 GiB host RAM. Batch 1 peaked at 37.7 GiB VRAM; larger batches are unvalidated.

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

Original recordings remain read-only. These commands create a new five-episode snapshot, convert it to LeRobot v3 and then v2.1, check raw values and decoded images, compute pilot statistics, and exercise GR00T's actual loader. Choose new run names when outputs already exist.

```bash
python3 scripts/prepare_pilot.py --count 5 --run pilot-v1 --max-mib 512
python3 scripts/convert_single_camera.py \
  --pilot-root derived/pilot-v1 --repo-id local/g1_inspire_pilot_v1
python3 scripts/convert_single_camera.py --to-v21 \
  --pilot-root derived/pilot-v1 --repo-id local/g1_inspire_pilot_v1
python3 scripts/prepare_gr00t_dataset.py \
  --dataset derived/lerobot/local/g1_inspire_pilot_v1 \
  --pilot-root derived/pilot-v1 --report derived/pilot-v1-validation.json \
  --apply-metadata --compute-stats
python3 scripts/check_gr00t_loader.py \
  --dataset derived/lerobot/local/g1_inspire_pilot_v1 \
  --pilot-root derived/pilot-v1 --report derived/pilot-v1-loader.json
```

After conversion, check dataset and training capacity together:

```bash
python3 scripts/verify_pipeline.py --level gpu --profile training \
  --dataset derived/lerobot/local/g1_inspire_pilot_v1 \
  --report reports/training-readiness.json
```

The training profile requires the dataset metadata, an actual GR00T loader sample, at least 40 GiB total and currently free VRAM on selected CUDA device 0, 60 GiB available host RAM, and 35 GiB free run-output space. It also checks the configured disk reserve. A passing check is a readiness snapshot, not a memory guarantee for larger batches.

The examples use the default relative directories; adjust explicit artifact paths when changing `paths.derived`. The converter preserves original instructions and absolute joint targets. GR00T's processor makes arm targets relative to the observed state; hands remain absolute. Joint order is left arm 7, right arm 7, left hand 6, right hand 6. The one camera maps `color_0` to `cam_left_high`.

The recorded task picks the same target box from varied positions. Selecting different boxes from changing instructions needs separate data/evaluation. Before longer training, review positional wording against the images and resolve the flagged frozen-image episode; see [the experiment plan](docs/execution-plan.md).

## Run and reload the two-step smoke

```bash
python3 scripts/run_gr00t_smoke.py \
  --dataset derived/lerobot/local/g1_inspire_pilot_v1 --run smoke-v1
python3 scripts/reload_gr00t_checkpoint.py \
  --checkpoint /dev/shm/vla-groot-smoke/smoke-v1/checkpoint-2 \
  --dataset derived/lerobot/local/g1_inspire_pilot_v1 \
  --report reports/smoke-v1-reload.json
```

Use the configured run path for `--checkpoint`. The training helper fixes batch 1, accumulation 1, zero loader workers, exactly two optimizer steps, and one model-only checkpoint. Both commands use pinned local model assets offline; an isolated Cosmos path alias handles the pinned Transformers tokenizer behavior without editing upstream code. A saved checkpoint still needs the pinned Cosmos assets for reload. Model-only checkpoints cannot resume optimizer/scheduler state exactly.

See [storage and archival](docs/storage.md) for durable publication. Upstream also writes a duplicate final model at the run root; archive the complete `checkpoint-2` directory, not both copies.

## Evidence and repository scope

The historical pilot used episodes 0037, 0040, 0042, 0046, and 0047: 2,116 frames. Its aggregate training loss was 1.4560; loss filtering means this does not prove every individual step loss was finite. Fresh reload took 10.84 s and the first prediction took 0.490 s on one observation, which is not a deployment benchmark.

The [portability validation record](docs/reproducibility-validation.md) separates checks performed on this host from untested fresh-host installation. Small sanitized records are tracked in [experiments/pilot-smoke-v2](experiments/pilot-smoke-v2). Artifact references use `archive:` paths relative to the configured archive root; recordings and weights are not distributed in this repository. Original machine-local reports remain unchanged outside Git. Historical file hashes identify that experiment's files, including documentation at archival time; they are not checksums of today's repository files.

The existing Apache 2.0 [license](LICENSE) is preserved. Each pinned upstream project and downloaded model retains its own license.
