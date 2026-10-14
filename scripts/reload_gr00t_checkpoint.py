"""Reload a saved checkpoint in a fresh process and predict one absolute action chunk."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import time

from run_gr00t_smoke import ROOT, GIB, available_memory, prepare_offline_workdir, existing_ancestor
from project_config import load_config, resolve_path, guard_derived, guard_path
from convert_single_camera import ensure_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    args.checkpoint = resolve_path(args.checkpoint)
    args.dataset = guard_derived(args.dataset, config)
    args.report = guard_path(args.report, config, "reports")
    for output in [args.report, args.report.with_suffix(".npz")]:
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    reserve = config["reserve_gib"] * GIB
    if (shutil.disk_usage(ROOT).free < reserve
            or shutil.disk_usage(existing_ancestor(args.report)).free < reserve
            or available_memory() < 40 * GIB):
        raise RuntimeError("Insufficient disk or host-memory reserve for checkpoint reload")
    runtime = ensure_python(config, "gr00t")
    os.chdir(prepare_offline_workdir(config, runtime))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    policy = Gr00tPolicy(embodiment_tag="NEW_EMBODIMENT", model_path=str(args.checkpoint.resolve()), device="cuda:0")
    loading_seconds = time.monotonic() - start
    loader = LeRobotEpisodeLoader(dataset_path=args.dataset, modality_configs=policy.modality_configs)
    trajectory = loader[0]
    modalities = copy.deepcopy(policy.modality_configs)
    modalities.pop("action")
    step = extract_step_data(trajectory, 0, modalities, EmbodimentTag.NEW_EMBODIMENT)
    obs = {"video": {k: np.asarray(v, dtype=np.uint8)[None] for k,v in step.images.items()},
           "state": {k: np.asarray(v, dtype=np.float32)[None] for k,v in step.states.items()},
           "language": {policy.language_key: [[step.text]]}}
    torch.cuda.synchronize()
    start = time.monotonic()
    action, _ = policy.get_action(obs)
    torch.cuda.synchronize()
    infer_seconds = time.monotonic() - start
    keys = ["left_arm", "right_arm", "left_hand", "right_hand"]
    joined = np.concatenate([action[k][0] for k in keys], axis=-1)
    assert joined.shape == (16, 26), joined.shape
    assert joined.dtype == np.float32 and np.isfinite(joined).all()
    np.savez_compressed(args.report.with_suffix(".npz"), absolute_actions=joined)
    trainer_state_path = args.checkpoint / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text()) if trainer_state_path.is_file() else {}
    record = {"passed": True, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_global_step": trainer_state.get("global_step"),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "dataset": str(args.dataset.resolve()),
              "fresh_process": True, "offline": True, "observation_episode": 0, "observation_frame": 0,
              "joint_order": keys, "action_shape": list(joined.shape), "dtype": str(joined.dtype), "finite": True,
              "output_semantics": "absolute, decoded by the saved GR00T processor", "instruction": step.text,
              "loading_seconds": loading_seconds, "first_inference_seconds": infer_seconds,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "groups": {k: {"shape":list(action[k].shape), "min":float(action[k].min()), "max":float(action[k].max())} for k in keys},
              "statistics_sha256": hashlib.sha256((args.checkpoint / "statistics.json").read_bytes()).hexdigest(),
              "limitation": "Offline checkpoint reload only; this does not establish task competence or robot safety."}
    with args.report.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
