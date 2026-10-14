#!/usr/bin/env python3
"""Exercise the actual GR00T loader and record the pilot's input/target contract."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from prepare_pilot import PROJECT, digest
from prepare_gr00t_dataset import derived_path
from project_config import load_config
from convert_single_camera import ensure_python



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_python(config, "gr00t")
    import numpy as np
    sys.path.insert(0, str(PROJECT / "configs"))
    from g1_inspire_config import g1_inspire_config
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    dataset, pilot, report_path = (derived_path(p, config) for p in [args.dataset, args.pilot_root, args.report])
    if report_path.exists():
        raise FileExistsError(f"Refusing existing report: {report_path}")
    manifest = json.loads((pilot / "manifest.json").read_text())
    loader = LeRobotEpisodeLoader(dataset_path=dataset, modality_configs=g1_inspire_config)
    if len(loader) != len(manifest["episodes"]):
        raise ValueError("GR00T loader episode count differs from snapshot")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset": str(dataset),
        "config_sha256": digest(PROJECT / "configs" / "g1_inspire_config.py"),
        "metadata_sha256": {name: digest(dataset / "meta" / name)
                           for name in ["modality.json", "stats.json", "relative_stats.json"]},
        "stats_source": f"all {len(manifest['episodes'])} pilot episodes; no holdout/generalization claim",
        "episodes": [],
    }
    for index, record in enumerate(manifest["episodes"]):
        frame = loader[index]
        if len(frame) != record["rows"]:
            raise ValueError(f"Loader row count mismatch: {record['episode']}")
        checked = []
        for position in sorted({0, len(frame)//2, len(frame)-16}):
            point = extract_step_data(frame, position, g1_inspire_config,
                                      EmbodimentTag.NEW_EMBODIMENT)
            states = np.concatenate([point.states[key] for key in g1_inspire_config["state"].modality_keys], axis=-1)
            actions = np.concatenate([point.actions[key] for key in g1_inspire_config["action"].modality_keys], axis=-1)
            assert states.shape == (1, 26), states.shape
            assert actions.shape == (16, 26), actions.shape
            assert np.isfinite(states).all() and np.isfinite(actions).all()
            assert point.text == record["goal"]
            assert len(point.images["ego_view"]) == 1
            decoded = np.asarray(point.images["ego_view"][0])
            assert decoded.shape == (480, 640, 3), decoded.shape
            assert decoded.dtype == np.uint8, decoded.dtype
            checked.append(position)
        report["episodes"].append({"episode": record["episode"], "rows": len(frame),
                                    "checked_rows": checked, "state_shape": [1, 26],
                                    "action_shape": [16, 26], "image_shape": [1, 480, 640, 3],
                                    "language_exact": True})
        print(f"GR00T loader passed {record['episode']}: {checked}", flush=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(report_path)


if __name__ == "__main__":
    main()
