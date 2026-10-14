#!/usr/bin/env python3
"""Add minimal GR00T v2.1 metadata and verify against the immutable pilot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_pilot import PROJECT, vector
from project_config import load_config, guard_derived, load_runtime, runtime_environment
from convert_single_camera import ensure_python

ANNOTATION = "annotation.human.task_description"
VIDEO_KEY = "observation.images.cam_left_high"


def derived_path(path, config=None):
    return guard_derived(path, config or load_config())


def validate_parquet_inventory(root, expected):
    """Upstream stats glob every parquet; reject unlisted episodes before mutation."""
    expected = {path.resolve() for path in expected}
    if any(not path.is_relative_to(root.resolve()) for path in expected):
        raise ValueError("Parquet metadata escapes the derived dataset")
    actual = {path.resolve() for path in root.glob("data/*/*.parquet")}
    if actual != expected:
        raise ValueError("Parquet inventory differs from the pilot manifest")


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def validate_video(path, raw_episode, raw_rows, ffmpeg, ffprobe):
    import numpy as np
    from PIL import Image

    result = subprocess.run([
        ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    if int(stream["nb_read_frames"]) != len(raw_rows):
        raise ValueError(f"Decoded video/frame mismatch: {path}")
    if (stream["width"], stream["height"]) != (640, 480):
        raise ValueError(f"Video dimensions changed: {path}")
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    if abs(numerator / denominator - 30) > 1e-6:
        raise ValueError(f"Video fps changed: {path}")
    comparisons = []
    for index in sorted({0, len(raw_rows)//2, len(raw_rows)-1}):
        result = subprocess.run([
            ffmpeg, "-v", "error", "-nostdin", "-i", str(path),
            "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ], check=True, capture_output=True)
        decoded = np.frombuffer(result.stdout, dtype=np.uint8).reshape(480, 640, 3)
        with Image.open(raw_episode / raw_rows[index]["colors"]["color_0"]) as image:
            original = np.asarray(image.convert("RGB"))
        error = float(np.abs(decoded.astype(np.float32) - original.astype(np.float32)).mean())
        comparisons.append({"row": index, "rgb_mae_0_255": error})
    return {"decoded_frames": int(stream["nb_read_frames"]),
            "sampled_raw_video_comparison": comparisons,
            "comparison_note": "Lossy video encoding; image errors are informational, visually inspect alignment."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--compute-stats", action="store_true", help="Run pinned GR00T statistics on this pilot only")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--apply-metadata", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_python(config, "gr00t")
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, report_path = derived_path(args.dataset, config), derived_path(args.report, config)
    pilot = derived_path(args.pilot_root, config)
    if report_path.exists():
        raise FileExistsError(f"Refusing existing report: {report_path}")
    if (pilot / "INCOMPLETE").exists():
        raise ValueError("Pilot snapshot is incomplete")
    manifest = json.loads((pilot / "manifest.json").read_text())
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("This helper only accepts the converted v2.1 dataset")
    expected_episodes = manifest["episodes"]
    if info["total_episodes"] != len(expected_episodes):
        raise ValueError("Episode count differs from pilot")
    if info["total_frames"] != sum(row["rows"] for row in expected_episodes):
        raise ValueError("Total frame count differs from pilot")
    expected_parquet = [root / info["data_path"].format(
        episode_chunk=record["derived_episode_index"] // info.get("chunks_size", 1000),
        episode_index=record["derived_episode_index"]) for record in expected_episodes]
    validate_parquet_inventory(root, expected_parquet)
    video_keys = [key for key, feature in info["features"].items()
                  if feature["dtype"] == "video"]
    if video_keys != [VIDEO_KEY]:
        raise ValueError(f"Unexpected camera features: {video_keys}")
    tasks = {row["task_index"]: row["task"] for row in
             map(json.loads, (root / "meta" / "tasks.jsonl").read_text().splitlines())}
    modalities = json.loads((PROJECT / "configs" / "modality.json").read_text())
    modality_path = root / "meta" / "modality.json"
    if modality_path.exists() and json.loads(modality_path.read_text()) != modalities:
        raise ValueError("Existing modality metadata differs from project configuration")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("ffmpeg and ffprobe must be available for video validation")
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "dataset": str(root), "pilot_manifest": str(pilot / "manifest.json"),
              "apply_metadata": args.apply_metadata, "episodes": [],
              "instruction_policy": "exact source labels retained", "normalization": "pilot only"}
    running_index = 0
    for record in expected_episodes:
        index = record["derived_episode_index"]
        chunk = index // info.get("chunks_size", 1000)
        parquet = root / info["data_path"].format(episode_chunk=chunk, episode_index=index)
        raw_episode = pilot / "raw" / "pick_up_black_box" / record["episode"]
        rows = json.loads((raw_episode / "data.json").read_text())["data"]
        table = pq.read_table(parquet)
        if table.num_rows != len(rows):
            raise ValueError(f"Parquet length differs: {parquet}")
        for column, raw_field in [("observation.state", "states"), ("action", "actions")]:
            actual = np.asarray(table[column].to_pylist(), dtype=np.float32)
            expected = np.asarray([vector(row, raw_field) for row in rows], dtype=np.float32)
            if actual.shape != (len(rows), 26) or not np.isfinite(actual).all():
                raise ValueError(f"Bad {column} shape/values: {parquet}")
            if not np.array_equal(actual, expected):
                raise ValueError(f"Raw values changed in {column}: {parquet}")
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), index)
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(len(rows)))
        np.testing.assert_array_equal(table["index"].to_numpy(),
                                      np.arange(running_index, running_index + len(rows)))
        np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(len(rows))/30,
                                   atol=1e-5, rtol=0)
        running_index += len(rows)
        task_index = table["task_index"].cast(pa.int64())
        if any(tasks[value] != record["goal"] for value in task_index.to_pylist()):
            raise ValueError(f"Task text changed: {parquet}")
        added = []
        required = {
            ANNOTATION: task_index,
            "next.done": pa.array([False] * (len(rows) - 1) + [True]),
            "next.reward": pa.array([0.0] * len(rows), type=pa.float32()),
        }
        for name, values in required.items():
            if name in table.column_names:
                if name != "next.reward" and table[name].cast(values.type).to_pylist() != values.to_pylist():
                    raise ValueError(f"Existing {name} conflicts with expected values: {parquet}")
            elif args.apply_metadata:
                table = table.append_column(name, values)
                added.append(name)
            else:
                raise ValueError(f"Missing {name}; run with --apply-metadata")
        if added:
            temporary = parquet.with_suffix(".parquet.tmp")
            if temporary.exists():
                raise FileExistsError(f"Refusing existing temporary file: {temporary}")
            pq.write_table(table, temporary)
            temporary.replace(parquet)
        video = root / info["video_path"].format(episode_chunk=chunk, episode_index=index,
                                                  video_key=VIDEO_KEY)
        video_report = validate_video(video, raw_episode, rows, ffmpeg, ffprobe)
        report["episodes"].append({
            "episode": record["episode"], "episode_index": index, "rows": len(rows),
            "state_action_float32_exact": True, "instruction_exact": True,
            "added_columns": added, **video_report,
        })
        print(f"Validated {record['episode']}: {len(rows)} rows", flush=True)
    if args.apply_metadata:
        for name, dtype in [(ANNOTATION, "int64"), ("next.done", "bool"), ("next.reward", "float32")]:
            info["features"].setdefault(name, {"dtype": dtype, "shape": [1], "names": None})
        write_json(info_path, info)
        if not modality_path.exists():
            write_json(modality_path, modalities)
    elif not modality_path.exists():
        raise ValueError("Missing modality.json")
    if args.compute_stats:
        runtime = load_runtime(config)
        subprocess.run([str(runtime["interpreters"]["gr00t"]),
                        str(runtime["sources"]["gr00t"] / "gr00t/data/stats.py"),
                        "--dataset-path", str(root), "--embodiment-tag", "NEW_EMBODIMENT",
                        "--modality-config-path", str(PROJECT / "configs/g1_inspire_config.py")],
                       cwd=PROJECT, env=runtime_environment(config, offline=True), check=True)
        report["statistics_checked_or_generated"] = True
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(report_path)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, AssertionError, subprocess.CalledProcessError) as error:
        sys.exit(f"GR00T preparation/validation failed: {error}")
