#!/usr/bin/env python3
"""Validate and snapshot a bounded pilot without changing source recordings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

from project_config import PROJECT_ROOT as PROJECT, load_config, resolve_path, guard_derived
PARTS = [("left_arm", 7), ("right_arm", 7), ("left_ee", 6), ("right_ee", 6)]
EXCLUDED = {"episode_0136": "Previously identified 20-frame frozen-image run; targeted review pending."}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while hashing: {path}")
    return h.hexdigest()


def contained_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Image path escapes episode: {relative}")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes episode: {relative}")
    if not resolved.is_file():
        raise ValueError(f"Not a regular file: {relative}")
    return resolved


def vector(row, field):
    values = []
    for part, size in PARTS:
        segment = row[field][part]["qpos"]
        if len(segment) != size or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in segment
        ):
            raise ValueError(f"{field}.{part} must contain {size} finite numbers")
        values.extend(segment)
    return values


def combined_digest(files):
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def inspect_episode(episode):
    from PIL import Image

    json_path = episode / "data.json"
    first_json_hash = digest(json_path)
    payload = json.loads(json_path.read_text())
    rows = payload["data"]
    if not rows:
        raise ValueError("Empty recording")
    goal = payload.get("text", {}).get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("Missing text.goal")
    info = payload["info"]["image"]
    if (info["width"], info["height"]) != (640, 480) or float(info["fps"]) != 30:
        raise ValueError(f"Unexpected image metadata: {info}")
    files = {"data.json": first_json_hash}
    images = []
    states = []
    hand_min, hand_max = math.inf, -math.inf
    byte_count = json_path.stat().st_size
    previous_hash, run_length, longest_run, duplicate_transitions = None, 0, 0, 0
    run_start, motion_freezes = 0, []
    first_idx = rows[0]["idx"]

    def close_run(end):
        if run_length > 2:
            span = max(
                max(state[j] for state in states[run_start:end + 1])
                - min(state[j] for state in states[run_start:end + 1])
                for j in range(14)
            )
            if span > 1e-4:
                motion_freezes.append({"first_row": run_start, "last_row": end,
                                      "frames": run_length, "max_arm_span_rad": span})

    for i, row in enumerate(rows):
        if row["idx"] != first_idx + i:
            raise ValueError(f"Noncontiguous idx at row {i}")
        if set(row["colors"]) != {"color_0"}:
            raise ValueError(f"Unexpected cameras at row {i}")
        state, action = vector(row, "states"), vector(row, "actions")
        states.append(state)
        hands = state[14:] + action[14:]
        hand_min, hand_max = min(hand_min, *hands), max(hand_max, *hands)
        if min(hands) < -1e-6 or max(hands) > 1 + 1e-6:
            raise ValueError(f"Hand values outside normalized [0,1] range at row {i}")
        relative = row["colors"]["color_0"]
        if relative in images:
            raise ValueError(f"Repeated image reference at row {i}")
        path = contained_file(episode, relative)
        image_hash = digest(path)
        with Image.open(path) as image:
            if image.size != (640, 480) or image.mode != "RGB":
                raise ValueError(f"Unexpected image at row {i}: {image.size}, {image.mode}")
            image.load()
        if digest(path) != image_hash:
            raise ValueError(f"Image changed during decoding: {relative}")
        files[relative] = image_hash
        images.append(relative)
        byte_count += path.stat().st_size
        if image_hash == previous_hash:
            run_length += 1
            duplicate_transitions += 1
        else:
            if i:
                close_run(i - 1)
            run_start, run_length = i, 1
        longest_run = max(longest_run, run_length)
        previous_hash = image_hash
    close_run(len(rows) - 1)
    actual_jpegs = {str(p.relative_to(episode)) for p in (episode / "colors").glob("*.jpg")}
    if actual_jpegs != set(images):
        raise ValueError("Referenced images do not match the JPEG inventory")
    if digest(json_path) != first_json_hash:
        raise ValueError("JSON changed during inspection")
    for relative, expected in files.items():
        if digest(contained_file(episode, relative)) != expected:
            raise ValueError(f"Input changed during inspection: {relative}")
    return {
        "episode": episode.name, "source_path": str(episode.resolve()),
        "source_date": payload.get("info", {}).get("date"), "rows": len(rows),
        "source_idx_start": first_idx, "source_idx_end": rows[-1]["idx"],
        "goal": goal, "goal_status": "preserved; visual consistency review pending",
        "camera": "color_0", "resolution": [640, 480], "fps_nominal": 30,
        "state_dimension": 26, "action_dimension": 26, "stored_action": "absolute",
        "hand_range": [hand_min, hand_max], "bytes": byte_count,
        "duplicate_transitions": duplicate_transitions, "longest_identical_run": longest_run,
        "motion_freezes": motion_freezes,
        "review_frames": [images[i] for i in sorted({0, len(images)//4, len(images)//2,
                                                    3*len(images)//4, len(images)-1})],
        "files_sha256": files, "content_sha256": combined_digest(files),
    }


def validate_snapshot(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (root / "INCOMPLETE").exists():
        raise ValueError("Pilot has an INCOMPLETE marker")
    records = manifest["episodes"]
    if not records:
        raise ValueError("No episodes in manifest")
    task = root / "raw" / "pick_up_black_box"
    expected_names = {record["episode"] for record in records}
    if {path.name for path in task.iterdir()} != expected_names:
        raise ValueError("Snapshot episode inventory differs from manifest")
    for record in records:
        episode = task / record["episode"]
        inspected = inspect_episode(episode)
        if inspected["files_sha256"] != record["files_sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {episode.name}")
        if combined_digest(record["files_sha256"]) != record["content_sha256"]:
            raise ValueError(f"Invalid manifest digest: {episode.name}")
    return manifest


def prepare(count, run, max_mib, config=None, raw_root=None):
    config = config or load_config()
    raw = resolve_path(raw_root or config["paths"]["raw_data"])
    if count < 1 or count > 10 or max_mib < 1:
        raise ValueError("Use 1–10 episodes and a positive size limit")
    if Path(run).name != run or run in {".", "..", ""}:
        raise ValueError("Run must be a single directory name")
    derived = resolve_path(config["paths"]["derived"])
    output = guard_derived(derived / run, config)
    if output.is_relative_to(raw) or raw.is_relative_to(output):
        raise ValueError("Snapshot destination must not overlap source recordings")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing output: {output}")
    candidates = sorted(raw.glob("episode_*"))
    selected, considered = [], []
    total_bytes = 0
    for episode in candidates:
        if not episode.is_dir():
            continue
        if episode.name in EXCLUDED:
            considered.append({"episode": episode.name, "status": "excluded",
                               "reason": EXCLUDED[episode.name]})
            continue
        try:
            record = inspect_episode(episode)
            if total_bytes + record["bytes"] > max_mib * 1024**2:
                raise ValueError("Pilot size cap would be exceeded")
            selected.append(record)
            total_bytes += record["bytes"]
            considered.append({"episode": episode.name, "status": "selected"})
            print(f"Selected {episode.name}: {record['rows']} rows", flush=True)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            considered.append({"episode": episode.name, "status": "not_selected",
                               "reason": str(error)})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible episodes found for requested {count}")
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < total_bytes + 2 * 1024**3:
        raise ValueError("Insufficient space for pilot plus 2 GiB reserve")
    output.mkdir(parents=True)
    (output / "INCOMPLETE").write_text("Remove only after validation succeeds.\n")
    for episode_index, record in enumerate(selected):
        source = Path(record["source_path"])
        destination = output / "raw" / "pick_up_black_box" / record["episode"]
        for relative, expected in record["files_sha256"].items():
            original = contained_file(source, relative)
            if digest(original) != expected:
                raise ValueError(f"Source changed before copy: {original}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            if digest(target) != expected or digest(original) != expected:
                raise ValueError(f"Copy or source verification failed: {original}")
        record["derived_episode_index"] = episode_index
    manifest = {
        "schema_version": 1, "created_at": now(), "source_inventory_provisional": True,
        "source_folders_observed": len(candidates), "requested_count": count,
        "source_root": str(raw), "max_snapshot_mib": max_mib, "snapshot_bytes": total_bytes,
        "excluded": EXCLUDED, "candidate_checks": considered, "episodes": selected,
        "split_status": "pilot for plumbing only; no final train/validation split established",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Verify again before clearing the incomplete marker.
    for record in selected:
        copied = inspect_episode(output / "raw" / "pick_up_black_box" / record["episode"])
        if copied["files_sha256"] != record["files_sha256"]:
            raise ValueError(f"Final snapshot mismatch: {record['episode']}")
    (output / "INCOMPLETE").unlink()
    validate_snapshot(output)
    report = {key: manifest[key] for key in [
        "created_at", "source_inventory_provisional", "source_folders_observed",
        "snapshot_bytes", "excluded", "split_status",
    ]}
    report["episodes"] = [{k: v for k, v in r.items() if k != "files_sha256"} for r in selected]
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "episodes": len(selected),
                      "rows": sum(r["rows"] for r in selected),
                      "snapshot_bytes": total_bytes, "inventory_provisional": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Machine config; relative paths use the checkout root")
    parser.add_argument("--raw-root", type=Path, help="Override configured source recordings (read-only)")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--run", default="pilot-v1")
    parser.add_argument("--max-mib", type=int, default=512)
    parser.add_argument("--verify", type=Path, help="Verify an existing pilot without writing")
    args = parser.parse_args()
    config = load_config(args.config)
    from convert_single_camera import ensure_python
    ensure_python(config, "unitree")
    if args.verify:
        manifest = validate_snapshot(resolve_path(args.verify))
        print(json.dumps({"verified": str(args.verify), "episodes": len(manifest["episodes"])}))
    else:
        prepare(args.count, args.run, args.max_mib, config, args.raw_root)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        sys.exit(f"Preparation failed: {error}")
