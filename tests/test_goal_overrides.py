"""Goal overlays change only reviewed snapshot labels and retain verifiable sources."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_pilot as prep
from project_config import load_config


class GoalOverrideTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        config_path = self.root / "machine.json"
        config_path.write_text(json.dumps({"paths": {
            "raw_data": str(self.raw), "derived": str(self.root / "derived"),
            "archive": str(self.root / "archive")}}))
        self.config = load_config(config_path)
        self.source = self.make_episode("episode_0001")
        self.labels = self.root / "labels.json"
        self.write_labels([self.entry(self.source)])

    def make_episode(self, name):
        episode = self.raw / name
        (episode / "colors").mkdir(parents=True)
        payload = {"info": {"image": {"width": 640, "height": 480, "fps": 30}},
                   "text": {"goal": "Pick up the box on the left.", "note": "Keep this field."},
                   "data": []}
        for i in range(2):
            image_path = f"colors/{i:06}_color_0.jpg"
            Image.new("RGB", (640, 480), (i * 30, 50, 70)).save(episode / image_path)
            parts = {part: {"qpos": [0.1 + i * 0.01] * size} for part, size in prep.PARTS}
            payload["data"].append({"idx": i, "colors": {"color_0": image_path},
                                    "states": copy.deepcopy(parts), "actions": copy.deepcopy(parts)})
        (episode / "data.json").write_text(json.dumps(payload))
        return episode

    def entry(self, episode, status="accepted"):
        return {"episode": episode.name, "source_data_json_sha256": prep.digest(episode / "data.json"),
                "original_goal": json.loads((episode / "data.json").read_text())["text"]["goal"],
                "status": status, "goal": "Pick up the box on the right.",
                "frames_viewed": [0, 1]}

    def write_labels(self, entries, **changes):
        manifest = {"schema_version": 1, "dataset_name": self.raw.name,
                    "reference_frame": "camera workspace", "episodes": entries}
        manifest.update(changes)
        self.labels.write_text(json.dumps(manifest, indent=2) + "\n")

    def prepare(self, name="overlay", **kwargs):
        prep.prepare(1, name, 10, self.config, label_overrides=self.labels, **kwargs)
        root = self.root / "derived" / name
        return root, prep.validate_snapshot(root)

    def test_only_derived_goal_changes_and_raw_bytes_are_untouched(self):
        before = prep.inspect_episode(self.source)
        original = (self.source / "data.json").read_bytes()
        root, manifest = self.prepare()
        copied = root / "raw/pick_up_black_box/episode_0001"
        expected = json.loads(original)
        expected["text"]["goal"] = "Pick up the box on the right."
        self.assertEqual(json.loads((copied / "data.json").read_bytes()), expected)
        self.assertEqual((self.source / "data.json").read_bytes(), original)
        self.assertEqual(prep.inspect_episode(self.source)["files_sha256"], before["files_sha256"])
        record = manifest["episodes"][0]
        self.assertEqual(record["source_files_sha256"], before["files_sha256"])
        self.assertEqual(record["source_content_sha256"], before["content_sha256"])
        self.assertEqual(record["source_data_json_sha256"], before["files_sha256"]["data.json"])
        self.assertEqual(record["original_goal"], before["goal"])
        self.assertEqual(record["effective_goal"], expected["text"]["goal"])
        self.assertEqual(record["goal_status"], "accepted")
        self.assertEqual(record["label_manifest_sha256"], prep.digest(self.labels))
        self.assertEqual((root / "label-overrides.json").read_bytes(), self.labels.read_bytes())
        self.assertEqual((root / "provenance/episode_0001.data.json").read_bytes(), original)
        self.assertEqual(record["bytes"], prep.inspect_episode(copied)["bytes"])
        self.assertNotEqual(record["files_sha256"]["data.json"], before["files_sha256"]["data.json"])
        for path, sha in before["files_sha256"].items():
            if path != "data.json":
                self.assertEqual(prep.digest(copied / path), sha)
        # A snapshot verifies independently of the original source and manifest locations.
        shutil.rmtree(self.raw)
        self.labels.unlink()
        prep.validate_snapshot(root)

    def test_manifest_and_original_provenance_are_hash_bound(self):
        root, _ = self.prepare()
        for relative, message in [("label-overrides.json", "manifest hash"),
                                  ("provenance/episode_0001.data.json", "provenance hash")]:
            with self.subTest(path=relative):
                path = root / relative
                content = path.read_bytes()
                path.write_bytes(content + b" ")
                with self.assertRaisesRegex(ValueError, message):
                    prep.validate_snapshot(root)
                path.write_bytes(content)

    def test_rehashed_non_goal_change_is_rejected_by_original_payload_check(self):
        root, manifest = self.prepare()
        path = root / "raw/pick_up_black_box/episode_0001/data.json"
        payload = json.loads(path.read_text())
        payload["data"][0]["actions"]["left_arm"]["qpos"][0] = 0.25
        path.write_text(json.dumps(payload))
        record = manifest["episodes"][0]
        record["files_sha256"]["data.json"] = prep.digest(path)
        record["content_sha256"] = prep.combined_digest(record["files_sha256"])
        (root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "other than text.goal"):
            prep.validate_snapshot(root)

    def test_unknown_and_needs_review_episodes_are_excluded(self):
        review = self.make_episode("episode_0002")
        accepted = self.make_episode("episode_0003")
        review_entry = self.entry(review, "needs_review")
        del review_entry["goal"]
        self.write_labels([review_entry, self.entry(accepted)])
        _, manifest = self.prepare()
        self.assertEqual([row["episode"] for row in manifest["episodes"]], [accepted.name])
        self.assertEqual([row["status"] for row in manifest["candidate_checks"]],
                         ["excluded", "excluded", "selected"])

    def test_stale_source_after_requested_count_is_fatal_before_output(self):
        other = self.make_episode("episode_0002")
        self.write_labels([self.entry(self.source), self.entry(other)])
        path = other / "data.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            self.prepare()
        self.assertFalse((self.root / "derived/overlay").exists())

    def test_original_goal_mismatch_is_fatal(self):
        entry = self.entry(self.source)
        entry["original_goal"] = "Another original goal"
        self.write_labels([entry])
        with self.assertRaisesRegex(ValueError, "original-goal mismatch"):
            self.prepare()
        self.assertFalse((self.root / "derived/overlay").exists())

    def test_duplicate_ids_and_invalid_bindings_are_rejected(self):
        entry = self.entry(self.source)
        self.write_labels([entry, entry])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()
        for field, value in [("episode", "../episode_0001"), ("episode", "/episode_0001"),
                             ("source_data_json_sha256", "invalid"), ("original_goal", " "),
                             ("goal", ""), ("goal", None), ("status", "unchecked")]:
            with self.subTest(field=field, value=value):
                self.write_labels([{**entry, field: value}])
                with self.assertRaises(ValueError):
                    self.prepare()
        self.assertFalse((self.root / "derived/overlay").exists())

    def test_invalid_manifest_and_source_path_escape_are_rejected(self):
        for changes in [{"schema_version": True}, {"schema_version": 2},
                        {"dataset_name": "another dataset"}, {"reference_frame": ""},
                        {"episodes": {}}]:
            with self.subTest(changes=changes):
                self.write_labels([self.entry(self.source)], **changes)
                with self.assertRaises(ValueError):
                    self.prepare()
        outside = self.root / "outside"
        shutil.move(self.source, outside)
        self.source.symlink_to(outside, target_is_directory=True)
        self.write_labels([self.entry(self.source)])
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare()

    def test_no_overlay_preserves_original_labels_and_existing_output(self):
        before = prep.inspect_episode(self.source)
        prep.prepare(1, "plain", 10, self.config)
        root = self.root / "derived/plain"
        manifest = prep.validate_snapshot(root)
        self.assertNotIn("label_overrides", manifest)
        self.assertEqual(manifest["episodes"][0]["files_sha256"], before["files_sha256"])
        original_manifest = (root / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prep.prepare(1, "plain", 10, self.config, label_overrides=self.labels)
        self.assertEqual((root / "manifest.json").read_bytes(), original_manifest)

    def test_eight_gib_project_and_output_reserve_is_required(self):
        usage = shutil.disk_usage(self.root)
        with patch.object(prep.shutil, "disk_usage", return_value=usage._replace(free=8 * 1024**3)):
            with self.assertRaisesRegex(ValueError, "8 GiB"):
                self.prepare()
        self.assertFalse((self.root / "derived/overlay").exists())


if __name__ == "__main__":
    unittest.main()
