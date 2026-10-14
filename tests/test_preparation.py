"""Focused checks for source integrity, camera assumptions, and overwrite protection."""
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_pilot as prep
import convert_single_camera as conversion
import project_config
import run_gr00t_smoke as smoke
import archive_smoke_checkpoint as archive
import prepare_gr00t_dataset as metadata


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config_path = self.root / "machine.local.json"
        self.config_path.write_text(json.dumps({"paths": {
            "raw_data": str(self.root / "raw"), "derived": str(self.root / "derived"),
            "reports": str(self.root / "reports"), "archive": str(self.root / "storage"),
            "runs": str(self.root / "runs")}}))
        self.config = project_config.load_config(self.config_path)
        self.episode = self.root / "raw" / "episode_0001"
        (self.episode / "colors").mkdir(parents=True)
        self.payload = {"info": {"image": {"width": 640, "height": 480, "fps": 30}},
                        "text": {"goal": "Original instruction"}, "data": []}
        for i in range(3):
            relative = f"colors/{i:06}_color_0.jpg"
            Image.new("RGB", (640, 480), (30 + i * 20, 50, 70)).save(self.episode / relative)
            parts = {part: {"qpos": [0.1 + i * 0.01] * size} for part, size in prep.PARTS}
            self.payload["data"].append({
                "idx": i, "colors": {"color_0": relative},
                "states": copy.deepcopy(parts), "actions": copy.deepcopy(parts),
            })
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.episode / "data.json").write_text(json.dumps(self.payload))

    def test_complete_episode_preserves_label_and_has_content_hash(self):
        record = prep.inspect_episode(self.episode)
        self.assertEqual(record["rows"], 3)
        self.assertEqual(record["goal"], "Original instruction")
        self.assertEqual(len(record["files_sha256"]), 4)
        self.assertEqual(record["motion_freezes"], [])

    def test_missing_image_rejected(self):
        (self.episode / "colors/000001_color_0.jpg").unlink()
        with self.assertRaises(OSError):
            prep.inspect_episode(self.episode)

    def test_nonfinite_joint_rejected(self):
        self.payload["data"][0]["actions"]["left_arm"]["qpos"][0] = float("nan")
        self.save()
        with self.assertRaisesRegex(ValueError, "finite"):
            prep.inspect_episode(self.episode)

    def test_wrong_vector_dimension_rejected(self):
        self.payload["data"][0]["states"]["left_ee"]["qpos"].append(0.0)
        self.save()
        with self.assertRaisesRegex(ValueError, "6 finite"):
            prep.inspect_episode(self.episode)

    def test_frame_reference_escape_rejected(self):
        self.payload["data"][0]["colors"]["color_0"] = "../outside.jpg"
        self.save()
        with self.assertRaisesRegex(ValueError, "escapes"):
            prep.inspect_episode(self.episode)

    def test_camera_freeze_with_motion_flagged(self):
        source = self.episode / "colors/000000_color_0.jpg"
        for index in [1, 2]:
            (self.episode / f"colors/{index:06}_color_0.jpg").write_bytes(source.read_bytes())
        record = prep.inspect_episode(self.episode)
        self.assertEqual(record["longest_identical_run"], 3)
        self.assertEqual(record["motion_freezes"][0]["frames"], 3)

    def test_joint_index_discontinuity_rejected(self):
        self.payload["data"][1]["idx"] = 100
        self.save()
        with self.assertRaisesRegex(ValueError, "Noncontiguous"):
            prep.inspect_episode(self.episode)

    def test_external_output_requires_configured_active_mount(self):
        config = copy.deepcopy(self.config)
        config["storage"].update(enabled=True, host="storage-host", remote_path="/archive")
        parent = self.root / "storage" / "derived"
        with patch("project_config.os.path.ismount", return_value=False):
            with self.assertRaisesRegex(ValueError, "not mounted"):
                conversion.guard_output(parent, "local/pilot", config)
        record = {"filesystems": [{"target": str(self.root / "storage"),
                  "source": "storage-host:/archive", "fstype": "fuse.sshfs"}]}
        with patch("project_config.os.path.ismount", return_value=True), patch(
                "project_config.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps(record))):
            self.assertEqual(conversion.guard_output(parent, "local/pilot", config), parent / "local/pilot")

    def test_upstream_backup_and_temporary_names_protected(self):
        parent = self.root / "derived" / "lerobot"
        for suffix in ["_v3.0", "_v2.1"]:
            backup = parent / "local" / ("pilot" + suffix)
            backup.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                conversion.guard_output(parent, "local/pilot", self.config)
            backup.rmdir()

    def test_existing_output_and_path_escape_rejected(self):
        parent = self.root / "derived" / "lerobot"
        target = parent / "local" / "pilot"
        target.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            conversion.guard_output(parent, "local/pilot", self.config)
        with self.assertRaises(ValueError):
            conversion.guard_output(self.root / "raw", "local/new", self.config)
        with self.assertRaises(ValueError):
            conversion.guard_output(parent, "../outside", self.config)

    def test_conversion_symlink_escape_rejected(self):
        parent = self.root / "derived" / "lerobot"
        parent.mkdir(parents=True)
        (parent / "local").symlink_to(self.root / "raw", target_is_directory=True)
        with self.assertRaises(ValueError):
            conversion.guard_output(parent, "local/pilot", self.config)

    def test_v21_mode_requires_v3_and_preserves_existing_backups(self):
        parent = self.root / "derived" / "lerobot"
        target = parent / "local" / "pilot"
        (target / "meta").mkdir(parents=True)
        info = target / "meta/info.json"
        info.write_text(json.dumps({"codebase_version": "v3.0"}))
        self.assertEqual(conversion.guard_output(parent, "local/pilot", self.config, allow_v3=True), target)
        info.write_text(json.dumps({"codebase_version": "v2.1"}))
        with self.assertRaisesRegex(ValueError, "Only v3.0"):
            conversion.guard_output(parent, "local/pilot", self.config, allow_v3=True)

    def test_snapshot_uses_configured_roots_and_preserves_originals(self):
        before = prep.inspect_episode(self.episode)["files_sha256"]
        prep.prepare(1, "pilot", 10, self.config)
        snapshot = self.root / "derived/pilot"
        manifest = prep.validate_snapshot(snapshot)
        self.assertEqual(manifest["episodes"][0]["files_sha256"], before)
        self.assertEqual(prep.inspect_episode(self.episode)["files_sha256"], before)
        with self.assertRaises(FileExistsError):
            prep.prepare(1, "pilot", 10, self.config)

    def test_smoke_recipe_remains_exactly_two_steps(self):
        runtime = project_config.load_runtime(self.config)
        command = smoke.smoke_command(self.config, runtime, self.root / "dataset", self.root / "run")
        for option, value in [("--max-steps", "2"), ("--save-steps", "2"),
                              ("--global-batch-size", "1"), ("--gradient-accumulation-steps", "1")]:
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertEqual(command[0], str(self.config["interpreters"]["gr00t"]))

    def test_archive_rejects_unverified_or_symlink_checkpoint(self):
        source = self.root / "checkpoint-2"
        source.mkdir()
        (source / "config.json").write_text("{}")
        (source / "model.safetensors").write_bytes(b"tiny test checkpoint")
        evidence = {"passed": True, "fresh_process": True, "checkpoint": str(source)}
        self.assertEqual(len(archive.validate_source(source, evidence)), 2)
        with self.assertRaises(ValueError):
            archive.validate_source(source, {**evidence, "passed": False})
        (source / "linked").symlink_to(self.episode / "data.json")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            archive.validate_source(source, evidence)

    def test_local_archive_hashes_and_refuses_overwrite(self):
        source = self.root / "model.safetensors"
        source.write_bytes(b"small integrity fixture")
        relative = "checkpoint-2/model.safetensors"
        record = {"path": relative, "size": source.stat().st_size, "sha256": archive.sha256(source)}
        manifest = {"checkpoint_source": str(source.parent), "checkpoint_directory": "checkpoint-2",
                    "files": [record], "total_bytes": record["size"], "checkpoint_bytes": record["size"]}
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        sources = [(source, relative)]
        signatures = {source: archive.signature(source)}
        stage, final = self.root / ".stage", self.root / "archive"
        report = archive.local_archive(stage, final, sources, signatures, manifest_path, manifest)
        self.assertTrue(report["passed"])
        self.assertEqual(archive.sha256(final / relative), archive.sha256(source))
        self.assertTrue(source.exists())
        with self.assertRaises(FileExistsError):
            archive.local_archive(stage, final, sources, signatures, manifest_path, manifest)

    def test_stats_input_inventory_excludes_unlisted_episodes(self):
        dataset = self.root / "dataset"
        listed = dataset / "data/chunk-000/episode_000000.parquet"
        listed.parent.mkdir(parents=True)
        listed.write_bytes(b"inventory fixture")
        metadata.validate_parquet_inventory(dataset, [listed])
        (listed.parent / "episode_999999.parquet").write_bytes(b"unlisted")
        with self.assertRaisesRegex(ValueError, "inventory"):
            metadata.validate_parquet_inventory(dataset, [listed])

    def test_relocated_checkout_prepares_from_another_cwd(self):
        checkout = self.root / "relocated checkout"
        (checkout / "scripts").mkdir(parents=True)
        (checkout / "locks").mkdir()
        original = Path(__file__).resolve().parents[1]
        for name in ["prepare_pilot.py", "convert_single_camera.py", "project_config.py"]:
            shutil.copy2(original / "scripts" / name, checkout / "scripts" / name)
        shutil.copy2(original / "locks/model-sources.json", checkout / "locks/model-sources.json")
        (checkout / "machine.local.json").write_text(json.dumps({
            "paths": {"raw_data": str(self.root / "raw"), "derived": "derived", "runs": "runs"},
            "interpreters": {"unitree": sys.executable}}))
        command = [sys.executable, "-B", str(checkout / "scripts/prepare_pilot.py"),
                   "--config", "machine.local.json", "--count", "1", "--run", "portable", "--max-mib", "10"]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = checkout / "derived/portable"
        self.assertEqual(len(prep.validate_snapshot(snapshot)["episodes"]), 1)
        verify = subprocess.run(command[:5] + ["--verify", "derived/portable"],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(verify.returncode, 0, verify.stderr)

    def test_python_selection_preserves_virtualenv_symlink(self):
        python = self.root / "environment/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        config = copy.deepcopy(self.config)
        config["interpreters"]["unitree"] = python
        with patch.object(conversion.os, "execve", side_effect=RuntimeError("intercept")) as execute:
            with self.assertRaisesRegex(RuntimeError, "intercept"):
                conversion.ensure_python(config, "unitree")
        self.assertEqual(execute.call_args.args[0], str(python))

    def test_imported_runtime_helper_reexecutes_the_actual_caller(self):
        interpreter = self.root / "runtime/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        config = json.loads(self.config_path.read_text())
        config["interpreters"] = {"unitree": str(interpreter)}
        self.config_path.write_text(json.dumps(config))
        caller = self.root / "a different caller.py"
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        caller.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            + "sys.path.insert(0, " + repr(str(scripts)) + ")\n"
            + "from project_config import load_config\n"
            + "from convert_single_camera import ensure_python\n"
            + "ensure_python(load_config(sys.argv[1]), 'unitree')\n"
            + "print(json.dumps({'caller':str(Path(__file__).resolve()),'python':sys.executable,'argument':sys.argv[2]}))\n")
        result = subprocess.run([sys.executable, "-B", str(caller), str(self.config_path), "sentinel"],
                                cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["caller"], str(caller))
        self.assertEqual(report["python"], str(interpreter))
        self.assertEqual(report["argument"], "sentinel")

    def test_same_interpreter_restarts_once_with_native_env_and_without_tokens(self):
        config = json.loads(self.config_path.read_text())
        config["interpreters"] = {"unitree": sys.executable}
        self.config_path.write_text(json.dumps(config))
        caller, trace = self.root / "same interpreter caller.py", self.root / "startup.jsonl"
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        caller.write_text(
            "import json,os,sys\nfrom pathlib import Path\n"
            + "with Path(sys.argv[2]).open('a') as stream: stream.write(json.dumps({'python':sys.executable,'library_path':os.environ.get('LD_LIBRARY_PATH'),'token_present':'HF_TOKEN' in os.environ})+'\\n')\n"
            + "sys.path.insert(0, " + repr(str(scripts)) + ")\n"
            + "from project_config import load_config\nfrom convert_single_camera import ensure_python\n"
            + "ensure_python(load_config(sys.argv[1]), 'unitree')\n"
            + "print(json.dumps({'token_present':'HF_TOKEN' in os.environ,'ready':True}))\n")
        env = os.environ.copy()
        for key in ["LD_LIBRARY_PATH", "VLA_RUNTIME_STARTUP", "VLA_CONFIG"]:
            env.pop(key, None)
        env["HF_TOKEN"] = "test-only-placeholder"
        result = subprocess.run([sys.executable, "-B", str(caller), str(self.config_path), str(trace)],
                                cwd=self.root, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        starts = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual(len(starts), 2, starts)
        self.assertEqual(starts[0]["python"], starts[1]["python"])
        self.assertIsNone(starts[0]["library_path"])
        self.assertTrue(starts[0]["token_present"])
        self.assertEqual(starts[1]["library_path"].split(":")[0], str(self.config["tools"]["ffmpeg_lib"]))
        self.assertFalse(starts[1]["token_present"])
        self.assertFalse(json.loads(result.stdout)["token_present"])

    def test_ssh_verifier_accepts_parameterized_host_and_checkpoint(self):
        stage, final = self.root / ".remote-stage", self.root / "remote-final"
        target = stage / "checkpoint-custom/model.safetensors"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"tiny remote-verification fixture")
        record = {"path": "checkpoint-custom/model.safetensors", "size": target.stat().st_size,
                  "sha256": archive.sha256(target)}
        manifest = {"checkpoint_source": "source/checkpoint-custom", "checkpoint_directory": "checkpoint-custom",
                    "destination_host": "test-host", "files": [record], "total_bytes": record["size"],
                    "checkpoint_bytes": record["size"], "reload_dependency": "test fixture"}
        manifest_path = stage / "archive_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        result = subprocess.run([sys.executable, "-B", "-c", archive.REMOTE_VERIFY,
                                 str(stage), str(final), archive.sha256(manifest_path), "123"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["remote_host"], "test-host")
        self.assertEqual(report["remote_checkpoint"], str(final / "checkpoint-custom"))
        self.assertTrue(report["all_source_remote_sha256_match"])


if __name__ == "__main__":
    unittest.main()
