"""Bounded checkpoint-512 warm start, source integrity, and effective dropout checks."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gr00t_pilot as pilot


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoint-512"
        self.checkpoint.mkdir()
        payloads = {"config.json": {"state_dropout_prob": 0.2},
                    "processor_config.json": {"processor_kwargs": {"state_dropout_prob": 0.2}},
                    "statistics.json": {}, "embodiment_id.json": {},
                    "trainer_state.json": {"global_step": 512},
                    "model.safetensors.index.json": {"weight_map": {"weight": "model.safetensors"}}}
        for name, value in payloads.items():
            (self.checkpoint / name).write_text(json.dumps(value))
        (self.checkpoint / "model.safetensors").write_bytes(b"synthetic-weight-fixture")
        self.pilot_path = self.root / "pilot.json"
        self.pilot_path.write_text(json.dumps({"episodes": [1, 2, 3, 4, 5]}))
        self.identity = {"dataset_files_sha256": {"one.parquet": "fixture"}}
        reports = self.root / "reports"
        reports.mkdir()
        self.config = {"paths": {"reports": reports}}
        prior = {"run": "pilot-fit-v1", "max_steps": 512, "seed": 42, "microbatch": 1,
                 "accumulation": 8, **self.identity, "pilot_manifest_sha256": pilot.digest(self.pilot_path)}
        run_path = reports / "pilot-fit-v1-run.json"
        run_path.write_text(json.dumps(prior))
        records = [{"path": "checkpoint-512/" + path.name, "size": path.stat().st_size,
                    "sha256": pilot.digest(path)} for path in sorted(self.checkpoint.iterdir())]
        records.append({"path": "reports/run-config.json", "size": run_path.stat().st_size,
                        "sha256": pilot.digest(run_path)})
        archive_path = reports / "pilot-fit-v1-archive-manifest.json"
        archive_path.write_text(json.dumps({"checkpoint_directory": "checkpoint-512", "files": records}))
        (reports / "pilot-fit-v1-archive.json").write_text(json.dumps({
            "passed": True, "all_source_remote_sha256_match": True,
            "manifest_sha256": pilot.digest(archive_path), "files": records}))
        self.runtime = {"interpreters": {"gr00t": Path("/python")}, "sources": {"gr00t": Path("/source")},
                        "models": {"nvidia/GR00T-N1.7-3B": {"snapshot": "/model"}}}

    def binding(self):
        return pilot.bind_initial_checkpoint(self.config, self.checkpoint, self.identity, self.pilot_path)

    def test_warm_recipe_changes_only_model_steps_dropout_and_explicit_fresh_start(self):
        original = pilot.pilot_command({}, self.runtime, Path("/data"), Path("/run"))
        expected = original.copy()
        for flag, value in [("--base-model-path", str(self.checkpoint)), ("--max-steps", "1024"),
                            ("--save-steps", "1024"), ("--state-dropout-prob", "0.0")]:
            expected[expected.index(flag) + 1] = value
        expected.append("--no-resume-from-checkpoint")
        actual = pilot.pilot_command({}, self.runtime, Path("/data"), Path("/run"),
                                     1024, 8, self.checkpoint, 0.0)
        self.assertEqual(actual, expected)

    def test_unsupported_recipes_are_rejected(self):
        for steps, accumulation, initial, dropout in [
                (1024, 8, None, 0.2), (512, 8, None, 0.0), (1023, 8, self.checkpoint, 0.0),
                (1025, 8, self.checkpoint, 0.0), (1024, 4, self.checkpoint, 0.0),
                (1024, 8, self.checkpoint, 0.2), (1024, 8, self.checkpoint, float("nan"))]:
            with self.subTest(recipe=(steps, accumulation, initial, dropout)), self.assertRaises(ValueError):
                pilot.pilot_command({}, self.runtime, Path("/data"), Path("/run"),
                                    steps, accumulation, initial, dropout)

    def test_weights_are_hashed_once_in_worker_not_in_preflight(self):
        with patch.object(pilot, "digest", wraps=pilot.digest) as hashed:
            binding = self.binding()
            self.assertFalse(any(str(call.args[0]).endswith(".safetensors") for call in hashed.call_args_list))
            pilot.verify_initial_files(binding, hash_weights=True)
            pilot.verify_initial_files(binding, signatures_only=True)
            self.assertEqual(sum(str(call.args[0]).endswith(".safetensors") for call in hashed.call_args_list), 1)

    def test_changed_weight_is_rejected_even_when_stat_signature_is_preserved(self):
        binding = self.binding()
        path = self.checkpoint / "model.safetensors"
        stat = path.stat()
        path.write_bytes(b"x" * stat.st_size)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            pilot.verify_initial_files(binding, hash_weights=True)

    def test_changed_metadata_extra_file_or_symlink_is_rejected(self):
        binding = self.binding()
        extra = self.checkpoint / "extra.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "inventory"):
            pilot.verify_initial_files(binding)
        extra.unlink()
        extra.symlink_to(self.pilot_path)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            pilot.verify_initial_files(binding)
        extra.unlink()
        (self.checkpoint / "config.json").write_text('{"state_dropout_prob": 0.0}')
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.binding()

    def test_different_dataset_or_pilot_cannot_reuse_checkpoint(self):
        changed = {"dataset_files_sha256": {"one.parquet": "different"}}
        with self.assertRaisesRegex(ValueError, "exact original"):
            pilot.bind_initial_checkpoint(self.config, self.checkpoint, changed, self.pilot_path)
        self.pilot_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "exact original"):
            self.binding()

    def test_loaded_model_head_and_processor_dropout_must_all_be_zero(self):
        binding = self.binding()
        training = NS(max_steps=1024, global_batch_size=1, gradient_accumulation_steps=8,
                      learning_rate=0.0001, warmup_ratio=0.05, num_gpus=1,
                      resume_from_checkpoint=False, skip_weight_loading=False,
                      start_from_checkpoint=str(self.checkpoint))
        pipeline = NS(config=NS(training=training, data=NS(seed=42), model=NS(state_dropout_prob=0.0)),
                      model=NS(config=NS(state_dropout_prob=0.0), action_head=NS(state_dropout_prob=0.0)),
                      processor=NS(state_dropout_prob=0.0))
        result = pilot.check_warm_start_runtime(pipeline, binding)
        self.assertEqual(result["cumulative_updates"], 1536)
        for target in [pipeline.config.model, pipeline.model.config, pipeline.model.action_head, pipeline.processor]:
            target.state_dropout_prob = 0.2
            with self.assertRaisesRegex(ValueError, "both the action head and processor"):
                pilot.check_warm_start_runtime(pipeline, binding)
            target.state_dropout_prob = 0.0
        training.resume_from_checkpoint = True
        with self.assertRaisesRegex(ValueError, "recipe"):
            pilot.check_warm_start_runtime(pipeline, binding)


class ActualUpstreamOverrideTests(unittest.TestCase):
    def test_upstream_pipeline_loaders_override_checkpoint_dropout_without_loading_large_model(self):
        try:
            from gr00t.configs.base_config import get_default_config
            from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
            from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
            from gr00t.model.gr00t_n1d7 import setup
        except ImportError:
            self.skipTest("Use the configured GR00T interpreter for upstream loader checks")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Gr00tN1d7Config(state_dropout_prob=0.2).save_pretrained(root)
            (root / "processor_config.json").write_text(json.dumps({"processor_kwargs": {
                "state_dropout_prob": 0.2, "modality_configs": {}, "random_rotation_angle": 7}}))
            (root / "statistics.json").write_text("{}")

            class CapturedProcessor(Gr00tN1d7Processor):
                def __init__(self, **kwargs):
                    self.state_dropout_prob = kwargs["state_dropout_prob"]
                    self.random_rotation_angle = kwargs["random_rotation_angle"]

            model_calls = []

            def load_model(path, **kwargs):
                model_calls.append(kwargs)
                config = Gr00tN1d7Config.from_pretrained(path, state_dropout_prob=kwargs["state_dropout_prob"])
                return NS(config=config, action_head=NS(state_dropout_prob=config.state_dropout_prob),
                          parameters=lambda: [NS(numel=lambda: 1, requires_grad=True)]), {}

            config = get_default_config()
            config.training.start_from_checkpoint = str(root)
            config.model.state_dropout_prob = 0.0
            pipeline = setup.Gr00tN1d7Pipeline(config, root)
            fake_data = NS(get_dataset_statistics=lambda: {})
            with patch.object(setup.AutoModel, "from_pretrained", side_effect=load_model), patch.object(
                    setup.AutoProcessor, "from_pretrained", side_effect=CapturedProcessor.from_pretrained), patch.object(
                    setup.DatasetFactory, "build", return_value=(fake_data, None)):
                model = pipeline._create_model()
                pipeline._create_dataset(root)
            self.assertEqual(model.config.state_dropout_prob, 0.0)
            self.assertEqual(model.action_head.state_dropout_prob, 0.0)
            self.assertEqual(pipeline.processor.state_dropout_prob, 0.0)
            self.assertEqual(pipeline.processor.random_rotation_angle, 7)
            self.assertEqual(model_calls[0]["state_dropout_prob"], 0.0)
            self.assertEqual(json.loads((root / "processor_config.json").read_text())["processor_kwargs"]["state_dropout_prob"], 0.2)

    def test_warm_start_callback_records_empty_optimizer_before_first_update(self):
        try:
            import torch
            from transformers import TrainingArguments
            from gr00t.experiment.trainer import Gr00tTrainer
        except ImportError:
            self.skipTest("Use the configured GR00T interpreter for actual CPU Trainer checks")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original_loss, original_init = Gr00tTrainer.compute_loss, Gr00tTrainer.__init__
            stream = pilot.install_finite_checks(root / "audit.jsonl", warm_start=True)
            try:
                class Tiny(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.weight = torch.nn.Parameter(torch.tensor(1.0))
                    def forward(self, features):
                        return {"loss": (self.weight * features).square().mean()}
                args = TrainingArguments(output_dir=str(root / "output"), use_cpu=True, max_steps=1,
                    per_device_train_batch_size=1, save_strategy="no", logging_strategy="no",
                    report_to=[], disable_tqdm=True, dataloader_num_workers=0)
                trainer = Gr00tTrainer(model=Tiny(), args=args,
                                      train_dataset=[{"features": torch.tensor([1.0])}])
                trainer.train()
                stream.flush()
                rows = pilot.read_audit(root / "audit.jsonl")
                self.assertEqual(rows[0]["event"], "warm_start_optimizer")
                self.assertEqual(rows[0]["initial_step"], 0)
                self.assertEqual(rows[0]["optimizer_state_entries"], 0)
                self.assertIn(rows[0]["initial_scheduler_epoch"], (-1, 0))
                self.assertTrue(pilot.finite_summary(rows, 1, 1)["passed"])
            finally:
                stream.close()
                Gr00tTrainer.compute_loss, Gr00tTrainer.__init__ = original_loss, original_init


if __name__ == "__main__":
    unittest.main()
