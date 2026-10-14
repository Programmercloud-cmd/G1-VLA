"""Bounded launch, evidence binding, and actual Trainer finite-check regressions."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gr00t_pilot as pilot


class RecipeTests(unittest.TestCase):
    def test_accumulation_changes_effective_batch_without_increasing_microbatch(self):
        runtime = {"interpreters": {"gr00t": Path("/python")}, "sources": {"gr00t": Path("/source")},
                   "models": {"nvidia/GR00T-N1.7-3B": {"snapshot": "/model"}}}
        command = pilot.pilot_command({}, runtime, Path("/data"), Path("/run"))
        for flag, value in [("--max-steps", "512"), ("--save-steps", "512"), ("--global-batch-size", "1"),
                            ("--gradient-accumulation-steps", "8"), ("--num-shards-per-epoch", "5"),
                            ("--save-total-limit", "1"), ("--state-dropout-prob", "0.2")]:
            self.assertEqual(command[command.index(flag) + 1], value)
        with self.assertRaises(ValueError): pilot.pilot_command({}, runtime, Path("/data"), Path("/run"), 513)

    def test_incomplete_or_reordered_audit_cannot_pass(self):
        records = []
        for step in range(2):
            for micro in range(2):
                records.append({"event": "micro_loss", "microbatch": 2*step+micro+1, "step_before": step,
                                "loss": 1.0, "finite": True})
            records.extend([{"event": "gradient", "step": step+1, "post_clip_norm": 0.5, "finite": True},
                            {"event": "step_end", "step": step+1}])
        self.assertTrue(pilot.finite_summary(records, 2, 2)["passed"])
        self.assertFalse(pilot.finite_summary(records[:-1], 2, 2)["passed"])
        broken = copy.deepcopy(records); broken[0]["loss"] = float("nan")
        self.assertFalse(pilot.finite_summary(broken, 2, 2)["passed"])
        broken = copy.deepcopy(records); broken[0]["microbatch"] = 2
        self.assertFalse(pilot.finite_summary(broken, 2, 2)["passed"])

    def test_exited_leader_does_not_skip_group_cleanup(self):
        process = Mock(pid=123, returncode=0)
        process.poll.return_value = 0
        with patch.object(pilot.os, "killpg", side_effect=[None, ProcessLookupError]) as kill:
            pilot.stop_process_group(process)
        self.assertEqual(kill.call_args_list[0].args, (123, signal.SIGTERM))
        process.wait.assert_called()

    def test_baseline_must_bind_samples_dataset_and_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); sample = root / "samples.json"; report = root / "baseline.json"
            arrays = report.with_suffix(".npz"); arrays.write_bytes(b"fixture")
            identity = {"dataset_files_sha256": {"data": "abc"}}
            sample.write_text(json.dumps(identity))
            record = {"passed": True, "role": "two_step_reference", "sample_manifest_sha256": pilot.digest(sample),
                      "dataset": str(root), "sample_count": 100, "inferences": 300,
                      "artifacts": {arrays.name: pilot.digest(arrays)}}
            report.write_text(json.dumps(record))
            with patch("evaluate_gr00t_pilot.dataset_identity", return_value=identity):
                self.assertEqual(pilot.verify_baseline(root, root / "pilot.json", sample, report), identity)
                record["sample_manifest_sha256"] = "mismatch"; report.write_text(json.dumps(record))
                with self.assertRaises(ValueError): pilot.verify_baseline(root, root / "pilot.json", sample, report)
                record["sample_manifest_sha256"] = pilot.digest(sample); report.write_text(json.dumps(record))
                arrays.write_bytes(b"changed")
                with self.assertRaises(ValueError): pilot.verify_baseline(root, root / "pilot.json", sample, report)


class ActualTrainerChecks(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from transformers import TrainingArguments
            from gr00t.experiment.trainer import Gr00tTrainer
        except ImportError:
            self.skipTest("Run this test with the configured GR00T interpreter for actual CPU Torch checks")
        self.torch, self.Arguments, self.Trainer = torch, TrainingArguments, Gr00tTrainer
        self.original = Gr00tTrainer.compute_loss
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.stream = pilot.install_finite_checks(self.root / "audit.jsonl")

    def tearDown(self):
        if hasattr(self, "stream"):
            self.stream.close(); self.Trainer.compute_loss = self.original; self.folder.cleanup()

    def trainer(self, mode="finite"):
        torch = self.torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, features):
                loss = (self.weight * features).square().mean()
                if mode == "loss_nan": loss = loss * torch.tensor(float("nan"))
                return {"loss": loss}
        model = Tiny()
        if mode == "gradient_nan": model.weight.register_hook(lambda gradient: gradient * float("nan"))
        args = self.Arguments(output_dir=str(self.root / "output"), use_cpu=True, max_steps=2,
            per_device_train_batch_size=1, gradient_accumulation_steps=2, save_strategy="no",
            logging_strategy="no", report_to=[], disable_tqdm=True, dataloader_num_workers=0)
        return self.Trainer(model=model, args=args, train_dataset=[{"features": torch.tensor([1.0])} for _ in range(4)])

    def test_real_trainer_records_each_microbatch_and_preupdate_gradient(self):
        trainer = self.trainer(); trainer.train(); self.stream.flush()
        report = pilot.finite_summary(pilot.read_audit(self.root / "audit.jsonl"), 2, 2)
        self.assertTrue(report["passed"], report)
        self.assertEqual(len([c for c in trainer.callback_handler.callbacks if c.__class__.__name__ == "FiniteGradients"]), 1)
        self.assertFalse(trainer.args.logging_nan_inf_filter)

    def test_nonfinite_loss_aborts_before_any_optimizer_update(self):
        trainer = self.trainer("loss_nan")
        with self.assertRaises(FloatingPointError): trainer.train()
        self.assertEqual(trainer.state.global_step, 0)
        self.assertEqual(pilot.read_audit(self.root / "audit.jsonl")[-1]["kind"], "microbatch loss")

    def test_nonfinite_gradient_aborts_before_any_optimizer_update(self):
        trainer = self.trainer("gradient_nan")
        with self.assertRaises(FloatingPointError): trainer.train()
        self.assertEqual(trainer.state.global_step, 0)
        self.assertEqual(pilot.read_audit(self.root / "audit.jsonl")[-1]["kind"], "post-clip gradient norm")


if __name__ == "__main__":
    unittest.main()
