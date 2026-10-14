import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archive_smoke_checkpoint as archive


class FollowupArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "pilot-fit-no-state-dropout-v1/checkpoint-1024"
        self.source.mkdir(parents=True)
        self.protocol = self.root / "protocol.json"
        self.previous = self.root / "previous.json"
        self.evaluation = self.root / "evaluation.json"
        self.config = self.root / "run.json"
        self.comparison = self.root / "comparison.json"
        self.arrays = self.previous.with_suffix(".npz")
        self.arrays.write_bytes(b"saved checkpoint-512 prediction fixture")
        self.template = json.loads((archive.ROOT / "experiments/pilot-fit-no-state-dropout-v1/evaluation-protocol.json").read_text())
        self.shared = {
            "baseline_report_sha256": self.template["bindings"]["baseline_report"]["sha256"],
            "sample_manifest_sha256": self.template["bindings"]["sample_manifest"]["sha256"],
        }
        self.write(self.previous, {**self.shared, "passed": True, "checkpoint_global_step": 512,
            "candidate_checkpoint": "/previous/pilot-fit-v1/checkpoint-512",
            "artifacts": {self.arrays.name: archive.sha256(self.arrays)}})
        protocol = copy.deepcopy(self.template)
        protocol["bindings"]["previous_report"]["sha256"] = archive.sha256(self.previous)
        protocol["bindings"]["previous_arrays"]["sha256"] = archive.sha256(self.arrays)
        self.write(self.protocol, protocol)
        protocol_sha = archive.sha256(self.protocol)
        self.write(self.config, {"run": self.source.parent.name, "checkpoint": str(self.source),
            "max_steps": 1024, "state_dropout_prob": 0.0, "fresh_optimizer_and_scheduler": True,
            "evaluation_protocol_sha256": protocol_sha})
        self.write(self.evaluation, {**self.shared, "passed": True,
            "candidate_checkpoint": str(self.source), "evaluation_protocol_sha256": protocol_sha})
        self.write(self.comparison, {**self.shared, "passed": True, "acceptance_passed": False,
            "candidate_checkpoint": str(self.source), "evaluation_protocol_sha256": protocol_sha,
            "candidate_report_sha256": archive.sha256(self.evaluation),
            "previous_report_sha256": archive.sha256(self.previous),
            "previous_npz_sha256": archive.sha256(self.arrays)})

    @staticmethod
    def write(path, record):
        path.write_text(json.dumps(record))

    def update(self, path, **fields):
        record = json.loads(path.read_text())
        record.update(fields)
        self.write(path, record)

    def collect(self):
        return archive.followup_evidence_sources(self.source, self.evaluation, self.config,
                                                  self.comparison, self.protocol, self.previous)

    def test_archive_keeps_failed_fitting_gate_and_complete_replay_inputs(self):
        result = self.collect()
        self.assertEqual([relative for _, relative in result], [
            "reports/evaluation-protocol.json", "reports/previous-evaluation.json",
            "reports/previous-evaluation.npz", *sorted(self.template["training_only_implementation_bindings"])])
        self.assertEqual(result[2][0].read_bytes(), self.arrays.read_bytes())
        self.assertFalse(json.loads(self.comparison.read_text())["acceptance_passed"])
        self.assertFalse(any(path.suffix == ".safetensors" for path, _ in result))

    def test_original_archival_stays_optional_but_followup_cannot_drop_protocol(self):
        self.assertEqual(archive.followup_evidence_sources(self.source), [])
        old = self.root / "original-run.json"
        self.write(old, {"max_steps": 512})
        self.assertEqual(archive.followup_evidence_sources(self.source, run_config=old), [])
        with self.assertRaisesRegex(ValueError, "requires evaluation protocol and previous report"):
            archive.followup_evidence_sources(self.source, run_config=self.config)

    def test_partial_followup_inputs_are_rejected(self):
        for protocol, previous in [(self.protocol, None), (None, self.previous)]:
            with self.subTest(protocol=protocol, previous=previous):
                with self.assertRaisesRegex(ValueError, "complete pilot/temporal evidence"):
                    archive.followup_evidence_sources(self.source, self.evaluation, self.config,
                                                      self.comparison, protocol, previous)
        with self.assertRaisesRegex(ValueError, "complete pilot/temporal evidence"):
            archive.followup_evidence_sources(self.source, self.evaluation, self.config,
                                              None, self.protocol, self.previous)

    def test_every_consuming_report_must_bind_the_same_protocol(self):
        for path in [self.config, self.evaluation, self.comparison]:
            original = path.read_bytes()
            with self.subTest(path=path):
                self.update(path, evaluation_protocol_sha256="0" * 64)
                with self.assertRaisesRegex(ValueError, "protocol hash mismatch"):
                    self.collect()
            path.write_bytes(original)

    def test_changed_previous_json_or_predictions_cannot_be_archived(self):
        original = self.previous.read_bytes()
        self.previous.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "binding mismatch: previous_report"):
            self.collect()
        self.previous.write_bytes(original)
        self.arrays.write_bytes(b"replacement predictions")
        with self.assertRaisesRegex(ValueError, "binding mismatch: previous_arrays"):
            self.collect()

    def test_prediction_companion_must_be_present_ordinary_and_bounded(self):
        self.arrays.unlink()
        with self.assertRaisesRegex(ValueError, "ordinary NPZ"):
            self.collect()
        self.arrays.symlink_to(self.previous)
        with self.assertRaisesRegex(ValueError, "ordinary NPZ"):
            self.collect()
        self.arrays.unlink()
        with self.arrays.open("wb") as stream:
            stream.truncate(32 * 1024**2 + 1)
        with self.assertRaisesRegex(ValueError, "32 MiB"):
            self.collect()

    def test_protocol_is_bounded_and_its_fixed_semantics_are_validated(self):
        original = self.protocol.read_bytes()
        with self.protocol.open("wb") as stream:
            stream.truncate(4 * 1024**2 + 1)
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            self.collect()
        self.protocol.write_bytes(original)
        protocol = json.loads(original)
        protocol["allowed_semantic_change"]["candidate_value"] = 0.1
        self.write(self.protocol, protocol)
        with self.assertRaisesRegex(ValueError, "Unsupported or changed"):
            self.collect()

    def test_wrong_checkpoint_run_or_recipe_is_rejected(self):
        original = self.config.read_bytes()
        for fields in [{"max_steps": 512}, {"state_dropout_prob": 0.2},
                       {"fresh_optimizer_and_scheduler": False}, {"run": "other-run"}]:
            with self.subTest(fields=fields):
                self.update(self.config, **fields)
                with self.assertRaisesRegex(ValueError, "declared 1024-update"):
                    self.collect()
            self.config.write_bytes(original)
        self.update(self.evaluation, candidate_checkpoint=str(self.root / "another/checkpoint-1024"))
        with self.assertRaisesRegex(ValueError, "identify the archived follow-up"):
            self.collect()

    def test_failed_integrity_or_wrong_previous_step_is_rejected(self):
        original = self.previous.read_bytes()
        for fields in [{"passed": False}, {"checkpoint_global_step": 2}]:
            with self.subTest(fields=fields):
                self.update(self.previous, **fields)
                with self.assertRaisesRegex(ValueError, "integrity checks and reference checkpoint 512"):
                    self.collect()
            self.previous.write_bytes(original)
        self.update(self.comparison, passed=False)
        with self.assertRaisesRegex(ValueError, "integrity checks"):
            self.collect()

    def test_comparison_cannot_reference_other_previous_or_candidate_evidence(self):
        original = self.comparison.read_bytes()
        for field in ["previous_report_sha256", "previous_npz_sha256"]:
            self.update(self.comparison, **{field: "0" * 64})
            with self.assertRaisesRegex(ValueError, "Previous prediction evidence hash mismatch"):
                self.collect()
            self.comparison.write_bytes(original)
        self.update(self.comparison, candidate_report_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "different candidate evidence"):
            self.collect()

    def test_replay_must_use_the_frozen_baseline_and_samples(self):
        original = self.comparison.read_bytes()
        for field in self.shared:
            with self.subTest(field=field):
                self.update(self.comparison, **{field: "0" * 64})
                with self.assertRaisesRegex(ValueError, "reference binding mismatch"):
                    self.collect()
            self.comparison.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
