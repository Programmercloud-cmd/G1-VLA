import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archive_smoke_checkpoint as archive


class PilotArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "pilot-fit-v1/checkpoint-512"
        self.source.mkdir(parents=True)
        self.samples = self.root / "samples.json"
        self.samples.write_text(json.dumps({"samples": [{"episode": 0, "frame": 0}]}))
        self.baseline = self.root / "baseline.json"
        self.evaluation = self.root / "evaluation.json"
        for report in [self.baseline, self.evaluation]:
            report.with_suffix(".npz").write_bytes(b"small predictions fixture")
        self.evaluation.with_suffix(".png").write_bytes(b"small trajectory fixture")
        self.baseline.write_text(json.dumps({"passed": True, "candidate_checkpoint": "baseline/checkpoint-2",
            "sample_manifest_sha256": archive.sha256(self.samples),
            "artifacts": {self.baseline.with_suffix(".npz").name: archive.sha256(self.baseline.with_suffix(".npz"))}}))
        self.evaluation.write_text(json.dumps({"passed": True, "candidate_checkpoint": str(self.source),
            "improved": False, "sample_manifest_sha256": archive.sha256(self.samples),
            "baseline_report_sha256": archive.sha256(self.baseline),
            "artifacts": {self.evaluation.with_suffix(suffix).name: archive.sha256(self.evaluation.with_suffix(suffix))
                          for suffix in [".npz", ".png"]}}))
        self.manifest = self.root / "snapshot.json"
        self.manifest.write_text(json.dumps({"episodes": [{"episode": "episode_0037"}]}))
        self.config = self.root / "run-config.json"
        self.config.write_text(json.dumps({"max_steps": 512, "gradient_accumulation_steps": 8,
            "checkpoint": str(self.source), "run": "pilot-fit-v1",
            "pilot_manifest_sha256": archive.sha256(self.manifest),
            "evaluation_manifest_sha256": archive.sha256(self.samples),
            "baseline_report_sha256": archive.sha256(self.baseline)}))

    def collect(self):
        return archive.pilot_evidence_sources(self.source, self.evaluation, self.config, self.manifest,
                                              self.baseline, self.samples)

    def test_complete_pilot_evidence_preserves_valid_unimproved_result(self):
        result = self.collect()
        self.assertEqual([relative for _, relative in result], ["reports/evaluation.json",
                         "reports/run-config.json", "dataset/snapshot-manifest.json",
                         "reports/baseline.json", "reports/sample-manifest.json", "reports/evaluation.npz",
                         "reports/evaluation.png", "reports/baseline.npz"])
        self.assertEqual(archive.pilot_evidence_sources(self.source), [])

    def test_mismatched_or_failed_evaluation_refuses_archive(self):
        for record in [{"passed": False, "candidate_checkpoint": str(self.source)},
                       {"passed": True, "candidate_checkpoint": str(self.root / "other")}]:
            self.evaluation.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "identify the checkpoint"):
                self.collect()

    def test_incomplete_or_unbounded_metadata_refuses_archive(self):
        with self.assertRaisesRegex(ValueError, "together"):
            archive.pilot_evidence_sources(self.source, self.evaluation)
        self.manifest.write_text('{"episodes":[]}')
        with self.assertRaisesRegex(ValueError, "snapshot episodes"):
            self.collect()
        with self.manifest.open("wb") as stream:
            stream.truncate(4 * 1024**2 + 1)
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            self.collect()

    def test_symlinked_evaluation_companion_refuses_archive(self):
        self.evaluation.with_suffix(".npz").unlink()
        self.evaluation.with_suffix(".npz").symlink_to(self.config)
        with self.assertRaisesRegex(ValueError, "ordinary files"):
            self.collect()

    def test_run_configuration_is_bound_to_checkpoint_and_run(self):
        for checkpoint, run in [(str(self.root / "wrong"), "pilot-fit-v1"), (str(self.source), "other-run")]:
            self.config.write_text(json.dumps({"checkpoint": checkpoint, "run": run}))
            with self.assertRaisesRegex(ValueError, "Run config must identify"):
                self.collect()

    def test_comparison_cannot_lose_or_replace_replay_evidence(self):
        self.baseline.with_suffix(".npz").unlink()
        with self.assertRaisesRegex(ValueError, "saved NPZ"):
            self.collect()
        self.baseline.with_suffix(".npz").write_bytes(b"fixture")
        self.samples.write_text('{"samples":[{"episode":1,"frame":2}]}')
        with self.assertRaisesRegex(ValueError, "input hash mismatch"):
            self.collect()

    def test_changed_prediction_cannot_be_archived_under_old_report(self):
        self.baseline.with_suffix(".npz").write_bytes(b"changed predictions")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self.collect()

    def test_changed_snapshot_cannot_replace_training_provenance(self):
        self.manifest.write_text('{"episodes":[{"episode":"different"}]}')
        with self.assertRaisesRegex(ValueError, "pilot_manifest_sha256"):
            self.collect()


if __name__ == "__main__":
    unittest.main()
