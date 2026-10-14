"""Fixed full-horizon sampling and honest paired metric regression checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_gr00t_pilot as evaluation


class SamplingTests(unittest.TestCase):
    def test_full_horizons_and_equal_episode_weight(self):
        episodes=[{'episode_index':i,'length':n} for i,n in enumerate([400,490,497,345,384])]
        samples=evaluation.sample_windows(episodes)
        self.assertEqual(len(samples),100)
        for i,episode in enumerate(episodes):
            starts=[x['start'] for x in samples if x['episode_index']==i]
            self.assertEqual(len(set(starts)),20)
            self.assertEqual(starts[0],0);self.assertEqual(starts[-1],episode['length']-16)
            self.assertTrue(all(start+15<episode['length'] for start in starts))
    def test_frozen_manifest_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'samples.json';evaluation.freeze_samples(path,{'hash':'original'})
            evaluation.freeze_samples(path,{'hash':'original'})
            with self.assertRaises(ValueError):evaluation.freeze_samples(path,{'hash':'changed'})


class MetricTests(unittest.TestCase):
    def setUp(self):
        try:import numpy as np
        except ImportError:self.skipTest('Numerical metric tests require the configured GR00T environment')
        self.np=np
    def fixture(self):
        np=self.np;target=np.zeros((2,16,26));state=np.zeros((2,26));target[0,:,0:7]=0.2
        prediction=np.repeat(target[:,None],3,axis=1)
        return prediction,target,state,np.zeros((16,26)),np.ones((16,26))
    def test_perfect_prediction_beats_persistence_on_target_defined_motion(self):
        pred,target,state,low,high=self.fixture()
        groups,moving,_=evaluation.summarize(pred,target,state,low,high)
        self.assertEqual(groups['left_arm']['all']['model']['mae'],0)
        self.assertGreater(groups['left_arm']['commanded_motion']['persistence']['mae'],0)
        self.assertEqual(moving[:,0].tolist(),[True,False])
        self.assertFalse(moving[:,1:].any())
    def test_regressions_unclipped_and_constants_not_hidden(self):
        pred,target,state,low,high=self.fixture();pred+=3;high[:,25]=0
        groups,moving,active=evaluation.summarize(pred,target,state,low,high)
        self.assertGreater(groups['left_arm']['all']['model']['normalized_mae'],2)
        self.assertEqual(groups['right_hand']['constant_horizon_joint_count'],16)
        self.assertAlmostEqual(groups['right_hand']['all']['model']['mae'],3)
        self.assertFalse(active[:,25].any())
    def test_nonfinite_predictions_fail(self):
        args=list(self.fixture());args[0][0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):evaluation.summarize(*args)
    def test_reference_identity_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'reference.json';path.write_text(json.dumps({'passed':True,'sample_manifest_sha256':'a','processor_semantics_sha256':'b'}))
            with self.assertRaises(ValueError):evaluation.verify_reference(path,'changed','b',None,None,None)


if __name__=='__main__':unittest.main()
