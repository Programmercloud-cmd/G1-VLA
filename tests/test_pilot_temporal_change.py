"""Ground-truth temporal change must not become a model-dependent selection."""
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import compare_pilot_temporal_change as temporal


class TemporalChangeTests(unittest.TestCase):
    def setUp(self):
        try:import numpy as np
        except ImportError:self.skipTest('Requires NumPy in the configured GR00T environment')
        self.np=np
    def arrays(self):
        np=self.np
        return np.zeros((2,16,26)),np.zeros((16,26)),np.ones((16,26))
    def test_constant_target_offset_is_not_temporal_change(self):
        target,low,high=self.arrays();target[:]=0.8
        mask,_,_=temporal.temporal_subset(target,low,high)
        self.assertFalse(mask.any())
    def test_threshold_and_maximum_horizon_span(self):
        np=self.np;target,low,high=self.arrays();target[0,:,0]=np.linspace(0,.08,16)
        high[15,0]=2
        mask,scores,_=temporal.temporal_subset(target,low,high)
        self.assertAlmostEqual(scores[0,0],.04);self.assertFalse(mask[0,0])
        high[15,0]=1;mask,_,_=temporal.temporal_subset(target,low,high)
        self.assertTrue(mask[0,0]);self.assertFalse(mask[1].any())
    def test_constant_normalizer_joint_excluded(self):
        np=self.np;target,low,high=self.arrays();target[0,:,0]=np.linspace(0,1,16);high[:,0]=0
        mask,_,valid=temporal.temporal_subset(target,low,high)
        self.assertFalse(valid[0]);self.assertFalse(mask.any())
    def test_candidate_cannot_create_or_change_declaration(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'frozen.json'
            with self.assertRaises(ValueError):temporal.freeze_declaration(path,{'threshold':.05},allow_create=False)
            temporal.freeze_declaration(path,{'threshold':.05},allow_create=True)
            with self.assertRaises(ValueError):temporal.freeze_declaration(path,{'threshold':.1},allow_create=False)
    def test_changed_paired_seed_or_target_rejected(self):
        np=self.np;report={'seeds':[42,43,44]}
        a={key:np.array([1,2]) for key in ['targets','states','seeds','episode_indices','starts','lower','upper']}
        b={key:value.copy() for key,value in a.items()};b['seeds'][0]=9
        with self.assertRaises(ValueError):temporal.assert_comparable(report,report,a,b)
        b={key:value.copy() for key,value in a.items()};b['targets'][0]=9
        with self.assertRaises(ValueError):temporal.assert_comparable(report,report,a,b)


if __name__=='__main__':unittest.main()
