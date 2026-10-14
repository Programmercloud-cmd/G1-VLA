"""In-place checkpoint inventories preserve bytes and reject stale/partial evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import archive_gr00t_dataset as archive


class InPlaceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.source=self.root/'checkpoint-2048';self.source.mkdir()
        for name,value in {'config.json':{},'trainer_state.json':{'global_step':2048},
                           'model.safetensors.index.json':{'weight_map':{'weight':'model.safetensors'}},
                           'processor_config.json':{},'statistics.json':{},'embodiment_id.json':{}}.items():
            (self.source/name).write_text(json.dumps(value))
        (self.source/'model.safetensors').write_bytes(b'fake test weights')
        self.inventory=self.root/'inventory.json'

    def freeze(self):
        files=archive.hash_checkpoint_in_place(self.source)
        value={'schema_version':1,'artifact_type':'durable_checkpoint_inventory','passed':True,
               'checkpoint':str(self.source),'step':2048,'files':files,'per_file_fsync_completed':True}
        self.inventory.write_text(json.dumps(value));return value

    def test_hash_fsync_and_load_inventory_preserve_source_bytes(self):
        before={p.name:p.read_bytes() for p in self.source.iterdir()}
        value=self.freeze()
        self.assertEqual(archive.load_checkpoint_inventory(self.inventory,self.source),value)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.source.iterdir()})

    def test_extra_missing_and_symlinked_shards_are_rejected(self):
        self.freeze();(self.source/'extra.safetensors').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError,'index/shard'): archive.load_checkpoint_inventory(self.inventory,self.source)
        (self.source/'extra.safetensors').unlink();(self.source/'model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError,'index/shard'): archive.load_checkpoint_inventory(self.inventory,self.source)
        (self.source/'model.safetensors').symlink_to(self.inventory)
        with self.assertRaisesRegex(ValueError,'symlinks'): archive.load_checkpoint_inventory(self.inventory,self.source)

    def test_stale_weight_signature_and_same_size_metadata_tamper_are_rejected(self):
        self.freeze();path=self.source/'model.safetensors';path.write_bytes(b'new test weights')
        with self.assertRaisesRegex(ValueError,'signature'): archive.load_checkpoint_inventory(self.inventory,self.source)
        self.freeze();path=self.source/'config.json';stat=path.stat();path.write_text('[]')
        os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'metadata digest'): archive.load_checkpoint_inventory(self.inventory,self.source)

    def test_remote_hash_program_matches_local_without_network(self):
        expected=archive.hash_checkpoint_in_place(self.source)
        result=subprocess.check_output([sys.executable,'-c',archive.REMOTE_HASH,str(self.source)],text=True)
        self.assertEqual(json.loads(result),expected)

    def test_remote_subsecond_timestamp_is_separate_from_mounted_view(self):
        files=archive.checkpoint_files(self.source,2048)
        for path in files.values(): os.utime(path,ns=(1700000000000000000,1700000000000000000))
        records=archive.hash_checkpoint_in_place(self.source)
        for row in records:
            row['mtime_ns']+=123456789;row['device']+=10;row['inode']+=10
        archive.bind_local_view(files,records)
        archive.verify_inventory_records(files,records)
        self.assertTrue(all(row['mtime_ns']!=row['local_view_signature']['mtime_ns'] for row in records))
        path=self.source/'model.safetensors'
        os.utime(path,ns=(1700000000000000000,1700000000000000001))
        with self.assertRaisesRegex(ValueError,'signature'): archive.verify_inventory_records(files,records)
        records[0]['mtime_ns']+=10**9
        with self.assertRaisesRegex(ValueError,'size/time'): archive.bind_local_view(files,records)

    def test_remote_signature_check_detects_subsecond_change_hidden_by_mount(self):
        value=self.freeze();files=archive.checkpoint_files(self.source,2048)
        archive.bind_local_view(files,value['files'])
        value['remote_identity']={'host':'test-host','path':'/durable/checkpoint-2048'}
        self.inventory.write_text(json.dumps(value))
        signatures=[{key:row[key] for key in ['path','size','mtime_ns','device','inode']} for row in value['files']]
        with patch.object(archive,'remote_python',return_value=SimpleNamespace(stdout=json.dumps(signatures))) as remote:
            archive.load_checkpoint_inventory(self.inventory,self.source)
            remote.assert_called_once()
        signatures[0]['mtime_ns']+=1
        with patch.object(archive,'remote_python',return_value=SimpleNamespace(stdout=json.dumps(signatures))):
            with self.assertRaisesRegex(ValueError,'Remote checkpoint signature'):
                archive.load_checkpoint_inventory(self.inventory,self.source)

    def test_reload_proof_requires_strict_fresh_finite_single_load(self):
        proof={'fresh_process':True,'reload_proof':{'strict_tensor_loading':True,
            'model_load_count_in_process':1,'finite_predictions':True,'single_prediction_shape':[16,26],
            'saved_prediction_shape':[100,3,16,26],
            **{key:[] for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']}}}
        archive.validate_reload_proof(proof)
        for key,value in [('model_load_count_in_process',2),('finite_predictions',False),
                          ('single_prediction_shape',[1,26]),('missing_keys',['weight'])]:
            changed=copy.deepcopy(proof);changed['reload_proof'][key]=value
            with self.assertRaises(ValueError): archive.validate_reload_proof(changed)

    def test_complete_prefix_survives_later_failure_but_incomplete_or_nonfinite_prefix_does_not(self):
        records=[]
        for step in range(2):
            for micro in range(8):
                records.append({'event':'micro_loss','microbatch':step*8+micro+1,'step_before':step,'loss':1.,'finite':True})
            records.extend([{'event':'gradient','step':step+1,'post_clip_norm':.5,'finite':True},
                            {'event':'step_end','step':step+1}])
        records.append({'event':'checkpoint','step':2,'passed':True})
        records.append({'event':'nonfinite','kind':'later loss'})
        self.assertTrue(archive.prefix_integrity(records,2)[0]['passed'])
        changed=copy.deepcopy(records);changed.pop(0)
        with self.assertRaises(ValueError): archive.prefix_integrity(changed,2)
        changed=copy.deepcopy(records);changed.insert(0,{'event':'nonfinite','kind':'earlier loss'})
        with self.assertRaises(ValueError): archive.prefix_integrity(changed,2)


if __name__=='__main__': unittest.main()
