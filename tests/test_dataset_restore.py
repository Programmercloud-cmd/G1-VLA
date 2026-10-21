"""Focused fixtures for the preserved larger-dataset-v1 archive restorer.

Only capacity is mocked; file hashes, extraction, fsync and Linux no-replace
publication run on real temporary files. Use Python without -O.
"""
import hashlib,importlib.util,io,json,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('restore',ROOT/'scripts/restore_larger_dataset_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(root,bad_hash=False,traversal=False):
 content=b'representative data\n';name='../escape' if traversal else 'snapshots/train/manifest.json'
 manifest={'files':[{'path':name,'size':len(content),'sha256':'0'*64 if bad_hash else hashlib.sha256(content).hexdigest()}],'total_bytes':len(content)}
 mp=root/'manifest.json';mp.write_text(json.dumps(manifest));archive=root/'dataset.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  data=mp.read_bytes();info=tarfile.TarInfo('archive_manifest.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
  info=tarfile.TarInfo(name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
 v={'passed':True,'all_member_sha256_match':True,'archive_manifest_report_fsync_completed':True,'compressed_sha256':m.sha(archive),'manifest_sha256':m.sha(mp),'compressed_bytes':archive.stat().st_size,'file_count':1,'total_bytes':len(content)}
 vp=root/'verification.json';vp.write_text(json.dumps(v));cp=root/'closeout.json';cp.write_text(json.dumps({'durable_archive':{'manifest':{'sha256':m.sha(mp)},'remote_verification':{'sha256':m.sha(vp)},'compressed_sha256':v['compressed_sha256'],'compressed_bytes':v['compressed_bytes'],'file_count':1,'uncompressed_bytes':len(content)}}))
 return dict(archive=archive,manifest=mp,verification=vp,closeout=cp,destination=root/'destination',report=root/'report.json')

class TestRestore(unittest.TestCase):
 def run_case(self,**kwargs):
  tmp=tempfile.TemporaryDirectory(prefix='vla-restore-fixture-');self.addCleanup(tmp.cleanup);args=fixture(Path(tmp.name),**kwargs)
  return args
 @patch.object(m.shutil,'disk_usage',return_value=type('Usage',(),{'free':16*1024**3})())
 def test_valid(self,_):
  args=self.run_case();r=m.restore(**args);self.assertTrue(r['passed']);self.assertEqual((args['destination']/'snapshots/train/manifest.json').read_bytes(),b'representative data\n')
 @patch.object(m.shutil,'disk_usage',return_value=type('Usage',(),{'free':16*1024**3})())
 def test_wrong_hash(self,_):
  args=self.run_case(bad_hash=True)
  with self.assertRaisesRegex(AssertionError,'Member hash'):m.restore(**args)
  self.assertFalse(args['destination'].exists())
 def test_traversal(self):
  args=self.run_case(traversal=True)
  with self.assertRaisesRegex(ValueError,'Unsafe'):m.restore(**args)
  self.assertFalse(args['destination'].exists())
 def test_no_overwrite(self):
  args=self.run_case();args['destination'].mkdir();(args['destination']/'preserve').write_text('original')
  with self.assertRaisesRegex(AssertionError,'occupied'):m.restore(**args)
  self.assertEqual((args['destination']/'preserve').read_text(),'original')
 @patch.object(m.shutil,'disk_usage',return_value=type('Usage',(),{'free':16*1024**3})())
 def test_wrong_compressed_hash(self,_):
  args=self.run_case();v=json.loads(args['verification'].read_text());v['compressed_sha256']='0'*64;args['verification'].write_text(json.dumps(v));c=json.loads(args['closeout'].read_text());c['durable_archive']['compressed_sha256']='0'*64;c['durable_archive']['remote_verification']['sha256']=m.sha(args['verification']);args['closeout'].write_text(json.dumps(c))
  with self.assertRaisesRegex(AssertionError,'Compressed'):m.restore(**args)
  self.assertFalse(args['destination'].exists())

if __name__=='__main__':unittest.main(verbosity=2)
