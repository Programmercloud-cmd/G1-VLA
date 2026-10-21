#!/usr/bin/env python3
"""One-session guarded restoration; deliberately outside frozen pipeline sources."""
import argparse,ctypes,errno,hashlib,json,os,shutil,tarfile,time,uuid
from pathlib import Path,PurePosixPath

BLOCK=1024*1024

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(BLOCK),b''):h.update(b)
 return h.hexdigest()

def safe(name):
 p=PurePosixPath(name)
 if not name or p.is_absolute() or '..' in p.parts or str(p)!=name or '\\' in name:raise ValueError('Unsafe member path: '+repr(name))
 return p

class HashedReader:
 def __init__(self,raw):self.raw=raw;self.h=hashlib.sha256();self.n=0
 def read(self,size=-1):
  data=self.raw.read(size);self.h.update(data);self.n+=len(data);return data

def publish_noreplace(source,destination):
 libc=ctypes.CDLL(None,use_errno=True)
 fn=libc.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
 if fn(-100,os.fsencode(source),-100,os.fsencode(destination),1):
  e=ctypes.get_errno();raise OSError(e,os.strerror(e),str(destination))

def bind(archive,manifest,verification,closeout):
 c=json.loads(closeout.read_text());a=c['durable_archive'];v=json.loads(verification.read_text())
 assert sha(manifest)==a['manifest']['sha256'],'Manifest differs from frozen closeout'
 assert sha(verification)==a['remote_verification']['sha256'],'Verification differs from frozen closeout'
 assert v['passed'] and v['all_member_sha256_match'] and v['archive_manifest_report_fsync_completed']
 assert v['compressed_sha256']==a['compressed_sha256'] and v['manifest_sha256']==a['manifest']['sha256']
 assert v['compressed_bytes']==a['compressed_bytes']==archive.stat().st_size
 m=json.loads(manifest.read_text());expected={}
 for row in m['files']:
  name=str(safe(row['path']))
  assert name!='archive_manifest.json' and name not in expected,'Duplicate/reserved manifest member'
  assert isinstance(row['size'],int) and row['size']>=0
  expected[name]=row
 assert len(expected)==v['file_count']==a['file_count']
 assert sum(r['size'] for r in expected.values())==m['total_bytes']==v['total_bytes']==a['uncompressed_bytes']
 return c,a,v,expected

def verify_project_files(stage,project):
 mapping=json.loads((stage/'evaluation-evidence/manifest.json').read_text())
 assert len(mapping['files'])==5
 rows=[{'original_project_relative_path':'reports/larger-dataset-v1-conversion.json','archive_relative_path':'provenance/conversion.json','sha256':sha(stage/'provenance/conversion.json'),'bytes':(stage/'provenance/conversion.json').stat().st_size}]+mapping['files']
 results=[]
 for r in rows:
  rel=safe(r['original_project_relative_path']);src=stage/str(safe(r['archive_relative_path']));dst=project/str(rel)
  assert dst.resolve().is_relative_to(project.resolve()),'Project restore escapes checkout'
  assert src.stat().st_size==r['bytes'] and sha(src)==r['sha256']
  if dst.exists():assert not dst.is_symlink() and dst.stat().st_size==r['bytes'] and sha(dst)==r['sha256'],'Existing project file differs: '+str(dst)
  results.append((src,dst,r))
 return results

def restore(archive,manifest,verification,closeout,destination,report,project=None):
 assert not destination.exists() and not destination.is_symlink(),'Destination already occupied'
 assert not report.exists(),'Report already exists'
 c,a,v,expected=bind(archive,manifest,verification,closeout)
 assert shutil.disk_usage(destination.parent).free>=a['uncompressed_bytes']+8*1024**3,'Insufficient restore space/reserve'
 stage=destination.with_name(destination.name+'.restore-'+uuid.uuid4().hex)
 stage.mkdir(mode=0o700);print('RESTORE_STAGE',stage,flush=True)
 before=archive.stat();seen=set();total=0;started=time.time()
 with archive.open('rb') as raw:
  reader=HashedReader(raw)
  with tarfile.open(fileobj=reader,mode='r|gz') as stream:
   for member in stream:
    name=str(safe(member.name))
    assert member.isfile() and not member.issym() and not member.islnk(),'Nonregular tar member'
    assert name not in seen,'Duplicate tar member';seen.add(name)
    if name=='archive_manifest.json':
     assert len(seen)==1 and member.size==manifest.stat().st_size,'Leading manifest differs'
     data=stream.extractfile(member).read()
     assert hashlib.sha256(data).hexdigest()==v['manifest_sha256']
     continue
    assert name in expected,'Unlisted member: '+name
    row=expected[name];assert member.size==row['size'],'Member size differs'
    target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256();size=0
    with stream.extractfile(member) as src,target.open('xb') as out:
     for data in iter(lambda:src.read(BLOCK),b''):
      out.write(data);h.update(data);size+=len(data)
     out.flush();os.fsync(out.fileno())
    assert size==row['size'] and h.hexdigest()==row['sha256'],'Member hash differs: '+name
    total+=size
    if (len(seen)-1)%5000==0:print('MEMBERS',len(seen)-1,'BYTES',total,'SECONDS',round(time.time()-started,1),flush=True)
  while reader.read(BLOCK):pass
  assert reader.n==v['compressed_bytes'] and reader.h.hexdigest()==v['compressed_sha256'],'Compressed archive hash differs'
 after=archive.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Archive changed while reading'
 assert seen==set(expected)|{'archive_manifest.json'} and total==v['total_bytes'],'Incomplete restore'
 print('STREAM_VERIFIED',len(expected),total,flush=True)
 actual={str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
 assert actual==set(expected),'Restored inventory differs'
 for name,row in expected.items():
  p=stage/name;assert not p.is_symlink() and p.stat().st_size==row['size'] and sha(p)==row['sha256'],'Post-extract hash differs: '+name
 print('FULL_INVENTORY_VERIFIED',len(actual),flush=True)
 project_rows=verify_project_files(stage,project) if project else []
 for p in sorted((p for p in stage.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
  fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 publish_noreplace(stage,destination)
 fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 for src,dst,row in project_rows:
  if not dst.exists():
   dst.parent.mkdir(parents=True,exist_ok=True)
   with (destination/src.relative_to(stage)).open('rb') as inp,dst.open('xb') as out:shutil.copyfileobj(inp,out);out.flush();os.fsync(out.fileno())
  assert sha(dst)==row['sha256']
 result={'passed':True,'closeout_sha256':sha(closeout),'archive_manifest_sha256':sha(manifest),'compressed_archive_sha256':v['compressed_sha256'],'compressed_bytes':v['compressed_bytes'],'restored_file_count':len(expected),'restored_bytes':total,'destination':str(destination),'full_post_extract_inventory_sha256_verified':True,'atomic_no_replace_publication':True,'project_files_verified':len(project_rows),'old_partial_staging_preserved':True,'elapsed_seconds':round(time.time()-started,2)}
 with report.open('x') as out:json.dump(result,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
 print(json.dumps(result,indent=2),flush=True)
 return result

def main():
 p=argparse.ArgumentParser()
 for k in ['archive','manifest','verification','closeout','destination','report']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--project',type=Path)
 a=p.parse_args();restore(**vars(a))
if __name__=='__main__':main()
