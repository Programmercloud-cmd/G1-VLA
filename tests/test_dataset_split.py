"""Frozen split isolation, provenance and lossless goal-only snapshots."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_dataset_split as ds
import prepare_pilot as pilot
from project_config import load_config

GOAL='Pick up the black box {} using both hands and bring it to the centered carry position.'
LEFT='on the left at the front';RIGHT='on the right at the front';FRONT='at the front'


class SplitSelectionTests(unittest.TestCase):
    def records(self):
        names=ds.PILOT_EPISODES+[f'episode_{i:04d}' for i in range(1000,1117)]
        return [{'episode':name,'spatial_phrase':LEFT if i<76 else RIGHT if i<120 else FRONT,'group_tokens':[]} for i,name in enumerate(names)]

    def test_seeded_stratification_pilot_isolation_and_front_coverage(self):
        records=self.records();splits,grouping=ds.stratified_split(records)
        self.assertEqual(splits,ds.stratified_split(list(reversed(records)))[0])
        self.assertEqual((len(splits['train']),len(splits['validation'])),(98,24))
        self.assertTrue(set(ds.PILOT_EPISODES).issubset(splits['train']))
        self.assertFalse(set(splits['train'])&set(splits['validation']))
        counts=ds.Counter(r['spatial_phrase'] for r in records if r['episode'] in splits['validation'])
        self.assertEqual(counts,{LEFT:15,RIGHT:8,FRONT:1})
        self.assertEqual(grouping['basis'],'episode_only_no_genuine_group_metadata')

    def test_genuine_group_containing_pilot_remains_training(self):
        records=[{'episode':f'episode_{i:04d}','spatial_phrase':LEFT,'group_tokens':['info.session_id=a'] if i<3 else []} for i in range(1,7)]
        splits,_=ds.stratified_split(records,2,forced_train=['episode_0001'])
        self.assertTrue({'episode_0001','episode_0002'}.issubset(splits['train']))
        self.assertEqual(len(splits['validation']),2)

    def test_shared_date_and_copy_like_fields_do_not_create_sessions(self):
        self.assertEqual(ds.group_tokens({'info':{'date':'same','author':'unitree','copy_timestamp':42}}),[])
        self.assertEqual(ds.group_tokens({'info':{'session_id':'recorded'}}),['info.session_id=recorded'])


class SplitSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.raw=self.root/'raw'
        config=self.root/'machine.json';config.write_text(json.dumps({'paths':{'raw_data':str(self.raw),'derived':str(self.root/'derived'),'archive':str(self.root/'archive')}}));self.config=load_config(config)
        self.original={}
        for n in [1,2,3]:self.make_episode(n)
        entries=[]
        for n in [1,2,3]:
            name=f'episode_{n:04d}';entries.append({'episode':name,'source_data_json_sha256':ds.digest(self.raw/name/'data.json'),'original_goal':GOAL.format(LEFT),'status':'needs_review' if n==3 else 'accepted',**({'goal':GOAL.format(RIGHT)} if n!=3 else {})})
        self.labels=self.root/'labels.json';self.labels.write_text(json.dumps({'schema_version':1,'dataset_name':'raw','reference_frame':'workspace','episodes':entries}))
        self.inv=self.root/'inventory.json';ds.inventory(self.raw,self.labels,self.inv,expected_count=2)
        self.split=self.root/'split.json';ds.freeze_split(self.inv,self.split,expected_count=2,validation_count=1,forced_train=['episode_0001'])

    def make_episode(self,n):
        name=f'episode_{n:04d}';base=self.raw/name;(base/'colors').mkdir(parents=True)
        payload={'info':{'image':{'width':640,'height':480,'fps':30},'date':'shared'},'text':{'goal':GOAL.format(LEFT),'note':'preserve'},'data':[]}
        for i in range(16):
            relative=f'colors/{i:06d}_color_0.jpg';Image.new('RGB',(640,480),(i*10,n*10,90)).save(base/relative)
            parts={part:{'qpos':[0.1+i*0.001]*size} for part,size in pilot.PARTS}
            payload['data'].append({'idx':i,'colors':{'color_0':relative},'states':deepcopy(parts),'actions':deepcopy(parts)})
        (base/'data.json').write_text(json.dumps(payload));self.original[name]=(base/'data.json').read_bytes()

    def test_snapshot_changes_goal_only_and_preserves_raw_and_frozen_mapping(self):
        dest=self.root/'derived/train';manifest=ds.snapshot(self.split,'train',dest,self.config)
        self.assertEqual([r['episode'] for r in manifest['episodes']],['episode_0001'])
        expected=json.loads(self.original['episode_0001']);expected['text']['goal']=GOAL.format(RIGHT)
        self.assertEqual(json.loads((dest/'raw/pick_up_black_box/episode_0001/data.json').read_text()),expected)
        self.assertEqual((self.raw/'episode_0001/data.json').read_bytes(),self.original['episode_0001'])
        self.assertEqual((dest/'provenance/episode_0001.data.json').read_bytes(),self.original['episode_0001'])
        self.assertEqual(manifest['split_manifest_sha256'],ds.digest(self.split));pilot.validate_snapshot(dest)
        frozen=ds.load_split_manifest(self.split);self.assertNotIn('episode_0003',frozen['splits']['train']+frozen['splits']['validation'])
        with self.assertRaises(FileExistsError):ds.snapshot(self.split,'train',dest,self.config)

    def test_stale_source_json_fails_before_snapshot_creation(self):
        p=self.raw/'episode_0001/data.json';p.write_bytes(p.read_bytes()+b' ');dest=self.root/'derived/stale'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):ds.snapshot(self.split,'train',dest,self.config)
        self.assertFalse(dest.exists())

    def test_mutated_image_rejected_and_incomplete_output_not_publishable(self):
        p=self.raw/'episode_0001/colors/000000_color_0.jpg';Image.new('RGB',(640,480),'red').save(p);dest=self.root/'derived/image'
        with self.assertRaisesRegex(ValueError,'Source changed'):ds.snapshot(self.split,'train',dest,self.config)
        self.assertTrue((dest/'INCOMPLETE').exists())
        with self.assertRaises((ValueError,FileNotFoundError)):pilot.validate_snapshot(dest)

    def test_inventory_tampering_and_low_space_fail_closed(self):
        original=self.inv.read_bytes();self.inv.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'inventory hash'):ds.snapshot(self.split,'train',self.root/'derived/hash',self.config)
        self.inv.write_bytes(original)
        with patch.object(ds.shutil,'disk_usage',return_value=type('Usage',(),{'free':1024**3})()):
            with self.assertRaisesRegex(ValueError,'reserve'):ds.snapshot(self.split,'train',self.root/'derived/full',self.config)
        self.assertFalse((self.root/'derived/full').exists())

    def test_overlap_group_leakage_and_invalid_names_rejected(self):
        frozen=ds.load_split_manifest(self.split)
        bad=deepcopy(frozen);bad['splits']['validation']=['episode_0001']
        with self.assertRaises(ValueError):ds.validate_split(bad)
        bad=deepcopy(frozen);bad['grouping']['groups']=[['episode_0001','episode_0002']]
        with self.assertRaisesRegex(ValueError,'group crosses'):ds.validate_split(bad)
        bad=deepcopy(frozen);bad['splits']['validation']=['../episode_0002']
        with self.assertRaisesRegex(ValueError,'identifiers'):ds.validate_split(bad)

    def test_original_pilot_episode_limit_is_unchanged(self):
        with self.assertRaisesRegex(ValueError,'1–10'):pilot.prepare(11,'must-not-create',512,self.config)

    def test_local_archive_rejects_transient_shm_before_writing(self):
        config=deepcopy(self.config);config['paths']['archive']=Path('/dev/shm/vla-local-archive-fixture')
        manifest=self.root/'must-not-write-manifest.json';report=self.root/'must-not-write-report.json'
        with self.assertRaisesRegex(ValueError,'durable storage outside /dev/shm'):
            ds.archive_dataset(config['paths']['derived'],self.split,self.root/'unused-conversion.json',manifest,report,config)
        self.assertFalse(manifest.exists());self.assertFalse(report.exists())

    def test_local_archive_uses_same_receiver_without_ssh_and_preserves_sources(self):
        data_root=self.config['paths']['derived'];parts={}
        for role in ['train','validation']:
            snap=data_root/'snapshots'/role;ds.snapshot(self.split,role,snap,self.config)
            dataset=data_root/'converted'/role;dataset.mkdir(parents=True)
            (dataset/'payload.bin').write_bytes((role+' exact conversion fixture').encode())
            parts[role]={'dataset_path':str(dataset),'snapshot_manifest':ds.binding(snap/'manifest.json'),
                'files_sha256':{'payload.bin':ds.digest(dataset/'payload.bin')}}
        conversion=self.root/'conversion.json'
        conversion.write_text(json.dumps({'artifact_type':'larger_dataset_conversion',
            'split_manifest':ds.binding(self.split),'splits':parts}))
        before={str(p.relative_to(data_root)):ds.digest(p) for p in data_root.rglob('*') if p.is_file()}
        # Tiny fixture mocks capacity only; production retains its real 16 GiB preflight.
        receiver=ds.ARCHIVE_RECEIVER.replace("final=Path(sys.argv[1]);", "shutil.disk_usage=lambda path:type('Usage',(),{'free':32*1024**3})()\nfinal=Path(sys.argv[1]);")
        evidence=self.root/'archive-manifest.json';report_path=self.root/'archive-report.json'
        with patch.object(ds,'ARCHIVE_RECEIVER',receiver),patch.object(ds.subprocess,'Popen',wraps=ds.subprocess.Popen) as launched:
            report=ds.archive_dataset(data_root,self.split,conversion,evidence,report_path,self.config)
        self.assertEqual(launched.call_args.args[0][0],sys.executable)
        self.assertTrue(report['passed']);self.assertTrue(report['all_member_sha256_match'])
        self.assertEqual(report['manifest_sha256'],ds.digest(evidence))
        destination=self.config['paths']['archive']/'derived/larger-dataset-v1'
        self.assertTrue((destination/'dataset.tar.gz').is_file())
        self.assertEqual(before,{str(p.relative_to(data_root)):ds.digest(p) for p in data_root.rglob('*') if p.is_file()})
        with self.assertRaises(FileExistsError):
            ds.archive_dataset(data_root,self.split,conversion,evidence,report_path,self.config)

    def test_staged_pipeline_reuses_v3_and_runs_statistics_only_for_train(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        parent=self.config['paths']['derived']/'lerobot';snapshots={}
        for role in ['train','validation']:
            snap=self.config['paths']['derived']/'snapshots'/role;snapshots[role]=snap
            manifest=ds.snapshot(self.split,role,snap,self.config)
            dataset=parent/('local/g1_inspire_larger_v1_'+role);(dataset/'data/chunk-000').mkdir(parents=True);(dataset/'meta').mkdir()
            rows=json.loads((snap/'raw/pick_up_black_box'/manifest['episodes'][0]['episode']/'data.json').read_text())['data']
            table=pa.table({'observation.state':[pilot.vector(r,'states') for r in rows],'action':[pilot.vector(r,'actions') for r in rows],
                'episode_index':[0]*16,'frame_index':list(range(16)),'index':list(range(16)),'task_index':[0]*16,'timestamp':[i/30 for i in range(16)]})
            pq.write_table(table,dataset/'data/chunk-000/file-000.parquet')
            (dataset/'meta/info.json').write_text(json.dumps({'codebase_version':'v3.0','total_episodes':1}))
            pq.write_table(pa.table({'task_index':[0],'__index_level_0__':[GOAL.format(RIGHT)]}),dataset/'meta/tasks.parquet')
            provenance={'pilot_manifest':str(snap/'manifest.json'),'episodes':[{'episode':r['episode'],'content_sha256':r['content_sha256']} for r in manifest['episodes']],
                'camera':{'color_0':'cam_left_high'},'push_to_hub':False,
                'unitree_commit':json.loads((ds.PROJECT_ROOT/'locks/runtime-sources.json').read_text())['unitree_lerobot']['commit']}
            (dataset/'conversion_provenance.json').write_text(json.dumps(provenance))
        preparation=[]
        def fake_command(command,**kwargs):
            def arg(name):return Path(command[command.index(name)+1])
            if 'verify-rows' in command:
                ds.verify_rows(arg('--dataset'),arg('--snapshot-root'))
            elif 'convert_single_camera.py' in command[1]:
                self.assertIn('--to-v21',command,'Complete v3 must not be re-encoded')
                dataset=arg('--output-parent')/str(arg('--repo-id'));backup=dataset.with_name(dataset.name+'_v3.0')
                dataset.rename(backup);ds.shutil.copytree(backup,dataset)
                info=json.loads((dataset/'meta/info.json').read_text());info['codebase_version']='v2.1';(dataset/'meta/info.json').write_text(json.dumps(info))
                (dataset/'meta/tasks.jsonl').write_text(json.dumps({'task_index':0,'task':GOAL.format(RIGHT)})+'\n')
            elif 'prepare_gr00t_dataset.py' in command[1]:
                dataset=arg('--dataset');preparation.append((dataset.name,'--compute-stats' in command))
                ds.verify_rows(dataset,arg('--pilot-root'))
                if '--compute-stats' in command:
                    for name in ['stats.json','relative_stats.json']:(dataset/'meta'/name).write_text(json.dumps({'owner':'train'}))
                arg('--report').write_text(json.dumps({'fixture':'validator invocation only'}))
            else:self.fail(str(command))
        stage=self.config['paths']['archive']/'derived/staged';report=self.root/'conversion.json'
        with patch.object(ds.subprocess,'run',side_effect=fake_command):
            result=ds.convert(self.split,snapshots['train'],snapshots['validation'],parent,report,self.config,stage,True)
        self.assertEqual(preparation,[('g1_inspire_larger_v1_train',True),('g1_inspire_larger_v1_validation',False)])
        self.assertEqual(result['statistics']['owner'],'train')
        before=len(preparation)
        with patch.object(ds.subprocess,'run',side_effect=fake_command):
            ds.convert(self.split,snapshots['train'],snapshots['validation'],parent,self.root/'resumed-report.json',self.config,stage,True)
        self.assertEqual(len(preparation),before)
        for name,original in self.original.items():self.assertEqual((self.raw/name/'data.json').read_bytes(),original)

    def test_exact_row_verifier_detects_conversion_corruption(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        snap=self.root/'derived/verify';ds.snapshot(self.split,'train',snap,self.config)
        dataset=self.root/'derived/converted';(dataset/'data/chunk-000').mkdir(parents=True);(dataset/'meta').mkdir()
        rows=json.loads((snap/'raw/pick_up_black_box/episode_0001/data.json').read_text())['data']
        table=pa.table({'observation.state':[pilot.vector(r,'states') for r in rows],'action':[pilot.vector(r,'actions') for r in rows],
            'episode_index':[0]*16,'frame_index':list(range(16)),'index':list(range(16)),'task_index':[0]*16,'timestamp':[i/30 for i in range(16)]})
        path=dataset/'data/chunk-000/file-000.parquet';pq.write_table(table,path)
        (dataset/'meta/info.json').write_text(json.dumps({'codebase_version':'v3.0','total_episodes':1}));pq.write_table(pa.table({'task_index':[0],'__index_level_0__':[GOAL.format(RIGHT)]}),dataset/'meta/tasks.parquet')
        self.assertTrue(ds.verify_rows(dataset,snap)['passed'])
        bad=table.to_pydict();bad['action'][0][0]=0.9;pq.write_table(pa.table(bad),path)
        with self.assertRaisesRegex(ValueError,'state/action differs'):ds.verify_rows(dataset,snap)


class DatasetArchiveTests(unittest.TestCase):
    def make_tar(self,root,corrupt=False,unsafe=False):
        import hashlib,io,tarfile
        data=b'correct dataset bytes';name='../escape' if unsafe else 'train/data.bin'
        manifest={'files':[{'path':name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}
        encoded=(json.dumps(manifest)+'\n').encode();target=root/'stream.tar.gz'
        with tarfile.open(target,'w:gz') as stream:
            info=tarfile.TarInfo('archive_manifest.json');info.size=len(encoded);stream.addfile(info,io.BytesIO(encoded))
            payload=b'changed dataset bytes' if corrupt else data
            info=tarfile.TarInfo(name);info.size=len(payload);stream.addfile(info,io.BytesIO(payload))
        return target,hashlib.sha256(encoded).hexdigest()

    def receive(self,stream,destination,manifest_sha):
        import subprocess
        # Exercise the exact receiver locally; fixture bytes need no 16 GiB reservation.
        code=ds.ARCHIVE_RECEIVER.replace("final=Path(sys.argv[1]);", "shutil.disk_usage=lambda path:type('Usage',(),{'free':32*1024**3})()\nfinal=Path(sys.argv[1]);")
        with stream.open('rb') as source:
            return subprocess.run([sys.executable,'-c',code,str(destination),manifest_sha],stdin=source,capture_output=True,text=True)

    def test_verified_archive_published_once_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,sha=self.make_tar(root);original=source.read_bytes();dest=root/'archive'
            result=self.receive(source,dest,sha);self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(result.stdout);self.assertTrue(report['all_member_sha256_match']);self.assertTrue(report['archive_manifest_report_fsync_completed'])
            self.assertEqual(report['file_count'],1);self.assertTrue((dest/'dataset.tar.gz').exists());self.assertFalse(dest.with_name('archive.partial').exists());self.assertEqual(source.read_bytes(),original)
            self.assertNotEqual(self.receive(source,dest,sha).returncode,0)

    def test_corrupt_or_unsafe_stream_never_publishes_destination(self):
        for options in [{'corrupt':True},{'unsafe':True}]:
            with self.subTest(options=options),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source,sha=self.make_tar(root,**options);dest=root/'archive'
                result=self.receive(source,dest,sha);self.assertNotEqual(result.returncode,0);self.assertFalse(dest.exists());self.assertFalse((root/'escape').exists())


class StagedConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.dataset=self.root/'ram/local/dataset';self.staged=self.root/'external/local/dataset'
        for path,version in [(self.dataset,'v3.0'),(self.staged,'v2.1')]:
            (path/'meta').mkdir(parents=True);(path/'meta/info.json').write_text(json.dumps({'codebase_version':version}))
            (path/'data.bin').write_bytes((version+' exact bytes').encode())
        self.v3=ds.tree_inventory(self.dataset);self.v2=ds.tree_inventory(self.staged)
        self.identity={'split':'train','split_manifest_sha256':'a'*64,'snapshot_manifest_sha256':'b'*64}

    def test_unsupported_ram_location_fails_before_any_encoding(self):
        with patch.object(ds,'convert_direct') as encode:
            with self.assertRaisesRegex(ValueError,'--v21-staging-parent'):
                ds.convert(None,None,None,Path('/dev/shm/new-data'),None,{})
        encode.assert_not_called()

    def test_hash_copy_rejects_existing_output_and_detects_corruption(self):
        target=self.root/'copy';expected=ds.copy_tree_verified(self.staged,target)
        self.assertEqual(expected,self.v2)
        with self.assertRaises(FileExistsError):ds.copy_tree_verified(self.staged,target)
        corrupt=self.root/'corrupt';original=ds.shutil.copyfile
        def damaged(source,destination):
            original(source,destination);Path(destination).write_bytes(b'corrupt')
        with patch.object(ds.shutil,'copyfile',side_effect=damaged):
            with self.assertRaisesRegex(ValueError,'copy hash mismatch'):ds.copy_tree_verified(self.staged,corrupt)
        self.assertEqual(ds.tree_inventory(self.staged),self.v2)
        self.assertTrue(corrupt.exists())

    def test_publish_retains_original_partial_and_v3_without_overwrite(self):
        partial=self.dataset.with_name('dataset_v2.1');partial.mkdir();(partial/'partial.txt').write_text('failed first attempt')
        old=ds.tree_inventory(partial)
        retained,_=ds.publish_staged_v2(self.dataset,self.staged,self.v3,self.v2,self.identity)
        self.assertEqual(ds.tree_inventory(self.dataset),self.v2)
        self.assertEqual(ds.tree_inventory(self.dataset.with_name('dataset_v3.0')),self.v3)
        self.assertEqual(ds.tree_inventory(self.dataset.parent/'retained/dataset_v2.1'),old)
        self.assertEqual(len(retained),1);self.assertFalse(partial.exists())
        self.assertEqual(ds.tree_inventory(self.staged),self.v2)
        resumed,_=ds.publish_staged_v2(self.dataset,self.staged,self.v3,self.v2,self.identity)
        self.assertEqual(resumed,retained)
        (self.dataset/'data.bin').write_bytes(b'changed published copy')
        with self.assertRaisesRegex(ValueError,'inventory/hash mismatch'):
            ds.publish_staged_v2(self.dataset,self.staged,self.v3,self.v2,self.identity)

    def test_resume_after_first_rename_and_reject_changed_staging(self):
        ready=self.dataset.with_name('dataset_v2.1-verified');original=Path.rename
        def interrupted(path,target):
            if path==ready:raise OSError('simulated interruption between publication renames')
            return original(path,target)
        with patch.object(Path,'rename',interrupted):
            with self.assertRaisesRegex(OSError,'between publication'):ds.publish_staged_v2(self.dataset,self.staged,self.v3,self.v2,self.identity)
        self.assertFalse(self.dataset.exists());self.assertTrue(ready.exists())
        self.assertEqual(ds.tree_inventory(self.dataset.with_name('dataset_v3.0')),self.v3)
        (self.staged/'data.bin').write_bytes(b'changed staging')
        with self.assertRaisesRegex(ValueError,'inventory/hash mismatch'):ds.publish_staged_v2(self.dataset,self.staged,self.v3,self.v2,self.identity)
        (self.staged/'data.bin').write_bytes(b'v2.1 exact bytes')
        ds.publish_staged_v2(self.dataset,self.staged,self.v3,self.v2,self.identity)
        self.assertEqual(ds.tree_inventory(self.dataset),self.v2)

    def test_unowned_final_and_existing_backup_fail_closed(self):
        (self.dataset/'meta/info.json').write_text(json.dumps({'codebase_version':'v2.1'}))
        with self.assertRaisesRegex(ValueError,'ownership receipt'):ds.publish_staged_v2(self.dataset,self.staged,self.v3,self.v2,self.identity)
        (self.dataset/'meta/info.json').write_text(json.dumps({'codebase_version':'v3.0'}))
        backup=self.dataset.with_name('dataset_v3.0');ds.copy_tree_verified(self.dataset,backup)
        with self.assertRaises(FileExistsError):ds.publish_staged_v2(self.dataset,self.staged,self.v3,self.v2,self.identity)
        self.assertEqual(ds.tree_inventory(self.dataset),self.v3);self.assertEqual(ds.tree_inventory(backup),self.v3)
