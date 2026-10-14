#!/usr/bin/env python3
"""Freeze the reviewed larger dataset, then build separate immutable split snapshots."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex
import tarfile
import shutil
import subprocess
import sys

from prepare_pilot import (checked_source_json, combined_digest, contained_file, digest,
                           inspect_episode, now, overridden_json, parse_label_overrides,
                           validate_snapshot, vector)
from project_config import PROJECT_ROOT, guard_derived, load_config, load_runtime, resolve_path, runtime_environment

PILOT_EPISODES = ['episode_0037', 'episode_0040', 'episode_0042', 'episode_0046', 'episode_0047']
HOLDS = ['episode_0092', 'episode_0136', 'episode_0165']
GROUP_FIELDS = ('session_id', 'collection_session_id', 'recording_session_id', 'layout_id', 'scene_id')
MAX_SNAPSHOT_MIB = 16384


def ref(path):
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def binding(path):
    return {'path': ref(path), 'sha256': digest(Path(path))}


def write_new(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing existing output: {path}')
    content = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if shutil.disk_usage(PROJECT_ROOT).free < 8 * 1024**3 + len(content):
        raise ValueError('Project reserve would fall below 8 GiB')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)


def spatial_phrase(goal):
    prefix = 'Pick up the black box '
    suffix = ' using both hands and bring it to the centered carry position.'
    if not goal.startswith(prefix) or not goal.endswith(suffix):
        raise ValueError('Unexpected reviewed task goal')
    phrase = goal[len(prefix):-len(suffix)]
    if phrase not in ('on the left at the front', 'on the right at the front', 'at the front'):
        raise ValueError('Unexpected workspace label')
    return phrase


def group_tokens(payload):
    """Only explicit recorded identifiers count; date/order/mtime never imply groups."""
    tokens = []
    for container in ('info', 'text'):
        for key in GROUP_FIELDS:
            value = payload.get(container, {}).get(key)
            if value is not None and value != '':
                if not isinstance(value, (str, int)) or isinstance(value, bool):
                    raise ValueError('Invalid explicit collection group identifier')
                tokens.append(f'{container}.{key}={value}')
    return sorted(tokens)


def inventory(raw, overlay, output, expected_count=122):
    raw, overlay = Path(raw).resolve(), Path(overlay).resolve()
    if Path(output).resolve().is_relative_to(raw):
        raise ValueError('Inventory output overlaps original recordings')
    label_content = overlay.read_bytes()
    labels, entries = parse_label_overrides(label_content, raw.name)
    accepted = sorted(name for name, entry in entries.items() if entry['status'] == 'accepted')
    if len(accepted) != expected_count:
        raise ValueError('Reviewed accepted episode count differs from the bounded inventory')
    source_json = {}
    for name, entry in entries.items():
        source_json[name] = checked_source_json(raw, name, entry)
    records = []
    for name in accepted:
        record = inspect_episode(raw / name)
        entry = entries[name]
        if record['files_sha256']['data.json'] != entry['source_data_json_sha256'] or record['goal'] != entry['original_goal']:
            raise ValueError(f'Stale reviewed source: {name}')
        payload = json.loads(source_json[name])
        record.update(original_goal=entry['original_goal'], effective_goal=entry['goal'],
                      spatial_phrase=spatial_phrase(entry['goal']), group_tokens=group_tokens(payload))
        records.append(record)
        print(json.dumps({'inventory_episode': name, 'rows': record['rows'], 'bytes': record['bytes']}), flush=True)
    # Bind the exact reviewed set. Later unreviewed arrivals cannot enter this freeze.
    for name, entry in entries.items():
        checked_source_json(raw, name, entry)
    if digest(overlay) != hashlib.sha256(label_content).hexdigest():
        raise ValueError('Label overlay changed during inventory')
    value = {'schema_version': 1, 'artifact_type': 'reviewed_dataset_inventory', 'created_at': now(),
             'source_root': str(raw), 'dataset_name': raw.name, 'label_overrides': binding(overlay),
             'reviewed_source_json_sha256': {name: entry['source_data_json_sha256'] for name, entry in entries.items()},
             'held_episodes': sorted(set(entries) - set(accepted)),
             'unreviewed_folders_observed': sorted(p.name for p in raw.glob('episode_*') if p.name not in entries),
             'episodes': records, 'total_rows': sum(x['rows'] for x in records),
             'total_source_bytes': sum(x['bytes'] for x in records),
             'grouping_inspection': {'recognized_fields': list(GROUP_FIELDS),
                 'episodes_with_explicit_identifiers': sum(bool(x['group_tokens']) for x in records),
                 'ignored_as_group_evidence': ['shared calendar date', 'folder order', 'copy timestamps']},
             'copy_completion': 'Reviewed local JSONs and all accepted referenced images verified; original remote inventory and physical copy completion are not independently established by this scan.'}
    write_new(output, value)
    return value


def joined_groups(records):
    parent = {r['episode']: r['episode'] for r in records}
    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]; name = parent[name]
        return name
    first = {}
    for row in records:
        for token in row.get('group_tokens', []):
            if token in first:
                parent[find(row['episode'])] = find(first[token])
            else:
                first[token] = row['episode']
    result = defaultdict(list)
    for row in records:
        result[find(row['episode'])].append(row)
    return [sorted(group, key=lambda r: r['episode']) for _, group in sorted(result.items())]


def stratified_split(records, validation_count=24, seed=42, forced_train=PILOT_EPISODES):
    names = [r['episode'] for r in records]
    if len(names) != len(set(names)) or not 0 < validation_count < len(names):
        raise ValueError('Invalid episode inventory or validation count')
    if not set(forced_train).issubset(names):
        raise ValueError('The previously used pilot episodes must be present')
    groups = joined_groups(records)
    forced_groups = [g for g in groups if set(forced_train).intersection(r['episode'] for r in g)]
    forced = {r['episode'] for g in forced_groups for r in g}
    counts = Counter(r['spatial_phrase'] for r in records)
    available = Counter(r['spatial_phrase'] for r in records if r['episode'] not in forced)
    ideal = {key: validation_count * count / len(records) for key, count in counts.items()}
    quotas = {key: min(available[key], max(1 if count > 1 else 0, math.floor(ideal[key]))) for key, count in counts.items()}
    while sum(quotas.values()) < validation_count:
        choices = [k for k in counts if quotas[k] < available[k] and quotas[k] < counts[k]-1]
        if not choices: break
        key = max(sorted(choices), key=lambda k: ideal[k]-quotas[k]); quotas[key] += 1
    while sum(quotas.values()) > validation_count:
        choices = [k for k in counts if quotas[k] > 1]
        if not choices: break
        key = max(sorted(choices), key=lambda k: quotas[k]-ideal[k]); quotas[key] -= 1
    rng = random.Random(seed); selected = set()
    if all(len(g) == 1 for g in groups):
        for key in sorted(counts):
            candidates = sorted(r['episode'] for r in records if r['spatial_phrase'] == key and r['episode'] not in forced)
            rng.shuffle(candidates); selected.update(candidates[:quotas[key]])
    else:
        candidates = [g for g in groups if not forced.intersection(r['episode'] for r in g)]
        rng.shuffle(candidates)
        def cost(chosen):
            actual = Counter(r['spatial_phrase'] for r in records if r['episode'] in chosen)
            return 100*(len(chosen)-validation_count)**2 + sum((actual[k]-quotas[k])**2 for k in counts)
        while candidates:
            index = min(range(len(candidates)), key=lambda i: cost(selected | {r['episode'] for r in candidates[i]}))
            proposed = selected | {r['episode'] for r in candidates[index]}
            if cost(proposed) >= cost(selected): break
            selected = proposed; candidates.pop(index)
    if not selected or selected == set(names):
        raise ValueError('Genuine grouping prevents a nonempty whole-group validation split')
    return {'train': sorted(set(names)-selected), 'validation': sorted(selected)}, {
        'basis': 'explicit_recorded_identifiers' if any(r.get('group_tokens') for r in records) else 'episode_only_no_genuine_group_metadata',
        'groups': [[r['episode'] for r in g] for g in groups], 'forced_training_episodes': sorted(forced),
        'target_validation_count': validation_count, 'actual_validation_count': len(selected),
        'target_stratum_counts': dict(quotas),
        'method': 'Seeded per-stratum shuffle for singleton episodes; deterministic whole-group greedy target-distance selection otherwise; groups are never split.',
        'limitation': 'An episode-only split does not establish generalization to new collection sessions or layouts.'}


def freeze_split(inventory_path, output, copy_report=None, expected_count=122, validation_count=24, forced_train=PILOT_EPISODES):
    path = Path(inventory_path); source = json.loads(path.read_text())
    records = source['episodes']
    if Path(output).resolve().is_relative_to(Path(source['source_root']).resolve()):
        raise ValueError('Split output overlaps original recordings')
    if source.get('artifact_type') != 'reviewed_dataset_inventory' or len(records) != expected_count:
        raise ValueError('Unexpected source inventory')
    overlay = resolve_path(source['label_overrides']['path'])
    if digest(overlay) != source['label_overrides']['sha256']:
        raise ValueError('Label overlay differs from inventory')
    labels, entries = parse_label_overrides(overlay.read_bytes(), source['dataset_name'])
    accepted = {name for name, entry in entries.items() if entry['status'] == 'accepted'}
    if accepted != {r['episode'] for r in records}:
        raise ValueError('Accepted inventory differs from overlay')
    for record in records:
        checked_source_json(Path(source['source_root']), record['episode'], entries[record['episode']])
        if combined_digest(record['files_sha256']) != record['content_sha256']:
            raise ValueError('Invalid source file inventory digest')
    splits, grouping = stratified_split(records, validation_count, 42, forced_train)
    episodes = []
    for split, names in splits.items():
        for index, name in enumerate(names):
            r = next(r for r in records if r['episode'] == name)
            episodes.append({'episode': name, 'split': split, 'derived_episode_index': index,
                **{k:r[k] for k in ['rows','source_idx_start','source_idx_end','original_goal','effective_goal','spatial_phrase']},
                'source_data_json_sha256': r['files_sha256']['data.json'],
                'source_content_sha256': r['content_sha256'], 'source_bytes': r['bytes']})
    value = {'schema_version': 1, 'artifact_type': 'frozen_dataset_split',
             'status': 'frozen_before_conversion_and_statistics', 'created_at': now(), 'seed': 42,
             'label_overrides': {**source['label_overrides'], 'dataset_name': labels['dataset_name'], 'reference_frame': labels['reference_frame']},
             'source_inventory': binding(path), 'splits': splits, 'episodes': episodes,
             'pilot_episodes': list(forced_train), 'held_episodes': source['held_episodes'], 'grouping': grouping,
             'copy_completion': {'assessment': source['copy_completion'], 'additional_evidence': binding(copy_report) if copy_report else None},
             'summary': {split:{'episodes':len(names),'rows':sum(r['rows'] for r in episodes if r['split']==split),
                 'spatial_counts':dict(Counter(r['spatial_phrase'] for r in episodes if r['split']==split))} for split,names in splits.items()}}
    validate_split(value)
    write_new(output, value)
    return value


def validate_split(value):
    if value.get('schema_version') != 1 or value.get('artifact_type') != 'frozen_dataset_split' or value.get('seed') != 42:
        raise ValueError('Unexpected frozen split contract')
    splits, records = value['splits'], value['episodes']
    if set(splits) != {'train','validation'} or not all(splits.values()):
        raise ValueError('Both split lists must be nonempty')
    names = splits['train'] + splits['validation']
    if len(names) > 122 or any(not isinstance(n,str) or not re.fullmatch(r'episode_[0-9]{4}',n) for n in names):
        raise ValueError('Episode identifiers or bounded split size are invalid')
    if len(names) != len(set(names)) or len(records) != len(names) or {r['episode'] for r in records} != set(names):
        raise ValueError('Duplicate, overlapping or unlisted split episodes')
    if not set(value['pilot_episodes']).issubset(splits['train']) or set(value['held_episodes']).intersection(names):
        raise ValueError('Pilot leakage or held episode admitted')
    for split, ordered in splits.items():
        if ordered != sorted(ordered):raise ValueError('Split episode ordering must be frozen and sorted')
        for index,name in enumerate(ordered):
            r=next(r for r in records if r['episode']==name)
            if r['split']!=split or r['derived_episode_index']!=index or r['rows']<16:
                raise ValueError('Invalid split-local episode mapping or horizon length')
    for r in records:
        spatial_phrase(r['effective_goal'])
        if any(not re.fullmatch(r'[0-9a-f]{64}',r[k]) for k in ['source_data_json_sha256','source_content_sha256']):
            raise ValueError('Invalid source hash binding')
    for group in value['grouping']['groups']:
        if set(group).intersection(splits['train']) and set(group).intersection(splits['validation']):
            raise ValueError('Genuine group crosses splits')
    return value


def load_split_manifest(path):
    return validate_split(json.loads(Path(path).read_text()))


def capacity(output, planned_bytes, config):
    ancestor=Path(output)
    while not ancestor.exists():ancestor=ancestor.parent
    reserve=max(8,config.get('reserve_gib',8))*1024**3
    project_extra=planned_bytes if ancestor.stat().st_dev==PROJECT_ROOT.stat().st_dev else 0
    if shutil.disk_usage(ancestor).free < planned_bytes+reserve or shutil.disk_usage(PROJECT_ROOT).free < project_extra+reserve:
        raise ValueError('Insufficient output capacity or 8 GiB project reserve')


def snapshot(split_path, split, output, config, max_mib=MAX_SNAPSHOT_MIB):
    if split not in ('train','validation') or not 1<=max_mib<=MAX_SNAPSHOT_MIB:
        raise ValueError('Snapshot split/size is outside the bounded recipe')
    frozen=load_split_manifest(split_path);output=guard_derived(output,config)
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    inventory_path=resolve_path(frozen['source_inventory']['path'])
    if digest(inventory_path)!=frozen['source_inventory']['sha256']:raise ValueError('Frozen inventory hash mismatch')
    inventory_data=json.loads(inventory_path.read_text());raw=Path(inventory_data['source_root']).resolve()
    if output.is_relative_to(raw) or raw.is_relative_to(output):raise ValueError('Snapshot overlaps original recordings')
    overlay_path=resolve_path(frozen['label_overrides']['path']);label_content=overlay_path.read_bytes()
    if hashlib.sha256(label_content).hexdigest()!=frozen['label_overrides']['sha256']:raise ValueError('Frozen label overlay hash mismatch')
    labels,entries=parse_label_overrides(label_content,raw.name)
    records=[];contents={};total=0
    for index,name in enumerate(frozen['splits'][split]):
        record=deepcopy(next(r for r in inventory_data['episodes'] if r['episode']==name));declared=next(r for r in frozen['episodes'] if r['episode']==name)
        if record['content_sha256']!=declared['source_content_sha256']:raise ValueError('Split/source content binding mismatch')
        content=checked_source_json(raw,name,entries[name]);replacement=overridden_json(content,entries[name]['goal']);contents[name]=(content,replacement)
        if entries[name]['status']!='accepted' or entries[name]['goal']!=declared['effective_goal']:raise ValueError('Unaccepted or altered split goal')
        record.update(source_data_json_sha256=record['files_sha256']['data.json'],source_files_sha256=record['files_sha256'].copy(),
            source_content_sha256=record['content_sha256'],source_bytes=record['bytes'],original_goal=entries[name]['original_goal'],
            effective_goal=entries[name]['goal'],label_manifest_sha256=frozen['label_overrides']['sha256'],goal_status='accepted',derived_episode_index=index)
        record['bytes']+=len(replacement)-len(content);total+=record['bytes']+len(content);records.append(record)
    planned=total+len(label_content)+4*len(json.dumps(records).encode())+1024**2
    if planned>max_mib*1024**2:raise ValueError('Explicit snapshot byte limit exceeded')
    capacity(output,planned,config);output.mkdir(parents=True);(output/'INCOMPLETE').write_text('Snapshot not yet validated.\n');(output/'provenance').mkdir();(output/'label-overrides.json').write_bytes(label_content)
    for record in records:
        name=record['episode'];destination=output/'raw/pick_up_black_box'/name
        for relative,expected in record['source_files_sha256'].items():
            source=contained_file(raw/name,relative)
            if digest(source)!=expected:raise ValueError(f'Source changed after freeze: {name}/{relative}')
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            if digest(target)!=expected or digest(source)!=expected:raise ValueError('Snapshot copy verification failed')
        original,replacement=contents[name];(output/'provenance'/f'{name}.data.json').write_bytes(original);(destination/'data.json').write_bytes(replacement)
        record['goal']=record['effective_goal'];record['files_sha256']['data.json']=digest(destination/'data.json');record['content_sha256']=combined_digest(record['files_sha256'])
        print(json.dumps({'snapshot':split,'episode':name,'rows':record['rows']}),flush=True)
    manifest={'schema_version':1,'artifact_type':'frozen_split_snapshot','created_at':now(),'split':split,'split_manifest':binding(split_path),
              'split_manifest_sha256':digest(Path(split_path)),'source_inventory_provisional':True,'source_root':str(raw),
              'source_folders_observed':len(inventory_data['reviewed_source_json_sha256']),'requested_count':len(records),'max_snapshot_mib':max_mib,
              'snapshot_bytes':sum(r['bytes'] for r in records),'excluded':frozen['held_episodes'],'episodes':records,
              'label_overrides':{'source_path':str(overlay_path),'sha256':frozen['label_overrides']['sha256'],'dataset_name':labels['dataset_name'],'reference_frame':labels['reference_frame']},
              'split_status':'Frozen whole-episode train/validation split; validation excluded from training normalization.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Validate before publishing completion; restore the marker on any failure.
    (output/'INCOMPLETE').unlink()
    try:validate_snapshot(output)
    except Exception:
        (output/'INCOMPLETE').write_text('Final snapshot verification failed.\n');raise
    return manifest


def verify_rows(dataset,snapshot_root):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    root=Path(dataset);snapshot_root=Path(snapshot_root);manifest=json.loads((snapshot_root/'manifest.json').read_text());info=json.loads((root/'meta/info.json').read_text())
    tables=[pq.read_table(p) for p in sorted((root/'data').glob('*/*.parquet'))];table=pa.concat_tables(tables)
    if info['total_episodes']!=len(manifest['episodes']) or table.num_rows!=sum(r['rows'] for r in manifest['episodes']):raise ValueError('Converted inventory mismatch')
    if info['codebase_version']=='v3.0':
        tasks={r['task_index']:r['__index_level_0__'] for r in pq.read_table(root/'meta/tasks.parquet').to_pylist()}
    else:tasks={r['task_index']:r['task'] for r in map(json.loads,(root/'meta/tasks.jsonl').read_text().splitlines())}
    if set(table['episode_index'].to_pylist())!=set(range(len(manifest['episodes']))):raise ValueError('Converted episode indices differ')
    running=0
    for record in manifest['episodes']:
        rows=json.loads((snapshot_root/'raw/pick_up_black_box'/record['episode']/'data.json').read_text())['data'];part=table.filter(pc.equal(table['episode_index'],record['derived_episode_index']))
        if part.num_rows!=len(rows) or part['frame_index'].to_pylist()!=list(range(len(rows))) or part['index'].to_pylist()!=list(range(running,running+len(rows))):raise ValueError('Converted row ordering changed')
        for field,column in [('states','observation.state'),('actions','action')]:
            actual=np.asarray(part[column].to_pylist(),dtype=np.float32);expected=np.asarray([vector(r,field) for r in rows],dtype=np.float32)
            if not np.isfinite(actual).all() or not np.array_equal(actual,expected):raise ValueError('Converted state/action differs from source float32')
        if any(tasks[t]!=record['goal'] for t in part['task_index'].to_pylist()):raise ValueError('Converted task goal differs')
        if not np.allclose(np.asarray(part['timestamp']),np.arange(len(rows))/30,rtol=0,atol=2e-6):raise ValueError('Converted nominal times differ')
        running+=len(rows)
    return {'passed':True,'format':info['codebase_version'],'episodes':len(manifest['episodes']),'rows':running,'float32_state_action_exact':True,'task_text_exact':True,'row_order_exact':True}


def convert(split_path,train_snapshot,validation_snapshot,output_parent,report_path,config):
    frozen=load_split_manifest(split_path);parent=guard_derived(output_parent,config);runtime=load_runtime(config)
    if Path(report_path).exists():raise FileExistsError(report_path)
    snapshots={'train':guard_derived(train_snapshot,config),'validation':guard_derived(validation_snapshot,config)}
    total=sum(r['source_bytes'] for r in frozen['episodes'])
    # Pilot measured 45,799,402 bytes for both formats /183,940,398 snapshot bytes.
    # Budget twice that conversion ratio plus two full RGB episode buffers and 256 MiB.
    conversion_budget=math.ceil(total*0.50)+2*max(r['rows'] for r in frozen['episodes'])*640*480*3+256*1024**2
    snapshot_bytes=sum(sum(p.stat().st_size for p in snap.rglob('*') if p.is_file()) for snap in snapshots.values())
    if snapshot_bytes+conversion_budget>8*1024**3:
        raise ValueError('Estimated data plus conversion temporary peak exceeds the 8 GiB bulk budget')
    capacity(parent,conversion_budget,config)
    for split in snapshots:
        destination=parent/f'local/g1_inspire_larger_v1_{split}'
        if any(p.exists() or p.is_symlink() for p in [destination,destination.with_name(destination.name+'_v3.0'),destination.with_name(destination.name+'_v2.1')]):
            raise FileExistsError(destination)
    results={}
    for split,snap in snapshots.items():
        manifest=json.loads((snap/'manifest.json').read_text())
        if manifest.get('split_manifest_sha256')!=digest(Path(split_path)) or manifest.get('split')!=split or [r['episode'] for r in manifest['episodes']]!=frozen['splits'][split]:raise ValueError('Snapshot does not match frozen split')
        repo=f'local/g1_inspire_larger_v1_{split}';dataset=parent/repo
        if dataset.exists() or dataset.with_name(dataset.name+'_v3.0').exists():raise FileExistsError(dataset)
        common=['--config',str(config['config_path'])] if config['config_path'].exists() else []
        env=runtime_environment(config,offline=True)
        base=[str(runtime['interpreters']['unitree']),str(PROJECT_ROOT/'scripts/convert_single_camera.py'),*common,'--pilot-root',str(snap),'--output-parent',str(parent),'--repo-id',repo]
        subprocess.run(base,check=True,cwd=PROJECT_ROOT,env=env)
        verify=[str(runtime['interpreters']['gr00t']),str(Path(__file__).resolve()),'verify-rows','--dataset',str(dataset),'--snapshot-root',str(snap)]
        subprocess.run(verify,check=True,cwd=PROJECT_ROOT,env=env)
        subprocess.run([*base,'--to-v21'],check=True,cwd=PROJECT_ROOT,env=env)
        subprocess.run(verify,check=True,cwd=PROJECT_ROOT,env=env)
        validation_report=dataset.parent/(dataset.name+'-validation.json')
        command=[str(runtime['interpreters']['gr00t']),str(PROJECT_ROOT/'scripts/prepare_gr00t_dataset.py'),*common,'--dataset',str(dataset),'--pilot-root',str(snap),'--report',str(validation_report),'--apply-metadata']
        if split=='train':command.append('--compute-stats')
        subprocess.run(command,check=True,cwd=PROJECT_ROOT,env=env)
        subprocess.run(verify,check=True,cwd=PROJECT_ROOT,env=env)
        results[split]={'dataset_path':ref(dataset),'snapshot_manifest':binding(snap/'manifest.json'),
            'files_sha256':{str(p.relative_to(dataset)):digest(p) for p in sorted(dataset.rglob('*')) if p.is_file()},
            'validation_report':binding(validation_report),'v3_dataset_path':ref(dataset.with_name(dataset.name+'_v3.0'))}
    train=resolve_path(results['train']['dataset_path'])
    value={'schema_version':1,'artifact_type':'larger_dataset_conversion','split_manifest':binding(split_path),'splits':results,
        'statistics':{'owner':'train','stats_json':binding(train/'meta/stats.json'),'relative_stats_json':binding(train/'meta/relative_stats.json'),
          'validation_policy':'Validation conversion metadata is never loaded into the training processor. Only training runs GR00T statistics; evaluation uses the frozen training processor.'},
        'validation_scope':'Exact float32 states/actions, goals and row order verified in both pinned conversion formats; v2.1 video frame counts decoded, three source-image anchors per episode compared by existing validator.'}
    write_new(report_path,value);return value



# A single compressed archive avoids tens of thousands of remote small-file writes.
# The receiver verifies every member before publishing the new immutable directory.
ARCHIVE_RECEIVER = r"""
import hashlib,json,os,shutil,sys,tarfile
from pathlib import Path
final=Path(sys.argv[1]);expected_manifest=sys.argv[2];stage=final.with_name(final.name+'.partial')
if not final.is_absolute() or final.exists() or stage.exists():raise RuntimeError('Refusing existing or relative archive destination')
final.parent.mkdir(parents=True,exist_ok=True)
if shutil.disk_usage(final.parent).free<16*1024**3:raise RuntimeError('Remote archive needs 8 GiB budget plus 8 GiB reserve')
stage.mkdir();archive=stage/'dataset.tar.gz';compressed=hashlib.sha256();size=0
with archive.open('xb') as out:
 while True:
  block=sys.stdin.buffer.read(1024*1024)
  if not block:break
  size+=len(block)
  if size>8*1024**3:raise RuntimeError('Compressed archive exceeds bound')
  compressed.update(block);out.write(block)
 out.flush();os.fsync(out.fileno())
seen=set();manifest=None;expected={};total=0
with tarfile.open(archive,'r:gz') as stream:
 for member in stream:
  p=Path(member.name)
  if p.is_absolute() or '..' in p.parts or not member.isfile() or member.name in seen:raise RuntimeError('Unsafe or duplicate archive member')
  seen.add(member.name)
  if manifest is None:
   if member.name!='archive_manifest.json' or member.size>64*1024**2:raise RuntimeError('Missing bounded leading manifest')
   data=stream.extractfile(member).read()
   if hashlib.sha256(data).hexdigest()!=expected_manifest:raise RuntimeError('Archive manifest hash mismatch')
   manifest=json.loads(data);expected={r['path']:r for r in manifest['files']}
   if len(expected)!=len(manifest['files']) or len(expected)>100000:raise RuntimeError('Invalid file inventory')
   if sum(r['size'] for r in expected.values())>8*1024**3:raise RuntimeError('Uncompressed archive exceeds bound')
   with (stage/'archive_manifest.json').open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
   continue
  record=expected.get(member.name)
  if record is None or member.size!=record['size']:raise RuntimeError('Unlisted archive file or size mismatch')
  h=hashlib.sha256()
  f=stream.extractfile(member)
  while True:
   block=f.read(1024*1024)
   if not block:break
   h.update(block);total+=len(block)
  if h.hexdigest()!=record['sha256']:raise RuntimeError('Archive member SHA256 mismatch: '+member.name)
if manifest is None or seen!=set(expected)|{'archive_manifest.json'}:raise RuntimeError('Archive file inventory incomplete')
report={'passed':True,'file_count':len(expected),'total_bytes':total,'compressed_bytes':size,'compressed_sha256':compressed.hexdigest(),'manifest_sha256':expected_manifest,'all_member_sha256_match':True,'archive_manifest_report_fsync_completed':True,'source_preserved':True,'format':'gzip tar; verified member hashes; restore to a new directory before use'}
with (stage/'archive_verification.json').open('x') as out:json.dump(report,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
fd=os.open(stage,os.O_RDONLY);os.fsync(fd);os.close(fd);os.rename(stage,final)
fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(report))
"""


def archive_dataset(data_root, split_path, conversion_path, manifest_path, report_path, config):
    root=guard_derived(data_root,config)
    if root!=config['paths']['derived'].resolve():raise ValueError('Archive only the dedicated configured larger-data root')
    if not config['storage']['enabled']:raise ValueError('This bounded archive requires configured SSH storage')
    if Path(report_path).exists() or Path(manifest_path).exists():raise FileExistsError('Archive evidence already exists')
    split=load_split_manifest(split_path);conversion=json.loads(Path(conversion_path).read_text())
    if conversion.get('artifact_type')!='larger_dataset_conversion' or conversion['split_manifest']['sha256']!=digest(Path(split_path)):raise ValueError('Conversion/split binding mismatch')
    sources={str(p.relative_to(root)):p for p in sorted(root.rglob('*')) if p.is_file()}
    if any(p.is_symlink() for p in root.rglob('*')) or any('INCOMPLETE' in name for name in sources):raise ValueError('Incomplete or symlinked data archive')
    extras={'provenance/split.json':Path(split_path),'provenance/conversion.json':Path(conversion_path),
            'provenance/source-inventory.json':resolve_path(split['source_inventory']['path']),
            'provenance/label-overrides.json':resolve_path(split['label_overrides']['path']),
            'provenance/prepare_dataset_split.py':Path(__file__).resolve()}
    if set(sources)&set(extras):raise ValueError('Reserved archive provenance path already exists')
    sources.update(extras)
    if digest(extras['provenance/source-inventory.json'])!=split['source_inventory']['sha256'] or digest(extras['provenance/label-overrides.json'])!=split['label_overrides']['sha256']:
        raise ValueError('Frozen inventory or overlay changed before archive')
    if sum(p.stat().st_size for p in sources.values())>8*1024**3:raise ValueError('Data archive exceeds 8 GiB bound')
    files=[{'path':name,'size':p.stat().st_size,'sha256':digest(p)} for name,p in sources.items()]
    manifest={'schema_version':1,'artifact_type':'larger_dataset_archive','split_manifest_sha256':digest(Path(split_path)),
        'conversion_manifest_sha256':digest(Path(conversion_path)),'files':files,'total_bytes':sum(r['size'] for r in files),'source_preserved':True}
    # Fail before transfer if any final dataset file has drifted from its frozen inventory.
    archive_hashes={r['path']:r['sha256'] for r in files}
    for part in conversion['splits'].values():
        snapshot_manifest=resolve_path(part['snapshot_manifest']['path'])
        if not snapshot_manifest.is_relative_to(root):raise ValueError('Snapshot outside dedicated archive root')
        if archive_hashes[str(snapshot_manifest.relative_to(root))]!=part['snapshot_manifest']['sha256']:raise ValueError('Snapshot manifest changed before archive')
        snapshot_record=json.loads(snapshot_manifest.read_text());snapshot_root=snapshot_manifest.parent
        if archive_hashes.get(str((snapshot_root/'label-overrides.json').relative_to(root)))!=split['label_overrides']['sha256']:raise ValueError('Snapshot overlay changed before archive')
        for record in snapshot_record['episodes']:
            for relative,expected in record['files_sha256'].items():
                key=str((snapshot_root/'raw/pick_up_black_box'/record['episode']/relative).relative_to(root))
                if archive_hashes.get(key)!=expected:raise ValueError('Snapshot data changed before archive')
            key=str((snapshot_root/'provenance'/f"{record['episode']}.data.json").relative_to(root))
            if archive_hashes.get(key)!=record['source_data_json_sha256']:raise ValueError('Snapshot source provenance changed before archive')
        dataset=resolve_path(part['dataset_path'])
        if not dataset.is_relative_to(root):raise ValueError('Converted dataset outside dedicated archive root')
        for name,expected in part['files_sha256'].items():
            key=str((dataset/name).relative_to(root))
            if archive_hashes.get(key)!=expected:raise ValueError('Converted data changed before archive')
    write_new(manifest_path,manifest);manifest_sha=digest(Path(manifest_path))
    destination=str(Path(config['storage']['remote_path'])/'derived/larger-dataset-v1')
    command=' '.join(shlex.quote(arg) for arg in ['python3','-c',ARCHIVE_RECEIVER,destination,manifest_sha])
    with subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',config['storage']['host'],command],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as process:
        try:
            with tarfile.open(fileobj=process.stdin,mode='w|gz',compresslevel=1) as stream:
                stream.add(manifest_path,arcname='archive_manifest.json',recursive=False)
                for name,path in sources.items():stream.add(path,arcname=name,recursive=False)
            process.stdin.close();result=process.stdout.read();code=process.wait()
            if code:raise RuntimeError(f'Remote archive verification failed with exit {code}')
        except Exception:
            process.kill();process.wait();raise
    report=json.loads(result)
    if not report.get('passed') or report['manifest_sha256']!=manifest_sha:raise ValueError('Remote archive report binding failed')
    report['artifact_ref']='archive:derived/larger-dataset-v1/dataset.tar.gz';write_new(report_path,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('inventory');p.add_argument('--raw-root',type=Path);p.add_argument('--label-overrides',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('freeze');p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--copy-completion-report',type=Path)
    p=sub.add_parser('snapshot');p.add_argument('--split-manifest',type=Path,required=True);p.add_argument('--split',choices=['train','validation'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-mib',type=int,default=MAX_SNAPSHOT_MIB)
    p=sub.add_parser('verify-rows');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--snapshot-root',type=Path,required=True)
    p=sub.add_parser('convert');p.add_argument('--split-manifest',type=Path,required=True);p.add_argument('--train-snapshot',type=Path,required=True);p.add_argument('--validation-snapshot',type=Path,required=True);p.add_argument('--output-parent',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p=sub.add_parser('archive');p.add_argument('--data-root',type=Path,required=True);p.add_argument('--split-manifest',type=Path,required=True);p.add_argument('--conversion-manifest',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();config=load_config(args.config)
    if args.action=='inventory':inventory(resolve_path(args.raw_root or config['paths']['raw_data']),resolve_path(args.label_overrides),resolve_path(args.output))
    elif args.action=='freeze':freeze_split(resolve_path(args.inventory),resolve_path(args.output),args.copy_completion_report)
    elif args.action=='snapshot':snapshot(resolve_path(args.split_manifest),args.split,resolve_path(args.output),config,args.max_mib)
    elif args.action=='verify-rows':print(json.dumps(verify_rows(resolve_path(args.dataset),resolve_path(args.snapshot_root))))
    elif args.action=='archive':archive_dataset(resolve_path(args.data_root),resolve_path(args.split_manifest),resolve_path(args.conversion_manifest),resolve_path(args.manifest),resolve_path(args.report),config)
    else:convert(resolve_path(args.split_manifest),resolve_path(args.train_snapshot),resolve_path(args.validation_snapshot),resolve_path(args.output_parent),resolve_path(args.report),config)


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as error:sys.exit(f'Dataset split preparation failed: {error}')
