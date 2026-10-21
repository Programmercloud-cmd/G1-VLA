#!/usr/bin/env python3
"""Additive CPU summary of completed frozen reports; never selects checkpoints."""
import argparse,csv,hashlib,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path

GROUPS=['left_arm','right_arm','left_hand','right_hand']
METRICS=['mae','rmse','normalized_mae','normalized_rmse']

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def binding(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def require(ok,message):
 if not ok:raise ValueError(message)

def evaluation(path,protocol,protocol_sha,split,baseline_sha=None):
 r=read(path);step=r['checkpoint_global_step']
 require(r.get('passed') is True and r.get('artifact_type')=='larger_dataset_evaluation','Incomplete evaluation: '+str(path))
 require(r['evaluation_protocol_sha256']==protocol_sha and r['split']==split,'Wrong evaluation binding')
 require(r['sample_manifest_sha256']==protocol['samples'][split]['sha256'],'Wrong samples')
 require(r['processor_semantics_sha256']==protocol['processor']['semantics_sha256'] and r['statistics_sha256']==protocol['processor']['files_sha256']['statistics.json'],'Wrong processor/statistics')
 require(r['role']==('step_zero_reference' if step==0 else 'checkpoint'),'Wrong evaluation role')
 if baseline_sha is not None:require(r['reference_report_sha256']==baseline_sha,'Wrong baseline')
 require(r['recipe']==protocol['recipe'] and r['evaluation_mode']=={'model_training':False,'processor_training':False},'Wrong evaluation recipe/mode')
 for name,h in r['artifacts'].items():require(sha(Path(path).parent/name)==h,'Evaluation artifact differs: '+name)
 score=statistics.mean(r['groups'][g]['all']['model']['normalized_mae'] for g in GROUPS)
 require(math.isclose(score,r['group_balanced_normalized_mae'],rel_tol=1e-12,abs_tol=1e-12),'Group-balanced score differs')
 for groups in [r['groups'],*r['per_episode'].values()]:
  for g in GROUPS:
   for subset in ['all','changing_target','small_target_change']:
    for role,metrics in groups[g][subset].items():
     if role=='change_from_step_zero':continue
     for key in METRICS:
      value=metrics[key];require(value is None or (math.isfinite(value) and value>=0),'Invalid metric')
 return {'path':Path(path),'report':r}

def writecsv(path,rows,columns):
 with path.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['protocol','run-manifest','train-report','finite-audit','baseline-train','baseline-validation','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--checkpoint-train',type=Path,action='append',default=[]);p.add_argument('--checkpoint-validation',type=Path,action='append',default=[])
 p.add_argument('--selection',type=Path);p.add_argument('--loss-window-updates',type=int,default=128);p.add_argument('--no-plots',action='store_true')
 a=p.parse_args();require(not a.output.exists(),'Output already exists');require(a.loss_window_updates>0,'Window must be positive')
 protocol=read(a.protocol);psha=sha(a.protocol);run=read(a.run_manifest);train=read(a.train_report)
 require(train['artifact_type']=='larger_dataset_training_report' and train['run']==run['run'],'Wrong final training report')
 require(train['run_manifest_sha256']==sha(a.run_manifest),'Run manifest changed')
 require(train['finite_audit_sha256']==sha(a.finite_audit),'Finite audit changed or training still writing')
 require(train['evaluation_protocol_sha256']==run['evaluation_protocol_sha256']==psha,'Protocol differs')
 baseline={s:evaluation(getattr(a,'baseline_'+s),protocol,psha,s) for s in ['train','validation']}
 require(all(x['report']['checkpoint_global_step']==0 for x in baseline.values()),'Baseline must be step zero')
 records=list(baseline.values());pairs={};certified={x['step'] for x in train['checkpoints'] if x['passed']}
 for split in ['train','validation']:
  seen=set()
  for path in getattr(a,'checkpoint_'+split):
   item=evaluation(path,protocol,psha,split,sha(baseline[split]['path']));step=item['report']['checkpoint_global_step']
   require(step in protocol['recipe']['checkpoint_steps'] and step in certified and step not in seen,'Duplicate/unplanned/uncertified checkpoint');seen.add(step);records.append(item);pairs.setdefault(step,{})[split]=item
 for step,pair in pairs.items():
  if len(pair)==2:require(pair['train']['report']['checkpoint_files_sha256']==pair['validation']['report']['checkpoint_files_sha256'],'Paired checkpoint weights differ')
 events=[json.loads(line) for line in a.finite_audit.read_text().splitlines() if line.strip()];losses=[x for x in events if x['event']=='micro_loss'];steps=[x['step'] for x in events if x['event']=='step_end'];accum=run['accumulation']
 require(steps==list(range(1,len(steps)+1)),'Optimizer steps are not contiguous')
 for i,row in enumerate(losses,1):require(row['microbatch']==i and row['step_before']==(i-1)//accum and row.get('finite') is True and math.isfinite(row['loss']),'Invalid microbatch audit')
 require(len(losses)>=len(steps)*accum and len(losses)<(len(steps)+1)*accum+1,'Loss/optimizer progress differs')
 consumed=Counter((x['episode_index'],x['row']) for x in losses);eligible=run['eligible_training_windows'];coverage=[]
 for i,episode in enumerate(run['episodes']):
  counts=[v for (ep,_),v in consumed.items() if ep==i];coverage.append({'episode':episode['episode'],'episode_index':i,'eligible_windows':episode['rows']-15,'consumed_examples':sum(counts),'unique_window_starts':len(counts)})
 require(sum(x['consumed_examples'] for x in coverage)==len(losses),'Out-of-range consumed episode')
 checkpoint_coverage=[]
 for event in events:
  if event['event']=='checkpoint':
   path=Path(event['coverage_path']);require(sha(path)==event['coverage_sha256'],'Saved coverage changed');c=read(path);require(c['consumed_examples']==event['step']*accum,'Checkpoint coverage count differs');checkpoint_coverage.append({'binding':binding(path),'coverage':c})
 windows=[];width=a.loss_window_updates*accum;curve_losses=losses[:len(steps)*accum]
 for begin in range(0,len(curve_losses),width):
  chunk=curve_losses[begin:begin+width];windows.append({'microbatch_first':chunk[0]['microbatch'],'microbatch_last':chunk[-1]['microbatch'],'update_first':chunk[0]['step_before']+1,'update_last':chunk[-1]['step_before']+1,'microbatch_count':len(chunk),'mean_loss':statistics.mean(x['loss'] for x in chunk),'complete_window':len(chunk)==width})
 samples={s:read(Path(protocol['samples'][s]['path'])) for s in ['train','validation']}
 sample_names={}
 for split,record in samples.items():
  for x in record.get('samples',[]):
   if 'source_episode' in x:sample_names[(split,str(x['episode_index']))]=x['source_episode']
 rows=[]
 for item in records:
  r=item['report'];split=r['split'];step=r['checkpoint_global_step']
  for ep,groups in [('all',r['groups']),*r['per_episode'].items()]:
   for group in GROUPS:
    for subset in ['all','changing_target','small_target_change']:
     for role,m in groups[group][subset].items():
      if role=='change_from_step_zero':continue
      row={'step':step,'split':split,'episode_index':ep,'source_episode':sample_names.get((split,ep),''),'group':group,'unit':groups[group]['unit'],'subset':subset,'role':role,'sample_count':m['sample_count']}
      row.update({k:m[k] for k in METRICS});rows.append(row)
 selected=None
 if a.selection:
  s=read(a.selection);require(s['artifact_type']=='larger_dataset_checkpoint_selection' and s['passed'] is True and s['evaluation_protocol_sha256']==psha,'Invalid official selection')
  require(set(pairs)==set(protocol['recipe']['checkpoint_steps']) and all(len(pair)==2 for pair in pairs.values()),'Official selection requires all pairs')
  for split,entries in s['evaluations'].items():
   for entry in entries:require(entry['report']['sha256']==sha(pairs[entry['checkpoint_global_step']][split]['path']),'Official selection references different evaluations')
  selected={'step':s['selected_step'],'source':binding(a.selection),'rule':s['selection_rule']}
 complete=train['passed'] is True and len(steps)==run['max_steps'];missing={str(step):[split for split in ['train','validation'] if split not in pairs.get(step,{})] for step in protocol['recipe']['checkpoint_steps']};missing={k:v for k,v in missing.items() if v}
 summary={'schema_version':1,'artifact_type':'additive_training_curve_analysis','run':run['run'],'training_status':'completed' if complete else 'incomplete','training_report_passed':train['passed'],'completed_optimizer_updates':len(steps),'planned_optimizer_updates':run['max_steps'],'aborted':train.get('aborted'),'finite_audit_summary':train['finite_audit'],'nonfinite_events':[x for x in events if x['event']=='nonfinite'],'offline_scope':'Action errors on frozen train/selection-validation samples. Persistence descriptive. No robot competence claim. This helper does not select a checkpoint.','protocol':binding(a.protocol),'run_manifest':binding(a.run_manifest),'training_report':binding(a.train_report),'finite_audit':binding(a.finite_audit),'analysis_source':binding(Path(__file__)),'evaluation_reports':[binding(x['path']) for x in records],'missing_evaluation_splits_by_checkpoint':missing,'official_selection':selected,'trailing_micro_losses_not_applied_to_a_completed_update':losses[len(steps)*accum:],'loss_curve_policy':'Window means use only complete optimizer-update microbatches; trailing partial-update losses are retained separately.','coverage':{'policy':'Loss-computed sample exposure, including any trailing partial update; separate complete-update count is explicit.','complete_update_microbatches':len(steps)*accum,'trailing_partial_update_microbatches':len(losses)-len(steps)*accum,'consumed_examples':len(losses),'unique_window_starts':len(consumed),'eligible_windows':eligible,'example_equivalent_passes':len(losses)/eligible,'unique_window_fraction':len(consumed)/eligible,'episodes_observed':sum(x['consumed_examples']>0 for x in coverage),'per_episode':coverage,'checkpoint_proofs':checkpoint_coverage},'loss_windows':windows,'scores':[{'step':x['report']['checkpoint_global_step'],'split':x['report']['split'],'group_balanced_normalized_mae':x['report']['group_balanced_normalized_mae']} for x in records]}
 a.output.mkdir(parents=True);(a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 writecsv(a.output/'loss-windows.csv',windows,['microbatch_first','microbatch_last','update_first','update_last','microbatch_count','mean_loss','complete_window'])
 writecsv(a.output/'metrics.csv',rows,['step','split','episode_index','source_episode','group','unit','subset','role','sample_count',*METRICS]);writecsv(a.output/'coverage.csv',coverage,['episode','episode_index','eligible_windows','consumed_examples','unique_window_starts'])
 if not a.no_plots:plot(a.output,records,windows)
 outputs={x.name:binding(x) for x in a.output.iterdir() if x.is_file()};(a.output/'artifacts.json').write_text(json.dumps(outputs,indent=2)+'\n')
 print(json.dumps({'output':str(a.output),'training_status':summary['training_status'],'completed_updates':len(steps),'evaluation_reports':len(records),'official_selected_step':selected['step'] if selected else None},indent=2))

def plot(output,records,windows):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'figure.dpi':150,'axes.grid':True,'grid.alpha':.25})
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 if windows:axes[0,0].plot([w['update_last'] for w in windows],[w['mean_loss'] for w in windows]);axes[0,0].set_title('Training loss: completed-update window means');axes[0,0].set_ylabel('Training loss')
 for split in ['train','validation']:
  items=sorted([x['report'] for x in records if x['report']['split']==split],key=lambda x:x['checkpoint_global_step']);xs=[x['checkpoint_global_step'] for x in items]
  axes[0,1].plot(xs,[x['group_balanced_normalized_mae'] for x in items],marker='o',label=split)
  ax=axes[1,0 if split=='train' else 1]
  for g in GROUPS:ax.plot(xs,[x['groups'][g]['all']['model']['normalized_mae'] for x in items],marker='o',label=g)
  ax.set_title(split+' all-window normalized MAE');ax.legend(fontsize=8)
 axes[0,1].set_title('Four-group mean normalized MAE');axes[0,1].legend()
 for ax in axes.flat:ax.set_xlabel('Optimizer update')
 fig.tight_layout();fig.savefig(output/'learning-curves.png');plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(15,7))
 for gi,g in enumerate(GROUPS):
  for mi,metric in enumerate(['mae','rmse']):
   ax=axes[mi,gi]
   for split in ['train','validation']:
    items=sorted([x['report'] for x in records if x['report']['split']==split],key=lambda x:x['checkpoint_global_step']);ax.plot([x['checkpoint_global_step'] for x in items],[x['groups'][g]['all']['model'][metric] for x in items],marker='o',label=split)
   ax.set_title(g+' '+metric);ax.set_ylabel(items[0]['groups'][g]['unit']);ax.set_xlabel('Optimizer update');ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(output/'native-error-curves.png');plt.close(fig)

if __name__=='__main__':main()
