#!/usr/bin/env python3
"""Freeze a target-only temporal-change split and compare saved pilot predictions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from project_config import PROJECT_ROOT, load_config, guard_path
from convert_single_camera import ensure_python
from evaluate_gr00t_pilot import GROUPS, SEEDS, HORIZON, digest, write_new_json, summarize

THRESHOLD = 0.05
METHOD = ('For each joint, peak-to-peak absolute ground-truth target within the 16-frame horizon '
          'divided by the maximum saved action normalization span over that horizon; '
          'take the maximum across nonconstant joints within each group and select scores > 0.05. '
          'This measures changing target commands, not state-action offset or semantic task phase.')


def temporal_subset(target, lower, upper):
    import numpy as np
    target = np.asarray(target)
    lower, upper = np.broadcast_to(lower,(HORIZON,26)), np.broadcast_to(upper,(HORIZON,26))
    if target.ndim != 3 or target.shape[1:] != (HORIZON,26):
        raise ValueError('Expected absolute targets shaped (samples,16,26)')
    if not all(np.isfinite(x).all() for x in (target,lower,upper)) or (upper<lower).any():
        raise ValueError('Invalid target or normalization bounds')
    maximum_span = np.max(upper-lower,axis=0)
    nonconstant = (~np.isclose(upper,lower)).any(axis=0) & (maximum_span>0)
    ratios = np.divide(np.ptp(target,axis=1),maximum_span,
                       out=np.zeros((len(target),26),dtype=np.float64),where=nonconstant)
    scores = np.stack([ratios[:,begin:end].max(axis=1) for _,begin,end,_ in GROUPS],axis=1)
    return scores>THRESHOLD, scores, nonconstant


def load_evidence(path):
    import numpy as np
    report = json.loads(path.read_text())
    arrays_path = path.with_suffix('.npz')
    if report.get('passed') is not True or report.get('artifacts',{}).get(arrays_path.name) != digest(arrays_path):
        raise ValueError('Evaluation is incomplete or its array artifact changed')
    with np.load(arrays_path,allow_pickle=False) as stream:
        arrays={key:stream[key].copy() for key in stream.files}
    if arrays['absolute_predictions'].shape != (100,3,HORIZON,26) or arrays['seeds'].tolist()!=SEEDS:
        raise ValueError('Expected the frozen100-window,3-seed evaluation')
    return report,arrays


def declaration_for(baseline_path, baseline_report, arrays):
    if baseline_report.get('role') != 'two_step_reference':
        raise ValueError('Declaration must be made from the two-step reference evaluation')
    mask,scores,valid = temporal_subset(arrays['targets'],arrays['lower'],arrays['upper'])
    groups={}
    for i,(name,begin,end,unit) in enumerate(GROUPS):
        groups[name]={'changing_target_count':int(mask[:,i].sum()),
                      'small_target_change_count':int((~mask[:,i]).sum()),
                      'changing_counts_per_episode':[int(mask[arrays['episode_indices']==episode,i].sum()) for episode in range(5)],
                      'nonconstant_joint_count':int(valid[begin:end].sum()),'unit':unit}
    return {'schema_version':1,'method':METHOD,'threshold':THRESHOLD,
            'baseline_report_sha256':digest(baseline_path),'baseline_npz_sha256':digest(baseline_path.with_suffix('.npz')),
            'sample_manifest_sha256':baseline_report['sample_manifest_sha256'],
            'processor_semantics_sha256':baseline_report['processor_semantics_sha256'],
            'sample_count':len(mask),'joint_order':[x[0] for x in GROUPS],'seeds':SEEDS,'horizon':HORIZON,
            'changing_target_mask':mask.tolist(),'scores':scores.tolist(),'groups':groups,
            'selection_inputs':'Ground-truth targets and saved normalization bounds only; no predictions or candidate metrics.'}


def freeze_declaration(path, declaration, allow_create):
    if path.exists():
        if path.is_symlink() or json.loads(path.read_text()) != declaration:
            raise ValueError('Frozen temporal-change declaration differs')
    elif allow_create:
        write_new_json(path,declaration)
    else:
        raise ValueError('Freeze the target-only declaration before candidate comparison')


def assert_comparable(baseline_report,candidate_report,baseline,candidate):
    import numpy as np
    for key in ['sample_manifest_sha256','processor_semantics_sha256','seeds','horizon','joint_order']:
        if baseline_report.get(key)!=candidate_report.get(key):
            raise ValueError('Evaluation reports differ: '+key)
    for key in ['targets','states','seeds','episode_indices','starts','lower','upper']:
        if not np.array_equal(baseline[key],candidate[key]):
            raise ValueError('Paired evaluation arrays differ: '+key)


def comparison_groups(baseline,candidate,mask):
    import numpy as np
    result={}
    for i,(name,_,_,unit) in enumerate(GROUPS):
        result[name]={'unit':unit}
        for subset,selection in [('changing_target',mask[:,i]),('small_target_change',~mask[:,i])]:
            metrics=[]
            for arrays in (baseline,candidate):
                groups,_,_=summarize(arrays['absolute_predictions'][selection],arrays['targets'][selection],
                    arrays['states'][selection],arrays['lower'],arrays['upper'])
                metrics.append(groups[name]['all'])
            before,after=metrics
            first,last=before['model']['mae'],after['model']['mae']
            result[name][subset]={'sample_count':int(np.sum(selection)),'reference':before['model'],
                'candidate':after['model'],'persistence':before['persistence'],
                'mae_change':last-first if first is not None else None,
                'relative_mae_change_percent':100*(last-first)/first if first else None}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--declaration',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();config=load_config(args.config)
    ensure_python(config,'gr00t')
    baseline_path=guard_path(args.baseline,config,'reports')
    declaration_path=guard_path(args.declaration,config,'reports')
    baseline_report,baseline=load_evidence(baseline_path)
    declaration=declaration_for(baseline_path,baseline_report,baseline)
    freeze_declaration(declaration_path,declaration,allow_create=args.candidate is None)
    if args.candidate is None:
        if args.report:parser.error('--report requires --candidate')
        print(json.dumps({'declaration':str(declaration_path),'groups':declaration['groups']},indent=2));return
    if not args.report:parser.error('--candidate requires --report')
    candidate_path=guard_path(args.candidate,config,'reports');report_path=guard_path(args.report,config,'reports')
    if report_path.exists() or report_path.is_symlink():raise FileExistsError(report_path)
    candidate_report,candidate=load_evidence(candidate_path)
    if candidate_report.get('baseline_report_sha256')!=digest(baseline_path):
        raise ValueError('Candidate evaluation references a different baseline')
    assert_comparable(baseline_report,candidate_report,baseline,candidate)
    import numpy as np
    frozen=json.loads(declaration_path.read_text())
    mask=np.array(frozen['changing_target_mask'],dtype=bool)
    result={'generated_at':datetime.now(timezone.utc).isoformat(),'passed':True,
        'candidate_checkpoint':candidate_report['candidate_checkpoint'],'baseline_report_sha256':digest(baseline_path),
        'candidate_report_sha256':digest(candidate_path),'declaration_sha256':digest(declaration_path),
        'baseline_npz_sha256':digest(baseline_path.with_suffix('.npz')),'candidate_npz_sha256':digest(candidate_path.with_suffix('.npz')),
        'sample_manifest_sha256':candidate_report['sample_manifest_sha256'],
        'method':METHOD,'threshold':THRESHOLD,'groups':comparison_groups(baseline,candidate,mask),
        'repository_commit':subprocess.check_output(['git','-C',str(PROJECT_ROOT),'rev-parse','HEAD'],text=True).strip(),
        'script_sha256':{name:digest(PROJECT_ROOT/name) for name in ['scripts/compare_pilot_temporal_change.py','scripts/evaluate_gr00t_pilot.py','scripts/project_config.py']},
        'scope':'Additional predeclared ground-truth-only subset analysis of saved predictions; no new inference, training, or task-phase labels. Passing means comparable finite evidence, not improvement.'}
    write_new_json(report_path,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
