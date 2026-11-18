"""Audit measured costs from completed receipts, without inventing missing timings.

Final mode requires 256 common development, 128 LoRA-selection, and 640 test
episodes. Component wall-time sums are not GPU compute time or parallel elapsed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

COMMON = {
    'temporal-development-v2':('development_collection',64),
    'spatial-development-v2':('development_collection',64),
    'temporal-qualification-v1':('qualification',32),
    'spatial-qualification-v1':('qualification',16),
    'temporal-qualification-teacher-v1':('qualification',16),
    'spatial-qualification-teacher-v1':('qualification',16),
    'spatial-left-competence-v1':('wording_screen',48),
}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8<<20):digest.update(block)
    return digest.hexdigest()


def seconds(value):
    if value is None:return None
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0:
        raise ValueError('Measured duration must be finite and nonnegative, or explicitly unknown')
    return float(value)


def timing(receipt):
    wall=seconds(receipt.get('wall_seconds'))
    queue=seconds(receipt.get('queue_seconds'))
    execution=seconds(receipt.get('execution_seconds'))
    if wall is not None and queue is not None and execution is not None and queue+execution>wall+1e-6:
        raise ValueError('Queue plus execution exceeds measured wrapper wall time')
    return {'wall_seconds':wall,'execution_seconds':execution,'queue_seconds':queue,
            'queue_execution_status':'measured' if queue is not None and execution is not None else 'not separately recorded',
            'note':'Execution includes policy loading/reloading and simulator work; no unmeasured startup cost is subtracted.'}


def residual(wrapper,internal):
    wrapper,internal=seconds(wrapper),seconds(internal)
    if wrapper is None or internal is None:return None
    value=wrapper-internal
    if value < -1e-6:raise ValueError('Internal duration exceeds its enclosing wrapper')
    return max(value,0.)


def parallel_interval(intervals):
    """Measured runner overlap, never an estimate of device utilization."""
    if not intervals:return {'elapsed_span_seconds':None,'union_runner_seconds':None,'sum_runner_seconds':None}
    spans=sorted((float(start),float(start)+seconds(duration)) for start,duration in intervals)
    if any(not math.isfinite(start) for start,_ in spans):raise ValueError('Invalid runner start time')
    total=sum(end-start for start,end in spans);union=0.;left,right=spans[0]
    for start,end in spans[1:]:
        if start>right:union+=right-left;left,right=start,end
        else:right=max(right,end)
    union+=right-left
    return {'elapsed_span_seconds':max(end for _,end in spans)-spans[0][0],
            'union_runner_seconds':union,'sum_runner_seconds':total,
            'interpretation':'Overlap of measured end-to-end runner intervals, including CPU work and waits; not active GPU compute time.'}


class Audit:
    def __init__(self):
        self.sources={};self.components=[];self.batch_paths=set();self.observations=set();self.errors=[]

    def read(self,path):
        path=Path(path).resolve();self.sources[str(path)]=sha(path)
        return json.loads(path.read_text())

    def component(self,category,scope,case,path,*,additive=True,**values):
        row={'category':category,'scope':scope,'case':case,'source_path':str(Path(path).resolve()),
             'additive_wall_component':additive,**values}
        self.components.append(row)
        return row

    def worker(self,root,expected_count,category,scope,split,allowed_conditions):
        root=Path(root).resolve();plan=self.read(root/'plan.json');status=self.read(root/'status.json')
        if plan.get('split')!=split or status.get('status')!='complete':
            raise ValueError(f'Incomplete or wrong-split worker: {root}')
        count=0
        for number,spec in enumerate(plan['batches']):
            batch=root/f'batch-{number:02d}-{spec["cell"]}-{spec["condition"]}'
            receipt_path=batch/'completed.json'
            if receipt_path in self.batch_paths:raise ValueError('A completed batch would be counted twice')
            if spec['condition'] not in allowed_conditions:raise ValueError('Worker uses unexpected comparison conditions')
            receipt=self.read(receipt_path);rows_path=batch/'episodes.json';rows=self.read(rows_path)
            if receipt['episodes_sha256']!=self.sources[str(rows_path.resolve())]:raise ValueError('Cost receipt episode hash mismatch')
            n=spec.get('n',16)
            if receipt['n']!=n or len(rows)!=n:raise ValueError('Cost receipt count mismatch')
            if (receipt['batch'],receipt['cell'],receipt['condition'])!=(number,spec['cell'],spec['condition']):
                raise ValueError('Cost receipt identity mismatch')
            paths=[str(Path(row['obs']).resolve()) for row in rows]
            if len(set(paths))!=len(paths) or self.observations.intersection(paths):
                raise ValueError('The same recorded episode would be counted twice')
            if any(row.get('ok') is not True or row.get('split')!=split or row.get('cohort_id')!=plan['cohort_id'] for row in rows):
                raise ValueError('Cost episode is incomplete or belongs to another cohort')
            values=timing(receipt)
            if values['wall_seconds'] is None:raise ValueError('Completed batch lacks its wall-time receipt')
            self.batch_paths.add(receipt_path);self.observations.update(paths);count+=n
            self.component(category,scope,plan['case'],receipt_path,episodes=n,**values)
        if count!=expected_count or status.get('completed_episodes')!=count:
            raise ValueError(f'Expected exactly {expected_count} episodes at {root}; got {count}')
        return plan


def audit_adaptation(audit,adaptation,case):
    status=audit.read(adaptation/'status.json')
    if status.get('status')!='complete' or status.get('case')!=case:raise ValueError('Original adaptation is incomplete')
    stages={stage['stage']:stage for stage in status['completed_stages']}
    expected={'edit-prepare','lora-targets','edit-capture','edit-solve','lora-train'}
    if set(stages)!=expected:raise ValueError('Original adaptation stage receipts are incomplete')
    dataset=audit.read(adaptation/'lora-dataset/dataset.json')
    if dataset.get('split')!='development' or dataset.get('live_replay_gate',{}).get('status')!='pass':
        raise ValueError('Teacher preparation lacks its development replay certificate')
    capture=[audit.read(adaptation/f'edit-prepared/captures/{which}-receipt.json') for which in ('original','teacher')]
    solve=audit.read(adaptation/'edit-fit/solve.json')
    nomination_path=adaptation/'lora-sweep/nomination.json';nomination=audit.read(nomination_path)
    sweep_status=audit.read(adaptation/'lora-sweep/status.json')
    if nomination.get('status')!='complete' or sweep_status.get('status')!='complete' or len(nomination['candidates'])!=3:
        raise ValueError('All three LoRA fit cost receipts are required')
    if nomination['dataset_manifest_sha256']!=audit.sources[str((adaptation/'lora-dataset/dataset.json').resolve())]:
        raise ValueError('Tuning cost receipt belongs to another teacher dataset')
    wrapper_sum=sum(seconds(stage['wall_seconds']) for stage in stages.values())
    audit.component('adaptation_controller_overhead','shared_preparation',case,adaptation/'status.json',
                    wall_seconds=residual(status['wall_seconds'],wrapper_sum),execution_seconds=None,queue_seconds=None,
                    note='Measured overall runner minus recorded stage wrappers; includes source verification and bookkeeping.')
    audit.component('closed_form_input_preparation','closed_form',case,adaptation/'status.json',
                    wall_seconds=seconds(stages['edit-prepare']['wall_seconds']),execution_seconds=None,queue_seconds=None)
    teacher_wall=seconds(stages['lora-targets']['wall_seconds'])
    audit.component('teacher_target_preparation_and_live_replay','shared_teacher_preparation',case,adaptation/'lora-dataset/dataset.json',
                    wall_seconds=teacher_wall,internal_preparation_seconds=seconds(dataset['wall_seconds']),
                    wrapper_overhead_seconds=residual(teacher_wall,dataset['wall_seconds']),
                    measured_live_replay_seconds=seconds(dataset['live_replay_gate']['wall_seconds']),
                    teacher_forward_calls=dataset['teacher_forward_calls'],live_replay_forward_calls=dataset['live_replay_gate']['forward_calls'],
                    execution_seconds=None,queue_seconds=None,
                    note='Native teacher sampling, loading, validation, and I/O are not separately timed; no cold-start or pure-forward latency claim.')
    capture_internal=sum(seconds(receipt['wall_seconds']) for receipt in capture)
    audit.component('activation_capture','closed_form',case,adaptation/'status.json',
                    wall_seconds=seconds(stages['edit-capture']['wall_seconds']),internal_capture_seconds=capture_internal,
                    wrapper_overhead_seconds=residual(stages['edit-capture']['wall_seconds'],capture_internal),execution_seconds=None,queue_seconds=None)
    audit.component('closed_form_solve_and_export','closed_form',case,adaptation/'edit-fit/solve.json',
                    wall_seconds=seconds(stages['edit-solve']['wall_seconds']),internal_seconds=seconds(solve['wall_seconds']),
                    solve_export_seconds=seconds(solve['solve_export_wall_seconds']),
                    wrapper_overhead_seconds=residual(stages['edit-solve']['wall_seconds'],solve['wall_seconds']),execution_seconds=None,queue_seconds=None)
    new_wrappers=[];fit_sum=0.
    ids=set()
    for candidate in nomination['candidates']:
        ids.add(candidate['candidate_id']);path=Path(candidate['training_path']);training=audit.read(path)
        if audit.sources[str(path.resolve())]!=candidate['training_sha256']:raise ValueError('Training cost report changed after nomination')
        measured=seconds(training['wall_seconds']);wrapper=seconds(candidate['wrapper_wall_seconds'])
        if measured!=candidate['training_wall_seconds']:raise ValueError('Nomination and training timing differ')
        fit_sum+=measured
        if candidate['stage']=='new':new_wrappers.append(wrapper)
        elif not math.isclose(wrapper,stages['lora-train']['wall_seconds'],rel_tol=0,abs_tol=1e-9):
            raise ValueError('Original fit wrapper cost differs from original stage receipt')
        audit.component('lora_fit','lora',case,path,candidate_id=candidate['candidate_id'],learning_rate=candidate['learning_rate'],
                        wall_seconds=wrapper,training_load_validation_export_seconds=measured,
                        wrapper_overhead_seconds=residual(wrapper,measured),execution_seconds=None,queue_seconds=None,
                        note='Includes the entire measured fit, including loading, development validation and tensor export.')
    if ids!={'initial-1e-3','lr-3e-5','lr-1e-4'} or not math.isclose(fit_sum,nomination['costs']['total_training_seconds'],rel_tol=0,abs_tol=1e-9):
        raise ValueError('Incomplete or inconsistent three-candidate tuning cost')
    audit.component('lora_sweep_controller_overhead','lora',case,adaptation/'lora-sweep/status.json',
                    wall_seconds=residual(sweep_status['wall_seconds'],sum(new_wrappers)),execution_seconds=None,queue_seconds=None,
                    note='Measured sweep wrapper minus both new fit wrappers; includes initial-candidate verification and sealing.')
    selection=audit.read(adaptation/'lora-selected/selection.json')
    if selection.get('status')!='complete' or selection.get('case')!=case or selection.get('split')!='development' or selection.get('heldout_used') is not False:
        raise ValueError('Completed development-only LoRA selection receipt is required')
    if selection['nomination_sha256']!=audit.sources[str(nomination_path.resolve())]:
        raise ValueError('Selection and nomination receipts differ')
    selected_seal=audit.read(adaptation/'lora-selected/seal.json')
    selected_path=str((adaptation/'lora-selected/selection.json').resolve())
    if selected_seal['artifacts'].get(selected_path)!=audit.sources[selected_path]:
        raise ValueError('Selection receipt differs from its selected-candidate seal')
    selection_root=Path(selection['evaluation_root'])
    audit.worker(selection_root,64,'lora_closed_loop_selection','lora','development',{'lora_initial','lora_nominee'})
    return (status['started_at'],status['wall_seconds']),(sweep_status['started_at'],sweep_status['wall_seconds'])


def preserved_attempts(audit,root):
    for case in ('temporal','spatial'):
        path=root/f'{case}-development-v1/status.json';status=audit.read(path)
        if status.get('status') not in {'stopped','failed'}:raise ValueError('Preserved startup attempt has unexpected status')
        audit.component('failed_startup_attempt','shared_research_debugging',case,path,
                        wall_seconds=seconds(status.get('wall_seconds')),execution_seconds=None,queue_seconds=None,
                        receipted_episodes=status.get('completed_episodes'),attempted_policy_episodes=None,
                        note='No duration inferred from timestamps or filesystem metadata; unreceipted execution effort is unknown.')
    manifest=root/'adaptation-attempts.json'
    for attempt in audit.read(manifest):
        folder=Path(attempt['attempt']);status=audit.read(folder/'status.json')
        known_values=[seconds(stage['wall_seconds']) for stage in status.get('completed_stages',[])]
        partial=folder/'lora-dataset/progress.json'
        if partial.exists():known_values.append(seconds(audit.read(partial)['wall_seconds']))
        known=sum(known_values) if known_values else None
        audit.component('failed_target_preparation','shared_research_debugging',status.get('case'),folder/'status.json',
                        wall_seconds=seconds(status.get('wall_seconds')),execution_seconds=None,queue_seconds=None,
                        measured_substage_lower_bound_seconds=known,closed_loop_episodes=attempt.get('closed_loop_episodes'),
                        note='Known completed/partial substage durations are a lower bound, not the full failed-attempt duration.')
        if attempt.get('diagnostic'):
            path=Path(attempt['diagnostic'])/'diagnostic.json';diagnostic=audit.read(path)
            if diagnostic.get('status')!='complete':raise ValueError('Referenced replay diagnostic is incomplete')
            audit.component('replay_numerics_diagnostic','shared_research_debugging',status.get('case'),path,
                            wall_seconds=seconds(diagnostic['wall_seconds']),execution_seconds=None,queue_seconds=None,
                            closed_loop_episodes=0,note='Development replay diagnostic only; not policy rollouts or fit success.')


def totals(rows,key):
    values=[row[key] for row in rows if row.get(key) is not None]
    return {'measured_sum_seconds':sum(values) if values else None,
            'measured_components':len(values),'unmeasured_components':len(rows)-len(values)}


def build(root,report_path,allow_partial=False):
    root=Path(root).resolve();audit=Audit();adaptation_intervals=[];sweep_intervals=[]
    def attempt(label,operation):
        try:return operation()
        except (OSError,KeyError,TypeError,ValueError) as exc:
            audit.errors.append(f'{label}: {exc}')
            return None
    for name,(category,n) in COMMON.items():
        attempt(name,lambda name=name,category=category,n=n:audit.worker(root/name,n,category,'shared_research_qualification','development',{'base'}))
    for case in ('temporal','spatial'):
        value=attempt(f'{case} adaptation and selection',lambda case=case:audit_adaptation(audit,root/f'{case}-adaptation',case))
        if value:adaptation_intervals.append(value[0]);sweep_intervals.append(value[1])
    attempt('preserved attempts',lambda:preserved_attempts(audit,root))
    def heldout():
        report=audit.read(report_path)
        if report.get('mode')!='complete_comparison' or not report['verification'].get('complete_and_verified') or report['evidence_counts']['new_policy_episodes']!=640:
            raise ValueError('Verified complete 640-episode test analysis is required')
        for case,identity in report['verification']['worker_identities'].items():
            worker_root=Path(identity['root'])
            plan=audit.worker(worker_root,320,'final_comparison','evaluation_only','heldout',{'base','saved_rewrite','closed_form','lora'})
            dispatch=plan['rewrite_dispatch']
            if dispatch['calls']!=10 or len(dispatch['records'])!=5:raise ValueError('Expected ten measured plan-time router calls per case')
            for mode in ('semantic','exact'):
                values=[seconds(row[mode]['cpu_wall_seconds']) for row in dispatch['records']]
                if any(value is None for value in values):raise ValueError('Missing measured rewrite-dispatch call time')
                audit.component(f'{mode}_rewrite_dispatch','rewrite_plan_preparation',case,worker_root/'plan.json',
                                wall_seconds=sum(values),calls=len(values),mean_call_seconds=sum(values)/len(values),
                                execution_seconds=None,queue_seconds=None,
                                note='One CPU call per mode/prompt at plan creation; lightweight measurement, not online inference latency or a benchmark.')
    attempt('final comparison and routing',heldout)
    policy_rows=[row for row in audit.components if 'episodes' in row]
    common=sum(row['episodes'] for row in policy_rows if row['scope']=='shared_research_qualification')
    selection=sum(row['episodes'] for row in policy_rows if row['category']=='lora_closed_loop_selection')
    test=sum(row['episodes'] for row in policy_rows if row['category']=='final_comparison')
    if (common,selection,test)!=(256,128,640):audit.errors.append(f'Incomplete episode accounting: common={common}, selection={selection}, test={test}')
    scopes={scope:[row for row in audit.components if row['scope']==scope and row['additive_wall_component']]
            for scope in sorted({row['scope'] for row in audit.components})}
    result={'mode':'partial_cost_inventory' if allow_partial or audit.errors else 'complete_receipt_cost_audit',
            'receipt_coverage_complete':not audit.errors,'errors':audit.errors,
            'completed_policy_episodes':{'shared_development_qualification_screen':common,'lora_selection':selection,
                                         'final_comparison':test,'total':common+selection+test,
                                         'derived_exact_cache_new_episodes':0},
            'scope_totals':{scope:{key:totals(rows,key) for key in ('wall_seconds','execution_seconds','queue_seconds')} for scope,rows in scopes.items()},
            'parallel_runner_intervals':{'original_adaptation':parallel_interval(adaptation_intervals),
                                         'additional_lora_sweep':parallel_interval(sweep_intervals),
                                         'common_policy_collection_elapsed_seconds':None,
                                         'policy_elapsed_note':'Historical policy receipts lack complete start/end intervals; summed batch wall time is not divided by GPU count.'},
            'unmeasured_costs':{'manual_wording_and_engineering_seconds':None,'historical_queue_execution_split':None,
                                'cold_start_latency':None,'unreceipted_failed_policy_execution_seconds':None},
            'interpretation':'Shared qualification costs, method-specific adaptation, and final evaluation are separate. Wall-time sums overlap across GPUs and include measured waits/loading. No unknown timing is represented as zero.',
            'components':audit.components,'source_sha256':audit.sources,'analysis_code_sha256':sha(__file__)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--analysis-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();result=build(args.run_root,args.analysis_report,args.allow_partial)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'costs.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=result['components']
    with (args.output/'costs.csv').open('w',newline='') as stream:
        if rows:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader();writer.writerows(rows)
    if result['errors'] and not args.allow_partial:parser.exit(2,'Cost audit incomplete; saved source/error inventory without claiming final completeness.\n')
    print(json.dumps({'mode':result['mode'],'completed_policy_episodes':result['completed_policy_episodes']}))


if __name__=='__main__':main()
