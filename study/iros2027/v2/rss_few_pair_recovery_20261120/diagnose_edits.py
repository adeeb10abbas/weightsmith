"""Fixed, separately reported development check of every new source-triplet edit."""
from __future__ import annotations
import argparse
import fcntl
import signal
import time
from pathlib import Path
from recovery import HERE,STUDY,RECROOT,AMENDMENT,amendment,command,goal_root,protocol,read,save,sha,verify,run_commands,stop_on_signal
from common import canonical,freeze_plan,goal,native_sources,slot_start

GOALS=('temporal','cube_behind')

def design():
    path=RECROOT/'development-edit-check/design.json'
    record={'cohort_id':protocol()['cohort_id'],'stage':'additional_development_check',
        'goal_ids':list(GOALS),'conditions':['base','weightsmith_g0','weightsmith_g1','weightsmith_g2'],
        'episodes_per_condition':16,'maximum_new_policy_episodes':128,
        'layouts':'Existing development layouts, excluding source collection and held-out layouts.',
        'comparison':'One unchanged reference plus all three predetermined edits per goal, same layouts/noise.',
        'use':'Exploratory behavioral diagnosis; no edit selection or tuning and no change to the original baseline grid or confirmatory test.',
        'report':'Report all goals and source groups, including failures. Do not merge these episodes with held-out results.',
        'amendment':str(AMENDMENT),'amendment_sha256':sha(AMENDMENT),
        'code_sha256':sha(__file__)}
    save(path,record,exclusive=True);return path

def verify_batches(goal_id,plan,owner):
    from worker import verify_global_freeze
    from analyze import contrast,interval
    verify_global_freeze(plan);verify(read(owner/'identity.json'))
    status=read(owner/'status.json')
    if status['status']!='complete' or status['completed_episodes']!=64:raise ValueError('Incomplete diagnostic matrix')
    groups={};artifacts={}
    for i,b in enumerate(plan['batches']):
        batch=owner/f'batch-{i:02d}-{b["cell"]}-{b["condition"]}'
        rec=read(batch/'completed.json');rows=read(batch/'episodes.json')
        if rec['n']!=16 or rec['episodes_sha256']!=sha(batch/'episodes.json'):raise ValueError('Diagnostic receipt changed')
        if sorted(r['slot'] for r in rows)!=list(range(b['slot_start'],b['slot_start']+16)):raise ValueError('Diagnostic slots differ')
        for row in rows:
            if row['split']!='development' or row['condition']!=b['condition'] or row['goal_id']!=goal_id or not row['ok']:
                raise ValueError('Wrong diagnostic trial')
            if row['selected_goal']!=b['goal'] or row['success']!=row['strict_final'][b['goal']]:raise ValueError('Diagnostic score mismatch')
            for key,hkey in (('obs','obs_sha256'),('metrics','metrics_sha256')):
                if sha(row[key])!=row[hkey]:raise ValueError('Diagnostic evidence changed')
                artifacts[row[key]]=row[hkey]
        if sum(bool(r['success']) for r in rows)!=rec['successes']:raise ValueError('Diagnostic totals mismatch')
        groups[b['condition']]=sorted(rows,key=lambda r:r['slot'])
        for p in (batch/'episodes.json',batch/'completed.json'):artifacts[str(p)]=sha(p)
    primary=[];paired=[]
    for name,rows in groups.items():
        values=[bool(r['success']) for r in rows]
        primary.append({'condition':name,'successes':sum(values),'n':16,'rate':sum(values)/16,
            **interval(values,[r['layout_id'] for r in rows])})
        if name!='base':paired.append(contrast(rows,groups['base'],name,'base','original'))
    return {'status':'complete_and_verified','goal_id':goal_id,'stage':'additional_development_check',
        'new_policy_episodes':64,'primary':primary,'paired':paired,'artifacts':artifacts,
        'scope':'Development-only, all predetermined source groups; no held-out result and no candidate selection.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--goal',required=True,choices=GOALS)
    ap.add_argument('--gpu',type=int,choices=(0,1),required=True);args=ap.parse_args()
    amendment();design_path=design();root=goal_root(args.goal);out=RECROOT/'development-edit-check'/args.goal
    out.mkdir(parents=True,exist_ok=True);lock=(out/'check.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop_on_signal);signal.signal(signal.SIGINT,stop_on_signal)
    state={'status':'running','goal_id':args.goal,'started_unix':time.time()}
    def update(**kw):state.update(kw,updated_unix=time.time());save(out/'status.json',state)
    try:
        update(stage='waiting_for_candidates_and_development_reset')
        registry=root/'adaptation/weightsmith/candidates.json';reset=root/'preflight/development/receipt.json'
        deadline=time.monotonic()+3600
        while not registry.exists() or not reset.exists():
            prep=RECROOT/args.goal/'prepare-status.json'
            if prep.exists() and read(prep)['status']=='stopped':raise RuntimeError('Goal preparation stopped')
            if time.monotonic()>deadline:raise TimeoutError('Candidates/reset gate not ready')
            time.sleep(5)
        record=read(registry);verify(record['artifacts']);reset_record=read(reset);verify(reset_record['artifacts'])
        from evaluate import candidate_identity,selection_cell
        candidates=[candidate_identity(c,args.goal) for c in record['candidates']]
        if [c['candidate_id'] for c in candidates]!=['weightsmith_g0','weightsmith_g1','weightsmith_g2']:
            raise ValueError('All three predetermined edits are required')
        g=goal(args.goal);batches=[selection_cell(args.goal,'base',None)]+[
            selection_cell(args.goal,c['candidate_id'],c['workspace']) for c in candidates]
        artifacts={**record['artifacts'],**reset_record['artifacts'],str(reset):sha(reset),str(registry):sha(registry),
            str(design_path):sha(design_path),str(AMENDMENT):sha(AMENDMENT),str(Path(__file__)):sha(__file__)}
        for p in native_sources(args.goal)+list((STUDY/'baselines').glob('*.py')):artifacts[str(p.resolve())]=sha(p)
        for c in candidates:artifacts.update(c['artifacts'])
        plan={'cohort_id':protocol()['cohort_id'],'goal_id':args.goal,'case':g['runtime_case'],'runtime_dir':g['runtime_dir'],
            'stage':'selection','split':'development','inference_batching':'fixed16','protocol_sha256':sha(STUDY/'protocol.json'),
            'batches':batches,'diagnostic_design_sha256':sha(design_path),'candidates':candidates}
        path=freeze_plan(out/'plan.json',plan,artifacts);owner=root/'development-edit-check'/f'g{args.gpu}'
        update(stage='running_development_policy_trials')
        run_commands([command('worker.py','--plan',path,'--root',owner,'--gpu',args.gpu)],out/'execution',timeout=10800)
        report=verify_batches(args.goal,read(path),owner);save(out/'report.json',report,exclusive=True)
        update(status='complete',stage='complete',report=str(out/'report.json'))
    except BaseException as exc:update(status='stopped',reason=repr(exc));raise

if __name__=='__main__':main()
