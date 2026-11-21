"""Read-only milestone monitor; never executes or modifies experiments."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

REPO=Path(__file__).resolve().parents[4]
COHORT=REPO/'projects/rss-few-pair-validation-20261120'
OUTPUT=REPO/'projects/rss-few-pair-monitor-20261120'
GOALS=('temporal','cube_behind','mustard_right')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)

def snapshot(root=COHORT):
    root=Path(root);events=[];states={};completed={'collection':0,'selection':0,'heldout':0}
    for path in root.glob('**/completed.json'):
        if not (path.parent/'episodes.json').is_file():continue
        receipt=read(path);rows=read(path.parent/'episodes.json')
        if receipt['episodes_sha256']!=sha(path.parent/'episodes.json') or receipt['n']!=len(rows):
            raise ValueError('Completed receipt failed integrity: '+str(path))
        parts=path.relative_to(root).parts
        if len(parts)>1 and parts[1] in completed:completed[parts[1]]+=len(rows)
    for path in [root/'collection-status.json',root/'supervisor-status.json']:
        if not path.exists():continue
        record=read(path);states[path.name]=record
        if record['status']=='stopped':
            events.append({'id':'stopped:'+sha(path),'kind':'technical_stop',
                'summary':f'{path.name} reports stopped. Inspect {path}. No automatic retry is authorized.'})
    collection=states.get('collection-status.json',{})
    if collection.get('status')=='complete':
        qualifications={gid:read(root/gid/'source-status.json') for gid in GOALS}
        for result in qualifications.values():
            if sha(result['manifest'])!=result['manifest_sha256']:raise ValueError('Source manifest changed')
        events.append({'id':'source-complete','kind':'source_qualification_complete',
            'summary':'Fresh source qualification completed: '+json.dumps({g:{k:r[k] for k in ('status','attempted_pairs','eligible_pairs')} for g,r in qualifications.items()})+
            '. This is acquisition evidence, not an edited-policy result.'})
    final=root/'analysis-v1/report.json'
    if final.exists():
        report=read(final)
        if report['status']!='complete_and_verified' or report['registered_goals']!=3:
            raise ValueError('Unexpected final report')
        events=[{'id':'final:'+sha(final),'kind':'final',
            'summary':f'All registered goals have a verified disposition. Report: {final}. '+
            f"Qualified goals: {report['qualified_goals']}/3; held-out episodes: {report['new_heldout_policy_episodes']}."}]
    else:
        for gid in GOALS:
            path=root/gid/'analysis-v1/report.json'
            if not path.exists():continue
            report=read(path)
            if report['status']=='complete_and_verified':
                events.append({'id':gid+':'+sha(path),'kind':'goal_comparison_complete',
                    'summary':f'Completed matched comparison for {gid}: {path}. Read primary and paired results before interpreting.'})
    return {'checked_unix':time.time(),'completed_receipted_episodes':completed,'states':states,'events':events,'final':final.is_file()}

def message(event,snapshot):
    return ('[USER-AUTHORIZED WEIGHTSMITH MONITOR — MACHINE-GENERATED UPDATE]\n'
        'A read-only milestone from the existing request to keep checking Weightsmith. Respect any later cancellation. '
        f"Event: {event['kind']}. {event['summary']}\n"
        f"Receipted episodes by stage: {snapshot['completed_receipted_episodes']}. Snapshot Unix: {snapshot['checked_unix']}. "
        'Freshly verify current receipts before replying. Explain whether completed evidence supports the few-source-pair repair claim, '
        'its repeatability across source groups and goals, competing-goal preservation, and advantage over the declared embedding/residual baselines. '
        'Separate source qualification from edited-policy evaluation, and complete matched comparisons from partial cells. '
        'Do not pool shared controls as independent source-group replications. Do not extrapolate pending tests or conceal failed qualification. '
        'If no new complete comparison exists, say the scientific assessment is unchanged. '
        'Reporting only: do not restart, retune, rerun, change frozen cohorts, rewrite/push the paper, or resume ICWM. '
        f'Experiment root: {COHORT}. Monitoring output: {OUTPUT}.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true')
    ap.add_argument('--notify-thread');ap.add_argument('--codex-bin',type=Path)
    ap.add_argument('--max-seconds',type=int,default=36*3600);args=ap.parse_args()
    if not 1<=args.max_seconds<=36*3600:ap.error('Maximum monitor duration is36hours')
    if bool(args.notify_thread)!=bool(args.codex_bin):ap.error('Supply both notification parameters')
    if args.notify_thread and not re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',args.notify_thread):ap.error('Invalid thread')
    if args.codex_bin and (not args.codex_bin.is_absolute() or not args.codex_bin.is_file()):ap.error('Missing executable')
    OUTPUT.mkdir(exist_ok=True);lock=(OUTPUT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(OUTPUT/'state.json') if (OUTPUT/'state.json').exists() else {'started_unix':time.time(),'attempted_events':{}}
    deadline=state['started_unix']+args.max_seconds
    while True:
        try:snap=snapshot()
        except Exception as exc:
            snap={'checked_unix':time.time(),'completed_receipted_episodes':{},'final':False,'events':[
                {'id':'audit-error:'+hashlib.sha256(repr(exc).encode()).hexdigest(),'kind':'audit_error','summary':repr(exc)}]}
        save(OUTPUT/'snapshot.json',snap)
        if args.once:print(json.dumps(snap));return
        for event in snap['events']:
            if event['id'] in state['attempted_events']:continue
            state['attempted_events'][event['id']]={'attempted_unix':time.time(),'status':'recorded_before_delivery'}
            save(OUTPUT/'state.json',state)
            delivery={'status':'notifications_disabled'}
            if args.notify_thread:
                try:
                    result=subprocess.run([str(args.codex_bin),'queue','--thread',args.notify_thread,'--message',message(event,snap),'--resume'],
                        capture_output=True,text=True,timeout=55)
                    delivery={'status':'queue_command_completed_delivery_unverified','returncode':result.returncode,
                              'output':(result.stdout+'\n'+result.stderr)[-3000:]}
                except Exception as exc:delivery={'status':'delivery_uncertain','error':repr(exc)}
            state['attempted_events'][event['id']]['delivery']=delivery;save(OUTPUT/'state.json',state)
            with (OUTPUT/'events.jsonl').open('a') as f:f.write(json.dumps({'event':event,'delivery':delivery})+'\n')
        if snap['final'] or time.time()>=deadline:
            state['status']='finished' if snap['final'] else 'deadline_reached';save(OUTPUT/'state.json',state);return
        state['status']='monitoring';state['last_checked_unix']=time.time();save(OUTPUT/'state.json',state)
        time.sleep(60)

if __name__=='__main__':main()
