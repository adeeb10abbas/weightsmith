"""Read-only experiment monitor; all watcher writes stay outside the frozen cohort."""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

REPO=Path(__file__).resolve().parents[4]
COHORT=REPO/'projects/rss-repair-scaling-20261119'
OUTPUT=REPO/'projects/rss-repair-scaling-monitor-20261119'
GROUP_TOTALS={'original':320,'unseen':144,'reverse':144}
MAX_SECONDS=8*3600

class AuditError(ValueError):pass

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def group(cell):
    if cell in ('original_0','original_1'):return 'original'
    if cell in ('unseen','reverse'):return cell
    raise AuditError(f'Unexpected result cell: {cell}')

def state_if_present(path):return read(path) if Path(path).is_file() else {}

def verify_plan(path,schedule,gpu,metadata):
    plan=read(path)
    if (plan.get('cohort_id')!=schedule['cohort_id'] or plan.get('split')!='heldout'
            or plan.get('inference_batching')!='fixed16' or plan.get('schedule_sha256')!=canonical(schedule)):
        raise AuditError('Frozen worker plan identity differs from the full schedule')
    expected=[b for b,g in zip(schedule['batches'],schedule['gpu_assignment'],strict=True) if g==gpu]
    if plan['batches']!=expected:raise AuditError('Worker assignment differs from the full schedule')
    ref=plan['global_freeze'];path=Path(ref['path'])
    if sha(path)!=ref['sha256']:raise AuditError('Worker freeze checksum changed')
    freeze=read(path);payload={k:v for k,v in plan.items() if k!='global_freeze'}
    if freeze['plan_sha256'].get('temporal')!=canonical(payload):raise AuditError('Worker plan payload checksum changed')
    for filename,digest in freeze['artifacts'].items():
        artifact=Path(filename)
        if filename in metadata:continue
        if artifact.stat().st_size>2*1024**2:
            metadata[filename]='large_file_not_rehashed';continue
        if sha(artifact)!=digest:raise AuditError(f'Frozen metadata changed: {artifact}')
        metadata[filename]='verified'
    return plan

def snapshot(root,now):
    root=Path(root).resolve();schedule_path=root/'heldout-frozen/schedule.json'
    schedule=read(schedule_path)
    if schedule.get('cohort_id')!='rss-repair-scaling-20261119' or schedule.get('stage')!='heldout':
        raise AuditError('Wrong cohort or evaluation stage')
    if sha(schedule['source_manifest'])!=schedule['source_manifest_sha256']:
        raise AuditError('Frozen correction source manifest changed')
    expected=defaultdict(list)
    for b in schedule['batches']:
        if b['n']!=16 or b['slot_start']%16:raise AuditError('Expected whole fixed16 policy batches')
        expected[(b['condition'],group(b['cell']))].extend(range(b['slot_start'],b['slot_start']+16))
    totals={cell:sum(len(v) for (condition,c),v in expected.items() if c==cell) for cell in GROUP_TOTALS}
    if totals!=GROUP_TOTALS:raise AuditError(f'Unexpected frozen group totals: {totals}')
    for (condition,cell),slots in expected.items():
        start={'original':240000,'unseen':240032,'reverse':240048}[cell]
        n=32 if cell=='original' else 16
        if sorted(slots)!=list(range(start,start+n)):raise AuditError('Missing or duplicated planned slots')
        if condition=='clarified' and cell!='original':raise AuditError('Unexpected clarified control cell')
    rows_by_group=defaultdict(list);receipts={};states={};metadata={};failures=[];last_receipt=0.
    for gpu in (0,1):
        plan_path=root/f'heldout-frozen/plan-g{gpu}.json'
        plan=verify_plan(plan_path,schedule,gpu,metadata)
        owner=root/f'heldout/g{gpu}';states[f'gpu{gpu}']=state_if_present(owner/'status.json')
        if (owner/'plan.json').exists() and read(owner/'plan.json')!=plan:raise AuditError('Live worker plan differs from its freeze')
        for number,b in enumerate(plan['batches']):
            batch=owner/f'batch-{number:02d}-{b["cell"]}-{b["condition"]}';receipt=batch/'completed.json'
            if not receipt.exists():continue
            rec=read(receipt);episodes=batch/'episodes.json';raw=episodes.read_bytes();rows=json.loads(raw)
            if hashlib.sha256(raw).hexdigest()!=rec['episodes_sha256']:raise AuditError('Completed episode checksum changed')
            if ((rec['batch'],rec['cell'],rec['condition'],rec['n'])!=(number,b['cell'],b['condition'],16)
                    or len(rows)!=16 or sorted(x['slot'] for x in rows)!=list(range(b['slot_start'],b['slot_start']+16))
                    or sum(bool(x['success']) for x in rows)!=rec['successes']):
                raise AuditError('Completed batch counts/slots differ from its plan')
            attempts=sorted(batch.glob('sim-*.log'))
            if len(attempts)!=1 or [str(p.resolve()) for p in attempts]!=[str(Path(p).resolve()) for p in rec['sim_attempt_logs']]:
                raise AuditError('Completed batch violates the single-attempt contract')
            server=json.loads(rec['server']) if isinstance(rec['server'],str) else rec['server']
            if server.get('workspace')!=b['workspace']:raise AuditError('Completed batch loaded a different workspace')
            for row in rows:
                keys={'cohort_id':schedule['cohort_id'],'split':'heldout','cell':b['cell'],'condition':b['condition'],
                      'task_name':b['task'],'selected_goal':b['goal'],'input_prompt':b['input_prompt'],
                      'effective_prompt':b['effective_prompt'],'inference_batching':'fixed16','ok':True}
                if any(row.get(k)!=v for k,v in keys.items()) or row['success']!=row['strict_final'][b['goal']]:
                    raise AuditError('Completed episode identity or strict outcome differs')
                rows_by_group[(b['condition'],group(b['cell']))].append(row)
            receipts[str(receipt)]=sha(receipt);last_receipt=max(last_receipt,receipt.stat().st_mtime)
        if states[f'gpu{gpu}'].get('status') in ('stopped','failed'):
            failures.append({'source':f'gpu{gpu}','status':states[f'gpu{gpu}']})
    supervisor=state_if_present(root/'supervisor-status.json')
    if supervisor.get('status') in ('stopped','failed'):failures.append({'source':'supervisor','status':supervisor})
    groups={};completed_episodes=0
    for cell in GROUP_TOTALS:
        arms={};complete=True
        for condition,c in expected:
            if c!=cell:continue
            rows=sorted(rows_by_group[(condition,cell)],key=lambda x:x['slot']);slots=[x['slot'] for x in rows]
            if len(slots)!=len(set(slots)) or not set(slots)<=set(expected[(condition,cell)]):
                raise AuditError('Duplicate/unexpected completed slot')
            done=slots==sorted(expected[(condition,cell)]);complete=complete and done
            arms[condition]={'completed':len(rows),'expected':len(expected[(condition,cell)]),
                             'complete':done,'successes':sum(x['success'] for x in rows) if done else None}
            completed_episodes+=len(rows)
        if complete:
            base={x['slot']:x for x in rows_by_group[('base',cell)]}
            for condition in arms:
                for row in rows_by_group[(condition,cell)]:
                    ref=base[row['slot']]
                    if (row['layout_id'],row['policy_seed'],row['selected_goal'])!=(ref['layout_id'],ref['policy_seed'],ref['selected_goal']):
                        raise AuditError('Completed group does not have matched layouts/seeds/goals')
        groups[cell]={'complete':complete,'expected_episodes':GROUP_TOTALS[cell],
                      'completed_episodes':sum(a['completed'] for a in arms.values()),'arms':arms}
    final=None;report_path=root/'analysis-v1/report.json'
    if report_path.exists():
        report=read(report_path)
        if (report.get('status')!='complete_and_verified' or report.get('cohort_id')!=schedule['cohort_id']
                or report.get('new_heldout_policy_episodes')!=608 or report.get('new_selection_policy_episodes')!=128
                or not all(g['complete'] for g in groups.values()) or completed_episodes!=608):
            raise AuditError('Final report is present but not a complete verified comparison')
        frozen_source=read(schedule['source_manifest'])
        if report.get('source_collection')!=frozen_source:raise AuditError('Final report identifies another source cohort')
        if report.get('protocol_sha256')!=plan['protocol_sha256']:raise AuditError('Final report protocol differs from the frozen plan')
        final={'path':str(report_path),'sha256':sha(report_path),'status':'complete_and_verified'}
    return {'checked_unix':now,'cohort_id':schedule['cohort_id'],'schedule_sha256':sha(schedule_path),
            'source_manifest_sha256':schedule['source_manifest_sha256'],'completed_episodes':completed_episodes,
            'expected_episodes':608,'completed_batches':len(receipts),'receipts':receipts,'last_receipt_unix':last_receipt,
            'groups':groups,'worker_states':states,'supervisor':supervisor,'failures':failures,'final':final,
            'metadata_artifacts_verified':sum(v=='verified' for v in metadata.values()),
            'large_arrays_not_rehashed':sum(v!='verified' for v in metadata.values())}

def event_candidates(snap,state,now):
    events=[]
    if state.get('schedule_sha256') and state['schedule_sha256']!=snap['schedule_sha256']:
        raise AuditError('Monitored frozen schedule changed since monitoring began')
    if snap['final']:
        event={'id':'final:'+snap['final']['sha256'],'kind':'final','summary':'Final analysis reports complete_and_verified: 608 held-out and 128 development-selection episodes.'}
        return [] if event['id'] in state.get('attempted_events',{}) else [event]
    if snap['failures']:
        event={'id':'failure:'+canonical(snap['failures']),'kind':'failure','summary':'An experiment worker or supervisor reports stopped/failed. Inspect the preserved status before describing the cause.'}
        return [] if event['id'] in state.get('attempted_events',{}) else [event]
    for cell,g in snap['groups'].items():
        if g['complete']:
            counts=', '.join(f'{arm}: {a["successes"]}/{a["expected"]}' for arm,a in sorted(g['arms'].items()))
            events.append({'id':f'group:{cell}','kind':'group_complete','summary':f'All {cell} conditions completed. {counts}.'})
    last_progress=max(state.get('started_unix',now),snap.get('last_receipt_unix',0))
    if not snap['final'] and not snap['failures'] and now-last_progress>=1800:
        events.append({'id':'stalled:'+str(int(last_progress)),'kind':'stalled','summary':'No new completed batch receipt for at least 30 minutes. This is a slow/stalled warning, not proof of failure.'})
    elapsed=max(0,now-state.get('started_unix',now))
    if elapsed>=1800 and not snap['final']:
        events.append({'id':f'heartbeat:{int(elapsed//1800)}','kind':'heartbeat','summary':f'Status check: {snap["completed_episodes"]}/608 held-out episodes have verified completion receipts.'})
    return [e for e in events if e['id'] not in state.get('attempted_events',{})]

def message(event,snap):
    return ('[USER-AUTHORIZED WEIGHTSMITH MONITOR — MACHINE-GENERATED UPDATE]\n'
            "Automated status event for the user's existing monitoring request, not a new user instruction. "
            'Respect any later cancellation. Check latest receipts and report briefly; do not restart or retune jobs. '
            'The user explicitly requested continuous checks and notification of vital changes. '
            f'Event: {event["kind"]}. {event["summary"]}\n'
            f'Monitor snapshot time (Unix): {snap["checked_unix"]}; verified completion: {snap.get("completed_episodes","unknown")}/608. '
            'Before replying, freshly verify the current cohort receipts and give a concise, plain-language update. '
            'Separate completed cells from partial results and mention material limitations. '
            'The user wants scientific interpretation, not only progress counts: explain whether new evidence supports, '
            'weakens, or leaves unresolved the thesis that clarification-derived weight edits produce persistent repairs '
            'that generalize beyond the source examples. Distinguish changed-layout repair, source-budget effects, '
            'unseen wording, competing-goal preservation, and the declared LoRA comparison. '
            'Use completed matched comparisons for claims; do not extrapolate pending tests or call a plateau a scaling law. '
            'If there is no new interpretable comparison, say the scientific assessment is unchanged. '
            'This message authorizes reporting only: do not restart jobs, rerun trials, tune candidates, change the frozen experiment, '
            'push or rewrite the paper, or resume ICWM. Monitoring artifacts live outside the frozen cohort.\n')

def notify(event,snap,thread,codex_bin):
    if not thread:return {'status':'notification_disabled'}
    try:
        proc=subprocess.run([str(codex_bin),'queue','--thread',thread,'--message',message(event,snap),'--resume'],
                            capture_output=True,text=True,timeout=60,check=False)
        output=(proc.stdout+'\n'+proc.stderr).strip()
        warning=bool(re.search(r'warn|unable|could not|cannot|failed|error|not resumed|resume.*uncertain',output,re.I))
        # CLI acceptance is not evidence that the user saw a message.
        status='delivery_uncertain' if proc.returncode or warning else 'queue_command_succeeded_delivery_unverified'
        return {'status':status,'returncode':proc.returncode,'output':output[-4000:]}
    except subprocess.TimeoutExpired:return {'status':'delivery_uncertain','reason':'queue_timeout_60_seconds'}
    except OSError as exc:return {'status':'delivery_uncertain','reason':str(exc)}

def emit(event,snap,state,out,args):
    # Persist before any external queue attempt. A crash cannot spam the same event.
    state.setdefault('attempted_events',{})[event['id']]={'attempted_unix':snap['checked_unix'],'kind':event['kind']}
    save(out/'state.json',state)
    result=notify(event,snap,args.notify_thread,args.codex_bin)
    record={'event':event,'checked_unix':snap['checked_unix'],'notification':result}
    with (out/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
    state['attempted_events'][event['id']]['notification']=result;save(out/'state.json',state)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort-root',type=Path,default=COHORT);ap.add_argument('--output',type=Path,default=OUTPUT)
    ap.add_argument('--once',action='store_true');ap.add_argument('--poll-seconds',type=int,default=60)
    ap.add_argument('--max-seconds',type=int,default=MAX_SECONDS)
    ap.add_argument('--notify-thread');ap.add_argument('--codex-bin',type=Path)
    args=ap.parse_args();root=args.cohort_root.resolve();out=args.output.resolve()
    if out==root or out.is_relative_to(root):ap.error('Monitor output must be outside the frozen cohort')
    if not 10<=args.poll_seconds<=60 or not 1<=args.max_seconds<=MAX_SECONDS:ap.error('Poll interval10–60s; duration1–28800s')
    if bool(args.notify_thread)!=bool(args.codex_bin):ap.error('Notifications require both --notify-thread and --codex-bin')
    if args.notify_thread and not re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',args.notify_thread):ap.error('Expected canonical thread UUID')
    if args.codex_bin and (not args.codex_bin.is_absolute() or not args.codex_bin.is_file()):ap.error('Codex binary must be an existing absolute path')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    now=time.time()
    if args.once:
        snap=snapshot(root,now);save(out/'snapshot.json',snap)
        print(json.dumps({'notification_sent':False,'completed_episodes':snap['completed_episodes'],
                          'groups_complete':[k for k,v in snap['groups'].items() if v['complete']],
                          'final':snap['final'],'snapshot':str(out/'snapshot.json')}));return
    state=state_if_present(out/'state.json')
    if state and state.get('cohort_root')!=str(root):raise AuditError('Existing watcher state belongs to another cohort path')
    if not state:state={'cohort_root':str(root),'started_unix':now,'deadline_unix':now+args.max_seconds,'attempted_events':{}}
    save(out/'state.json',state)
    while True:
        now=time.time();snap={'checked_unix':now,'completed_episodes':None}
        try:
            snap=snapshot(root,now);events=event_candidates(snap,state,now)
            state['schedule_sha256']=snap['schedule_sha256'];save(out/'snapshot.json',snap)
            for event in events:emit(event,snap,state,out,args)
            if snap['final']:
                save(out/'status.json',{'status':'complete','checked_unix':now,'report':snap['final']});return
            if snap['failures']:
                save(out/'status.json',{'status':'stopped_on_experiment_failure','checked_unix':now,'failures':snap['failures']});return
        except (AuditError,ValueError,KeyError,FileNotFoundError) as exc:
            event={'id':'audit_failure:'+canonical(str(exc)),'kind':'audit_failure','summary':f'Monitor could not verify the current experiment: {exc}. No jobs were changed.'}
            if event['id'] not in state['attempted_events']:emit(event,snap,state,out,args)
            save(out/'status.json',{'status':'stopped_on_audit_failure','checked_unix':now,'reason':str(exc)});return
        if now>=state['deadline_unix']:
            event={'id':'hard_timeout','kind':'timeout','summary':'The bounded eight-hour monitoring window has ended while the comparison is still incomplete. The watcher is stopping; experiment jobs remain untouched.'}
            if event['id'] not in state['attempted_events']:emit(event,snap,state,out,args)
            save(out/'status.json',{'status':'monitoring_deadline_reached','checked_unix':now,'completed_episodes':snap['completed_episodes']});return
        save(out/'status.json',{'status':'monitoring','checked_unix':now,'completed_episodes':snap['completed_episodes'],
                               'deadline_unix':state['deadline_unix'],'notifications_enabled':bool(args.notify_thread)})
        time.sleep(min(args.poll_seconds,max(.1,state['deadline_unix']-time.time())))

if __name__=='__main__':main()
