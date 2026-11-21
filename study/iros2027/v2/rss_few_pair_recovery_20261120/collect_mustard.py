"""Resume untouched mustard acquisition after batch-matched reset qualification."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
import signal
import time
from recovery import HERE,STUDY,RECROOT,AMENDMENT,amendment,command,goal_root,protocol,read,save,sha,verify,run_commands,stop_on_signal
from common import freeze_plan,goal,slot_start
from prepare_sources import export

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reset-receipt',type=Path,required=True);args=ap.parse_args()
    amendment();gid='mustard_right';root=goal_root(gid);out=RECROOT/gid;out.mkdir(exist_ok=True)
    lock=(out/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop_on_signal);signal.signal(signal.SIGINT,stop_on_signal)
    state={'status':'running','goal_id':gid,'started_unix':time.time()}
    def update(**kw):state.update(kw,updated_unix=time.time());save(out/'collection-status.json',state)
    try:
        update(stage='waiting_for_batch_matched_reset_certificate')
        deadline=time.monotonic()+3600
        while not args.reset_receipt.exists():
            if time.monotonic()>deadline:raise TimeoutError('Batch-isolated reset certificate not available')
            time.sleep(5)
        cert=read(args.reset_receipt);verify(cert['artifacts'])
        layout=root/'design/collection-layouts.json'
        if (cert['status']!='passed' or cert['goal_id']!=gid or cert['layout_count']!=48 or cert['reset_cycles']!=6
                or cert['policy_calls']!=0 or cert['action_calls']!=0 or cert['geometry_changed']
                or cert['scoring_changed'] or cert['tolerances_changed'] or cert['original_layouts_sha256']!=sha(layout)):
            raise ValueError('Certificate does not qualify all original poses unchanged')
        if [p['group'] for p in cert['partitions']]!=[0,1,2]:raise ValueError('Incorrect reset partition order')
        retained=[row for part in cert['partitions'] for row in read(part['path'])['layouts']]
        if retained!=read(layout)['layouts']:raise ValueError('Reset partitions changed or omitted original poses')
        seal=read(root/'design/collection-seal.json');verify(seal['artifacts'])
        artifacts={**seal['artifacts'],**cert['artifacts'],str(args.reset_receipt.resolve()):sha(args.reset_receipt),
            str(AMENDMENT):sha(AMENDMENT),str(Path(__file__).resolve()):sha(__file__)}
        recovery={'status':'accepted','change':'Fresh simulator process per16-layout preflight partition, matching each policy batch.',
            'geometry_changed':False,'scoring_changed':False,'tolerances_changed':False,'policy_outcomes_used':False,
            'reset_certificate':str(args.reset_receipt.resolve()),'reset_certificate_sha256':sha(args.reset_receipt),
            'artifacts':artifacts}
        save(out/'collection-recovery.json',recovery,exclusive=True)
        artifacts[str(out/'collection-recovery.json')]=sha(out/'collection-recovery.json')
        g=goal(gid)
        for block in range(3):
            commands=[]
            for gpu,cell in enumerate(('original','teacher')):
                owner=root/f'collection/block-{block:02d}'/cell
                if (owner/'status.json').exists():
                    if read(owner/'status.json')['status']=='complete':continue
                    raise ValueError('Partial policy batch preserved: '+str(owner))
                batch={'cell':cell,'condition':'base','task':g['task'],'goal':g['goal'],
                    'input_prompt':g['original'],'effective_prompt':g['original'] if cell=='original' else g['teacher'],
                    'workspace':None,'layouts':str(layout),'slot_start':slot_start(gid,'collection')+16*block,'n':16}
                plan={'cohort_id':protocol()['cohort_id'],'goal_id':gid,'case':g['runtime_case'],'runtime_dir':g['runtime_dir'],
                    'stage':'collection','split':'development','inference_batching':'fixed16',
                    'protocol_sha256':sha(STUDY/'protocol.json'),'batches':[batch],
                    'technical_recovery':str(out/'collection-recovery.json')}
                path=freeze_plan(out/f'block-{block:02d}-{cell}-plan.json',plan,artifacts)
                commands.append(command('worker.py','--plan',path,'--root',owner,'--gpu',gpu))
            update(stage='source_policy_trials',block=block)
            if commands:run_commands(commands,out/f'collection-block-{block:02d}',timeout=7200)
            result=export(gid,block+1);update(source_status=result)
            if result['status'] in ('qualified','unqualified_at_cap'):break
        update(status='complete',stage='source_qualification_complete',qualification=result['status'])
    except BaseException as exc:update(status='stopped',reason=repr(exc));raise

if __name__=='__main__':main()
