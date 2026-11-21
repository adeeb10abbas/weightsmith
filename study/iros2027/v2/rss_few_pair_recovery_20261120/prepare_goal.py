"""Advance one qualified goal through its original development comparison."""
from __future__ import annotations
import argparse
import fcntl
import signal
import time
from recovery import HERE,STUDY,RECROOT,AMENDMENT,amendment,command,goal_root,protocol,read,save,sha,verify,run_commands,stop_on_signal,worker_commands

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--goal',required=True,choices=protocol()['goal_ids'])
    ap.add_argument('--gpu',type=int,choices=(0,1),required=True);args=ap.parse_args()
    amendment();out=RECROOT/args.goal;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop_on_signal);signal.signal(signal.SIGINT,stop_on_signal)
    root=goal_root(args.goal);state={'status':'running','goal_id':args.goal,'started_unix':time.time()}
    def update(**values):state.update(values,updated_unix=time.time());save(out/'prepare-status.json',state)
    try:
        source=read(root/'source-status.json')
        if source['status']!='qualified':raise ValueError('Goal has no qualified nine-pair source set')
        if sha(source['manifest'])!=source['manifest_sha256']:raise ValueError('Source identity changed')
        status=root/'adaptation/status.json'
        if not status.exists():
            update(stage='constructing_edits')
            run_commands([command('adaptation/run.py','--goal',args.goal,'--gpu',args.gpu)],out/'adaptation',timeout=7200)
        update(stage='waiting_for_existing_edit_construction')
        deadline=time.monotonic()+7200
        while True:
            r=read(status)
            if r['status']=='complete':break
            if r['status']=='stopped':raise RuntimeError('Edit construction stopped: '+r.get('reason',''))
            if time.monotonic()>deadline:raise TimeoutError('Edit construction did not finish')
            time.sleep(5)
        registry=root/'adaptation/weightsmith/candidates.json';verify(read(registry)['artifacts'])
        update(stage='constructing_simpler_baselines')
        baseline=root/'adaptation/baselines/candidates.json'
        if not baseline.exists():
            run_commands([command('prepare_baselines.py','--goal',args.goal,'--gpu',args.gpu)],out/'baselines',timeout=7200)
        verify(read(baseline)['artifacts'])
        for split in ('development','heldout'):
            update(stage='checking_'+split+'_resets')
            receipt=root/f'preflight/{split}/receipt.json'
            if receipt.exists():verify(read(receipt)['artifacts'])
            else:run_commands([command('preflight.py','--goal',args.goal,'--split',split,'--gpu',args.gpu)],out/('reset-'+split),timeout=3600)
        update(stage='freeze_development_selection')
        import evaluate
        evaluate.prepare_selection(args.goal)
        update(stage='development_selection')
        commands=worker_commands(args.goal,'selection')
        if commands:run_commands(commands,out/'selection',timeout=21600)
        from analyze import collect_stage
        groups,receipts,artifacts=collect_stage(args.goal,'selection')
        if sum(x['n'] for x in receipts)!=208:raise ValueError('Incomplete development matrix')
        ready={'status':'ready','goal_id':args.goal,'development_episodes':208,
            'source_manifest':source['manifest'],'source_manifest_sha256':source['manifest_sha256'],
            'artifacts':{**artifacts,str(AMENDMENT):sha(AMENDMENT),str(HERE/'prepare_goal.py'):sha(HERE/'prepare_goal.py')},
            'heldout_policy_episodes':0,'completed_unix':time.time()}
        save(out/'ready-for-heldout.json',ready,exclusive=True)
        update(status='complete',stage='ready_for_global_heldout_freeze')
    except BaseException as exc:
        update(status='stopped',reason=repr(exc));raise

if __name__=='__main__':main()
