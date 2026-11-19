"""CPU-only tests: notification calls are mocked and no experiment is changed."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import watch as W

def snap(complete=()):
    return {'checked_unix':1000,'schedule_sha256':'sealed','completed_episodes':32,
      'groups':{cell:{'complete':cell in complete,'arms':{'base':{'successes':3,'expected':32}}} for cell in W.GROUP_TOTALS},
      'failures':[],'final':None,'last_receipt_unix':1000}

class WatchTests(unittest.TestCase):
    def test_only_complete_group_not_partial_count_notifies(self):
        state={'started_unix':1000,'attempted_events':{}}
        self.assertEqual(W.event_candidates(snap(),state,1001),[])
        events=W.event_candidates(snap(('original',)),state,1001)
        self.assertEqual([e['id'] for e in events],['group:original'])
        state['attempted_events']['group:original']={}
        self.assertEqual(W.event_candidates(snap(('original',)),state,1002),[])

    def test_final_does_not_repeat_group_notifications(self):
        s=snap(W.GROUP_TOTALS);s['final']={'sha256':'report'}
        events=W.event_candidates(s,{'started_unix':0},5000)
        self.assertEqual([e['kind'] for e in events],['final'])

    def test_failure_has_priority_over_group_and_heartbeat(self):
        s=snap(('original',));s['failures']=[{'source':'gpu0','status':{'status':'stopped'}}]
        events=W.event_candidates(s,{'started_unix':1000},5000)
        self.assertEqual([e['kind'] for e in events],['failure'])

    def test_stalled_is_one_event_per_progress_receipt(self):
        s=snap();state={'started_unix':1000,'attempted_events':{}}
        events=W.event_candidates(s,state,2800)
        self.assertEqual(set(e['kind'] for e in events),{'stalled','heartbeat'})
        state['attempted_events']={e['id']:{} for e in events}
        self.assertEqual(W.event_candidates(s,state,2810),[])
        s['last_receipt_unix']=2810
        self.assertFalse(any(e['kind']=='stalled' for e in W.event_candidates(s,state,4000)))

    def test_schedule_mutation_stops_audit(self):
        with self.assertRaises(W.AuditError):W.event_candidates(snap(),{'schedule_sha256':'other'},1000)

    @patch('watch.subprocess.run')
    def test_default_notification_never_executes_codex(self,run):
        r=W.notify({'kind':'heartbeat','summary':'Example'},snap(),None,None)
        self.assertEqual(r['status'],'notification_disabled');run.assert_not_called()

    @patch('watch.subprocess.run')
    def test_warning_with_exit_zero_is_not_delivery(self,run):
        run.return_value=subprocess.CompletedProcess([],0,'Queued message abc','Warning: unable to resume thread')
        r=W.notify({'kind':'heartbeat','summary':'Example'},snap(),'thread',Path('/bin/codex'))
        self.assertEqual(r['status'],'delivery_uncertain')
        argv=run.call_args.args[0]
        self.assertEqual(argv[:4],['/bin/codex','queue','--thread','thread'])
        self.assertEqual(argv[-1],'--resume')
        self.assertIn('Respect any later cancellation',argv[-2])
        self.assertEqual(run.call_args.kwargs['timeout'],60)

    @patch('watch.subprocess.run')
    def test_success_still_does_not_claim_user_delivery(self,run):
        run.return_value=subprocess.CompletedProcess([],0,'Queued message abc','')
        r=W.notify({'kind':'heartbeat','summary':'Example'},snap(),'thread',Path('/bin/codex'))
        self.assertEqual(r['status'],'queue_command_succeeded_delivery_unverified')

    @patch('watch.subprocess.run')
    def test_timeout_is_uncertain(self,run):
        run.side_effect=subprocess.TimeoutExpired(['codex'],60)
        self.assertEqual(W.notify({'kind':'heartbeat','summary':'Example'},snap(),'thread',Path('/bin/codex'))['status'],'delivery_uncertain')

    def test_attempt_is_deduplicated_even_if_notification_fails(self):
        with tempfile.TemporaryDirectory() as td:
            state={'attempted_events':{}};event={'id':'heartbeat:1','kind':'heartbeat','summary':'Example'}
            args=type('Args',(),{'notify_thread':'thread','codex_bin':Path('/bin/codex')})()
            def verify_before_queue(*args):
                self.assertIn(event['id'],W.read(Path(td)/'state.json')['attempted_events'])
                return {'status':'delivery_uncertain'}
            with patch('watch.notify',side_effect=verify_before_queue):W.emit(event,snap(),state,Path(td),args)
            self.assertEqual(state['attempted_events'][event['id']]['notification']['status'],'delivery_uncertain')
            self.assertEqual(len((Path(td)/'events.jsonl').read_text().splitlines()),1)

if __name__=='__main__':unittest.main()
