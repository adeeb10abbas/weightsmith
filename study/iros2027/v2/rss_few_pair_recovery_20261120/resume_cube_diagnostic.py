"""Resume a dependency-only stop before this diagnostic made any policy call."""
import sys
import time
from recovery import RECROOT,goal_root,read,save,sha

root=goal_root('cube_behind');out=RECROOT/'development-edit-check/cube_behind'
status=read(out/'status.json')
if status['status']!='stopped' or status['stage']!='waiting_for_candidates_and_development_reset':
    raise ValueError('Recovery applies only to the recorded dependency stop')
if (out/'execution').exists() or (root/'development-edit-check').exists():
    raise ValueError('A policy attempt exists; separate recovery assessment required')
save(out/'dependency-recovery.json',{'status':'authorized','reason':'Baseline preparation failed before the independent edit diagnostic could obtain a reset certificate.',
    'prior_status':status,'prior_status_sha256':sha(out/'status.json'),
    'new_policy_episodes_in_failed_attempt':0,
    'change':'Reuse unchanged diagnostic design after independently qualifying the original development layouts; original baseline failure is preserved.'},exclusive=True)
deadline=time.monotonic()+7200
while not (root/'preflight/development/receipt.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Cube development reset receipt not available')
    time.sleep(5)
import diagnose_edits
sys.argv=[str(diagnose_edits.__file__),'--goal','cube_behind','--gpu','1']
diagnose_edits.main()
