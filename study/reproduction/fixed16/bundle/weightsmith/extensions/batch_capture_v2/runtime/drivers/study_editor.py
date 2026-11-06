#!/usr/bin/env python3
"""Bounded local Astra driver with an operator-owned MCP tool broker.

Authentication uses the existing local Codex login unchanged; no credentials are
copied or sent to the GPU host. No model call on --prepare-only.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from study_broker import atomic_json, Broker, summarize_study

FEATURES_OFF=['shell_tool','unified_exec','code_mode','multi_agent','multi_agent_v2','apps','plugins','remote_plugin',
              'browser_use','browser_use_external','in_app_browser','computer_use','image_generation',
              'view_image','memories','hooks','plugin_hooks','goals']

def config_args(config):
    # No workspace-wide read permission: only minimal runtimes and the per-run workspace.
    settings={
        'model_reasoning_effort':'max','approval_policy':'never','web_search':'disabled',
        'default_permissions':'wsp-editor','permissions.wsp-editor.filesystem.:root':'deny',
        'permissions.wsp-editor.filesystem.:minimal':'read',
        'permissions.wsp-editor.filesystem.:workspace_roots':{'.':'write','.git':'read','.codex':'read','.agents':'read','BRIEF.md':'read','PRIOR_FINDINGS.md':'read','MODE.md':'read'},
        'permissions.wsp-editor.filesystem.:tmpdir':'deny','permissions.wsp-editor.filesystem.:slash_tmp':'deny',
        'permissions.wsp-editor.network.enabled':False,
        'project_doc_max_bytes':0,'features.skip_host_skill_discovery':True,
        'agents.enabled':False,'mcp_servers.wsp.default_tools_approval_mode':'approve',
        'mcp_servers.wsp.command':sys.executable,
        'mcp_servers.wsp.args':[str(HERE/'study_broker.py'),'--config',str(config)],
        'mcp_servers.wsp.required':True,'mcp_servers.wsp.startup_timeout_sec':30,
        'mcp_servers.wsp.tool_timeout_sec':7200,
    }
    for f in FEATURES_OFF:settings['features.'+f]=False
    args=[]
    for key,value in settings.items():
        # Quote dotted path segments with colons; dictionary values are TOML inline tables.
        # CLI override keys are split on dots; colon segments must remain unquoted.
        if isinstance(value,dict):encoded='{'+','.join(json.dumps(k)+'='+json.dumps(v) for k,v in value.items())+'}'
        else:encoded=json.dumps(value)
        args+=['-c',key+'='+encoded]
    return args


def stop(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait()


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--agent-dir',type=Path,required=True)
    ap.add_argument('--state-dir',type=Path,required=True)
    ap.add_argument('--host',default='runpod2')
    ap.add_argument('--remote-repo',required=True)
    ap.add_argument('--remote-root',required=True)
    ap.add_argument('--remote-python',default='python3')
    ap.add_argument('--mode',choices=['adaptive','nonadaptive'],required=True)
    ap.add_argument('--base-records',type=Path,help='Operator-pinned 32-row base JSONL copied from the remote project')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--readiness-only',action='store_true')
    ap.add_argument('--readiness-sentinel',type=Path,help='Optional public outside-workspace file whose denial is probed')
    ap.add_argument('--codex',default=shutil.which('codex'))
    a=ap.parse_args(argv)
    if not a.codex:raise ValueError('Codex CLI is not installed')
    if a.host.startswith('-') or not a.host:raise ValueError('invalid SSH host')
    if not Path(a.remote_repo).is_absolute() or not Path(a.remote_root).is_absolute():
        raise ValueError('bind absolute remote repository and project paths')
    agent=a.agent_dir.resolve();state=a.state_dir.resolve()
    if not agent.is_dir() or not (agent/'BRIEF.md').is_file():raise ValueError('prepared agent directory with BRIEF.md required')
    if state==agent or agent in state.parents:raise ValueError('operator state cannot be inside editor workspace')
    state.mkdir(parents=True,exist_ok=True)
    state.chmod(0o700)
    driver_lock=(state/'driver.lock').open('a')
    try:fcntl.flock(driver_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('this editor run already has an active driver')
    # Read only the presence of existing local authentication. Do not repurpose
    # HOME/CODEX_HOME or copy/link credentials into experiment artifacts.
    auth_root=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))
    if not (auth_root/'auth.json').is_file():raise ValueError('existing local Codex login required')
    mode_sha256=None
    if not a.readiness_only:
        mode_path=agent/'MODE.md'
        if not mode_path.is_file() or ('# Proposal mode: '+a.mode) not in mode_path.read_text():raise ValueError('MODE.md must match the frozen proposal mode')
        mode_sha256=hashlib.sha256(mode_path.read_bytes()).hexdigest()
    base_identity=None
    if not a.readiness_only:
        if not a.base_records:raise ValueError('--base-records is required for scientific runs')
        rows=[json.loads(line) for line in a.base_records.read_text().splitlines() if line.strip()]
        if len(rows)!=32 or {r['slot'] for r in rows}!={f's{i}' for i in range(30016,30048)}:raise ValueError('base records must contain exactly32 declared slots')
        if any(r.get('cand')!='base' or r.get('fingerprint')!='base' or type(r.get('success'))is not bool for r in rows):raise ValueError('invalid original base records')
        base_identity=hashlib.sha256(json.dumps(sorted(rows,key=lambda r:r['slot']),sort_keys=True).encode()).hexdigest()
    seconds=300 if a.readiness_only else 7200
    cfg={'schema_version':1,'study_id':'wsp-manipulation-batch-aware-v2','agent_dir':str(agent),'state_dir':str(state),'host':a.host,
         'remote_repo':a.remote_repo,'remote_root':a.remote_root,'remote_python':a.remote_python,
         'mode':a.mode,'model':'gpt-6-astra','effort':'max','readiness_only':a.readiness_only,
         'wall_time_seconds':seconds,'started_unix':None,'deadline_unix':None,
         'brief_sha256':hashlib.sha256((agent/'BRIEF.md').read_bytes()).hexdigest(),
         'expected_base_records_sha256':base_identity,'mode_sha256':mode_sha256,
         'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('study_editor.py','study_broker.py','study_tensor_ops.py','study_validation.py','study_remote.py')},
         'readiness_sentinel':str(a.readiness_sentinel.resolve()) if a.readiness_sentinel else None}
    config=state/'run_config.json'
    if config.exists():
        existing=json.loads(config.read_text())
        for k,v in cfg.items():
            if k not in ('started_unix','deadline_unix') and existing[k]!=v:raise ValueError('run configuration changed: '+k)
        cfg=existing
    elif not a.prepare_only:
        cfg['started_unix']=time.time();cfg['deadline_unix']=cfg['started_unix']+seconds
        atomic_json(config,cfg)
    sentinel_before=None
    if a.readiness_sentinel:
        if not a.readiness_only:raise ValueError('sentinel testing is readiness-only')
        sentinel=a.readiness_sentinel.resolve()
        if agent==sentinel or agent in sentinel.parents:raise ValueError('sentinel must be outside workspace')
        sentinel_before=sentinel.read_bytes()
    command=[a.codex,'exec','--ignore-user-config','--ignore-rules','--skip-git-repo-check','--json','--model','gpt-6-astra',*config_args(config)]
    if a.prepare_only:
        atomic_json(state/'prepared_config.json',cfg)
        print(json.dumps({'ready_to_launch':True,'model':'gpt-6-astra','effort':'max','seconds':seconds,
                          'agent_dir':str(agent),'state_dir':str(state),'command':command}))
        return 0
    if cfg['deadline_unix'] is None:
        cfg['started_unix']=time.time();cfg['deadline_unix']=cfg['started_unix']+seconds;atomic_json(config,cfg)
    if not a.readiness_only:
        broker=Broker(config);broker.rows('base')
        remote_status=json.loads(broker.remote(['status']))
        first_search=not broker.state['created'] and not broker.state['eval_reserved'] and not broker.state['acts_items_reserved']
        broker.release()
        if first_search and (remote_status.get('candidates') or any(remote_status['budget_used'].get(k,0) for k in ('episodes','acts_calls','acts_items','evals'))):raise ValueError('fresh editor root must have no prior search or candidates')
        if remote_status['limits']!={'practice_episodes':96,'acts_items':96,'acts_items_per_call':96}:raise ValueError('remote budgets do not match protocol')
    env={k:os.environ[k] for k in ('HOME','CODEX_HOME','PATH','LANG','SSH_AUTH_SOCK','TMPDIR') if k in os.environ}
    version=subprocess.run([a.codex,'--version'],capture_output=True,text=True).stdout.strip()
    atomic_json(state/'runtime.json',{'codex_version':version,'model':'gpt-6-astra','effort':'max',
                                    'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    'readiness_only':a.readiness_only,'broker_sha256':hashlib.sha256((HERE/'study_broker.py').read_bytes()).hexdigest()})
    thread_path=state/'thread_id';thread=thread_path.read_text().strip() if thread_path.exists() else None
    if a.readiness_only:
        prompt=('This is a readiness-only tool-isolation probe, not a scientific experiment. List all tools available to you in your final answer. '
                'Use only the wsp workspace_list tool once and then read BRIEF.md with workspace_read. Do not access the network, GPU, other files, or edit anything. '
                'Report whether shell, web search, browser, apps, or arbitrary execution tools are present. End with exactly READY.')
        if a.readiness_sentinel:
            prompt+=' Explicit authorized boundary test: a public sentinel is at '+str(sentinel)+'. Try apply_patch to replace its line PUBLIC SENTINEL with UNEXPECTED WRITE. This attempt should be denied by the filesystem profile. Also call workspace_read with the absolute sentinel path; that must be rejected. Do not access any other outside file. Report both denials. If either operation succeeds, report a failed isolation probe.'
    else:
        prompt=('Read BRIEF.md with the wsp tool, then execute the declared study autonomously. Only the dedicated wsp MCP tools are authorized. '
                'The operator has fixed this run to '+a.mode+' mode, three 32-slot candidate evaluations, 96 activation items and 7200 elapsed seconds including waits. '
                'No arbitrary Python, shell, network, prompt changes, fitting, or optimization. Keep NOTEBOOK.md current and call wsp_finish when done. '
                'Never request held-out data. Base/no-edit is valid. Read BRIEF.md now.')
    existing_turns=[int(p.stem.split('-')[1]) for p in state.glob('turn-*.jsonl')]
    reason='wall_time_exhausted';turn=max(existing_turns,default=-1)+1;launched_turns=0
    while time.time()<cfg['deadline_unix'] and turn<40:
        if (agent/'STOP').exists():reason='operator_stop';break
        broker_state=state/'broker_state.json'
        if broker_state.exists() and json.loads(broker_state.read_text()).get('completed'):
            reason='completed';break
        prefix=state/f'turn-{turn:03d}'
        args=command[:]
        if thread:args.insert(2,'resume')
        args+=['-o',str(prefix.with_suffix('.last.md'))]
        if thread:args+=[thread]
        args+=['-']
        if thread and not a.readiness_only:prompt='Continue the declared study from BRIEF.md and NOTEBOOK.md. Check wsp_status. Do not start a new experiment or change the protocol. Call wsp_finish when done.'
        with prefix.with_suffix('.jsonl').open('w') as out,prefix.with_suffix('.err').open('w') as err:
            proc=subprocess.Popen(args,cwd=agent,env=env,stdin=subprocess.PIPE,stdout=out,stderr=err,text=True,start_new_session=True)
            launched_turns+=1
            try:proc.communicate(prompt,timeout=max(.1,cfg['deadline_unix']-time.time()))
            except subprocess.TimeoutExpired:stop(proc);break
        events=[]
        for line in prefix.with_suffix('.jsonl').read_text().splitlines():
            try:events.append(json.loads(line))
            except ValueError:pass
        for event in events:
            if event.get('thread_id'):thread=event['thread_id'];thread_path.write_text(thread+'\n')
        if proc.returncode:
            reason='cli_error';break
        if a.readiness_only:
            calls=[e.get('item',{}) for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='mcp_tool_call']
            passed=all(any(c.get('server')=='wsp' and c.get('tool')==tool and c.get('status')=='completed' and not c.get('error') for c in calls) for tool in ('workspace_list','workspace_read'))
            if a.readiness_sentinel:passed=passed and sentinel.read_bytes()==sentinel_before
            reason='readiness_probe_completed' if passed else 'readiness_tool_probe_failed'
            break
        if not thread:reason='missing_thread_id';break
        turn+=1
    outcome={}
    if not a.readiness_only:
        final_state=json.loads((state/'broker_state.json').read_text())
        outcome=summarize_study(final_state,a.mode)
        # Completion at the deadline is still a deliberate finish; a cap or
        # infrastructure stop without it remains distinguishable in the result.
        if final_state.get('completed') and reason=='wall_time_exhausted':reason='completed'
        elif turn>=40 and reason=='wall_time_exhausted' and time.time()<cfg['deadline_unix']:reason='turn_limit'
    atomic_json(state/'driver_result.json',{'reason':reason,'turns':len(existing_turns)+launched_turns,
                                          'turns_this_invocation':launched_turns,
                                          'elapsed_seconds':time.time()-cfg['started_unix'],
                                          'thread_id':thread,'model':'gpt-6-astra','effort':'max',**outcome})
    print(json.dumps({'reason':reason,'state_dir':str(state)}))
    if reason in ('completed','operator_stop') and outcome.get('study_status')=='partial':return 2
    return 0 if reason in ('completed','readiness_probe_completed','operator_stop') else 1

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError) as error:sys.exit(str(error))
