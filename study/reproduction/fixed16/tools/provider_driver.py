#!/usr/bin/env python3
"""Prepare/run one explicit-model Claude CLI or custom stdio-MCP client.

Default is preparation only. --run authorizes the selected local CLI to contact
its provider. A separate validated readiness attestation is required. No keys
are copied, printed, or stored. Existing run roots are never reset.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'bundle/weightsmith/extensions/batch_capture_v2/runtime/drivers'))
from study_broker import summarize_study
from broker_fixed16 import ReadinessBroker
from enrollment import check_enrollment


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, data):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2) + '\n'); tmp.replace(path)


def exact_model(model):
    if not isinstance(model, str) or not model.strip() or model != model.strip() or model.lower() in {'opus', 'sonnet', 'haiku', 'default', 'best', 'latest'} or '<' in model:
        raise ValueError('Use the exact resolved provider model ID, never a moving alias')
    return model


def claude_command(config, mcp, session=None):
    settings = json.dumps({'disableAllHooks': True, 'fallbackModel': []})
    command = [config.get('cli', 'claude'), '-p', '--bare', '--restricted', '--model', exact_model(config['model']),
               '--effort', config['effort'], '--tools', '', '--allowedTools', 'mcp__wsp__*', '--strict-mcp-config',
               '--mcp-config', str(mcp), '--setting-sources', '', '--settings', settings, '--disable-slash-commands',
               '--permission-mode', 'dontAsk', '--output-format', 'stream-json', '--verbose']
    if session: command += ['--resume', session]
    return command


def event_metadata(text, expected):
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    ids = {e['session_id'] for e in events if e.get('session_id')}
    models = set(); initialized = False
    for event in events:
        if event.get('model'): models.add(event['model'])
        if isinstance(event.get('message'), dict) and event['message'].get('model'): models.add(event['message']['model'])
        for name in event.get('modelUsage', {}): models.add(name)
        if event.get('type') == 'system' and event.get('subtype') == 'init':
            initialized = True
            tools = event.get('tools', [])
            if not tools or any(not t.startswith('mcp__wsp__') and t != 'EndConversation' for t in tools):
                raise ValueError('Provider exposed tools outside the study MCP surface')
    if not initialized or len(ids) != 1 or models != {expected}: raise ValueError('Missing tool/session/model attribution or model fallback occurred')
    return next(iter(ids))


def probe_tools_completed(text):
    requested, completed = {}, set()
    for event in [json.loads(line) for line in text.splitlines() if line.strip()]:
        content = event.get('message', {}).get('content', [])
        if not isinstance(content, list): continue
        for block in content:
            if not isinstance(block, dict): continue
            if block.get('type') == 'tool_use': requested[block.get('id')] = block.get('name')
            if block.get('type') == 'tool_result' and not block.get('is_error', False):
                completed.add(block.get('tool_use_id'))
    return {'mcp__wsp__workspace_list', 'mcp__wsp__workspace_read'} <= {requested.get(i) for i in completed}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--probe', action='store_true', help='Run a separate 300-second workspace-only provider readiness probe')
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); exact_model(cfg['model'])
    if cfg['provider'] not in {'claude-cli', 'custom-cli'} or not cfg.get('effort'):
        raise ValueError('Provider and exact effort setting are required')
    enrollment = check_enrollment(cfg, require_runtime=not a.probe, fresh_workspace=not a.resume)
    agent, state = Path(cfg['agent_dir']).resolve(), Path(cfg['state_dir']).resolve()
    if a.probe:
        if a.run or a.resume: raise ValueError('A readiness probe is a separate fresh invocation')
        state = state.with_name(state.name + '-readiness')
    if agent == state or agent in state.parents or not (agent / 'BRIEF.md').is_file(): raise ValueError('Separate prepared editor/operator directories required')
    if cfg['mode'] not in {'adaptive', 'nonadaptive'}: raise ValueError('Unknown proposal mode')
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (state / 'driver.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mcp = state / 'mcp.json'; run_config = state / 'run_config.json'
    save(mcp, {'mcpServers': {'wsp': {'command': sys.executable, 'args': [str(HERE / 'broker_fixed16.py'), '--config', str(run_config)]}}})
    seconds = 300 if a.probe else 7200
    prepared = {**cfg, 'agent_dir': str(agent), 'state_dir': str(state), 'wall_time_seconds': seconds, 'readiness_only': a.probe,
                'brief_sha256': digest(agent / 'BRIEF.md'), 'mode_sha256': digest(agent / 'MODE.md'),
                'adapter_sha256': digest(__file__), 'broker_transport_sha256': digest(HERE / 'broker_fixed16.py'),
                'fixed16_enrollment': enrollment}
    if not a.probe:
        rows = [json.loads(line) for line in Path(cfg['base_records']).read_text().splitlines() if line.strip()]
        if len(rows) != 32 or {r['slot'] for r in rows} != {f's{i}' for i in range(30016, 30048)}:
            raise ValueError('Exactly the pinned 32 base records are required')
        prepared['expected_base_records_sha256'] = hashlib.sha256(json.dumps(sorted(rows, key=lambda r: r['slot']), sort_keys=True).encode()).hexdigest()
    command = claude_command(prepared, mcp) if cfg['provider'] == 'claude-cli' else cfg['command_argv']
    if not a.run and not a.probe:
        save(state / 'prepared_config.json', prepared)
        print(json.dumps({'preparation_only': True, 'command': command, 'mcp_config': str(mcp), 'no_model_call': True}, indent=2)); return
    if not a.probe:
        attestation = json.loads(Path(cfg['readiness_attestation']).read_text())
        if any(attestation.get(k) != v for k, v in {'passed': True, 'model': cfg['model'], 'effort': cfg['effort'], 'tools_only_wsp': True,
                                                  'adapter_sha256': digest(__file__)}.items()):
            raise ValueError('A matching actual provider readiness/isolation probe is required')
        version_command = cfg.get('version_argv', [cfg.get('cli', 'claude'), '--version'])
        version = subprocess.run(version_command, text=True, capture_output=True, check=True).stdout.strip()
        if version != attestation.get('cli_version'): raise ValueError('Provider CLI changed after readiness validation')
        prepared['backend_version'] = version
    if cfg['provider'] == 'claude-cli' and not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError('The bare CLI adapter requires the operator’s own existing ANTHROPIC_API_KEY environment variable')
    if run_config.exists():
        if not a.resume: raise ValueError('Existing run requires explicit --resume; never reset its clock or counters')
        existing = json.loads(run_config.read_text())
        if any(existing.get(k) != v for k, v in prepared.items()): raise ValueError('Prepared inputs changed on resume')
        prepared = existing
    else:
        if a.resume: raise ValueError('Cannot resume a nonexistent run')
        prepared.update(started_unix=time.time(), deadline_unix=time.time() + seconds); save(run_config, prepared)
    broker = ReadinessBroker(run_config)
    try:
        if not a.probe:
            broker.rows('base')
            remote = json.loads(broker.remote(['status']))
            if not a.resume and (remote.get('candidates') or any(remote['budget_used'].get(k, 0) for k in ('episodes', 'acts_items', 'evals'))):
                raise ValueError('A fresh discovery project must have no previous search or charges')
    finally: broker.release()
    session_file = state / 'session_id'; session = session_file.read_text().strip() if session_file.exists() else None
    start_turn = len(list(state.glob('turn-*.jsonl'))); reason = 'wall_time_exhausted'
    for turn in range(start_turn, 40):
        remaining = prepared['deadline_unix'] - time.time()
        if remaining <= 0: break
        broker_state = json.loads((state / 'broker_state.json').read_text())
        if broker_state.get('completed'): reason = 'completed'; break
        if (agent / 'STOP').exists(): reason = 'operator_stop'; break
        cmd = claude_command(prepared, mcp, session) if cfg['provider'] == 'claude-cli' else [v.format(model=cfg['model'], effort=cfg['effort'], mcp_config=str(mcp), session_id=session or '') for v in cfg['command_argv']]
        prompt = 'Read BRIEF.md and MODE.md using only wsp tools. Continue this same declared study. Preserve all counters and use only development evidence. Call wsp_finish when done.'
        if a.probe: prompt = 'Readiness probe only. Call workspace_list and workspace_read for BRIEF.md. List available tools. Do not invoke any remote/GPU tool or edit a file. Finish with READY.'
        with (state / f'turn-{turn:03d}.jsonl').open('x') as out, (state / f'turn-{turn:03d}.err').open('x') as err:
            process = subprocess.Popen(cmd, cwd=agent, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True, start_new_session=True)
            try: process.communicate(prompt, timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
                break
        if process.returncode: reason = 'provider_error'; break
        session = event_metadata((state / f'turn-{turn:03d}.jsonl').read_text(), cfg['model']); session_file.write_text(session + '\n')
        if a.probe:
            if not probe_tools_completed((state / f'turn-{turn:03d}.jsonl').read_text()):
                raise ValueError('Readiness did not exercise both required MCP tools')
            version = subprocess.run(cfg.get('version_argv', [cfg.get('cli', 'claude'), '--version']), text=True, capture_output=True, check=True).stdout.strip()
            save(state / 'readiness_attestation.json', {'passed': True, 'model': cfg['model'], 'effort': cfg['effort'], 'tools_only_wsp': True,
                'cli_version': version, 'session_id': session, 'probe_log_sha256': digest(state / f'turn-{turn:03d}.jsonl'),
                'adapter_sha256': digest(__file__), 'note': 'Validated provider tool inventory and two local workspace MCP calls; no GPU operation was attempted.'})
            print(str(state / 'readiness_attestation.json')); return
    final = json.loads((state / 'broker_state.json').read_text())
    if final.get('completed'): reason = 'completed'
    result = {'reason': reason, 'model': cfg['model'], 'effort': cfg['effort'], 'provider': cfg['provider'], 'session_id': session,
              'backend_version': prepared['backend_version'], 'elapsed_seconds': time.time() - prepared['started_unix'], **summarize_study(final, cfg['mode'])}
    save(state / 'driver_result.json', result); print(json.dumps(result, indent=2))
    if reason != 'completed' or result['study_status'] != 'complete': raise SystemExit(2)


if __name__ == '__main__': main()
