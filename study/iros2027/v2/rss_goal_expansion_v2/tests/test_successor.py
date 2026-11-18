import ast
import json
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
OLD = HERE.with_name('rss_goal_expansion')


def read(path):
    return json.loads(path.read_text())


def test_exact_single_geometry_change_and_fresh_noise():
    old, new = read(OLD/'protocol.json'), read(HERE/'protocol.json')
    assert new['case_order'] == old['case_order']
    assert new['scoring'] == old['scoring']
    assert new['adaptation'] == old['adaptation']
    for field in ['first_pass_episodes','episodes_per_wording','layout_count','noise_replicates',
                  'gate','repeat_max_episodes','competitor_max_episodes','max_policy_episodes_before_adaptation']:
        assert new['qualification'][field] == old['qualification'][field]
    for case in new['case_order']:
        for field in ('original','teacher','unseen','competing','task','competing_task','horizon_seconds'):
            assert new['cases'][case][field] == old['cases'][case][field]
        for stage in ('initial','repeat','competing'):
            assert new['qualification']['slot_start'][stage][case] == old['qualification']['slot_start'][stage][case] + 10000
        assert new['heldout']['slot_start'][case] == old['heldout']['slot_start'][case] + 10000
        for split in ('development','heldout'):
            before = read(OLD/f'runtime/layouts/{case}_{split}.json')
            after = read(HERE/f'runtime/layouts/{case}_{split}.json')
            for a,b in zip(before['layouts'],after['layouts'],strict=True):
                assert a['id']==b['id']
                for name,pose in a['poses'].items():
                    expected = np.array(pose['pos'])
                    if case=='glasses_pickup' and name=='rubiks_cube':
                        expected[1]-=.025
                    np.testing.assert_allclose(b['poses'][name]['pos'],expected,rtol=0,atol=1e-9)
                    assert pose['quat']==b['poses'][name]['quat']
    for file in ('runtime/scoring.py','runtime/tasks.py','runtime/layouts.py','rewrites.py'):
        assert (HERE/file).read_bytes()==(OLD/file).read_bytes()


def test_no_policy_calls_and_no_environment_action_steps():
    source=(HERE/'runtime/reset_preflight.py').read_text()
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('infer_batch','start_server','run_evaluation','run_episode','step') for n in calls)
    assert 'NoPolicyImports()' in source and 'env.step = forbid_action' in source
    assert 'for reset in (1, 2):' in source


def test_one_attempt_copy_does_not_modify_shared_pool():
    repo=HERE.parents[3]
    original=(repo/'weightsmith/pool.py').read_text()
    single=(HERE/'runtime/single_attempt_pool.py').read_text()
    assert single==original.replace('while todo and attempt < 4:','while todo and attempt < 1:')


def test_missing_or_failed_preflight_cannot_qualify(tmp_path):
    import preflight
    with pytest.raises(FileNotFoundError):
        preflight.verify(tmp_path)
    p=tmp_path/'preflight';p.mkdir()
    (p/'complete.json').write_text(json.dumps({'status':'stopped'}))
    with pytest.raises(ValueError):
        preflight.verify(tmp_path)


@pytest.mark.parametrize('failure', ['nonzero', 'timeout', 'missing_receipt'])
def test_preflight_stops_and_accounts_after_first_bad_start(tmp_path, monkeypatch, failure):
    import contextlib
    import subprocess
    import sys
    import preflight
    import weightsmith.locks
    calls=[]
    def run(command, **kw):
        calls.append(command)
        if failure=='timeout':
            raise subprocess.TimeoutExpired(command,900)
        return subprocess.CompletedProcess(command,1 if failure=='nonzero' else 0)
    monkeypatch.setattr(preflight.subprocess,'run',run)
    monkeypatch.setattr(weightsmith.locks,'GpuLock',lambda gpu:contextlib.nullcontext())
    monkeypatch.setattr(sys,'argv',['preflight','--root',str(tmp_path),'--gpu','1'])
    with pytest.raises((RuntimeError,subprocess.TimeoutExpired,FileNotFoundError)):
        preflight.main()
    state=read(tmp_path/'preflight/status.json')
    assert len(calls)==state['simulator_starts']==len(state['starts'])==1
    assert state['status']=='stopped' and state['starts'][0]['execution_seconds']>=0
    assert state['starts'][0]['evidence_sha256']
    assert not (tmp_path/'preflight/heldout').exists()
    assert not (tmp_path/'preflight/complete.json').exists()
