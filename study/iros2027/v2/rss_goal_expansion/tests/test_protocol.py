import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import qualify
spec=importlib.util.spec_from_file_location('expansion_layout_generator',HERE/'runtime/generate_layouts.py')
layouts=importlib.util.module_from_spec(spec);spec.loader.exec_module(layouts)
P=json.loads((HERE/'protocol.json').read_text())


def test_gate_boundaries():
    assert qualify.gate(4,8)
    assert qualify.gate(12,16)
    assert not qualify.gate(3,7)
    assert not qualify.gate(5,8)
    assert not qualify.gate(13,16)


def test_fixed_budget_pairing_new_slots_and_no_heldout_in_qualification():
    total=0
    for case in P['case_order']:
        first=qualify.build_plan(P,case,'initial')
        repeat=qualify.build_plan(P,case,'repeat')
        assert [b['cell'] for b in repeat['batches']]==['teacher','original']
        for plan in (first,repeat):
            total+=sum(b['n'] for b in plan['batches'])
            assert plan['split']=='development' and plan['inference_batching']=='fixed16'
            assert len({b['slot_start'] for b in plan['batches']})==1
            assert all(b['n']==16 and len(b['layout_ids'])==4 for b in plan['batches'])
            assert min(b['slot_start'] for b in plan['batches'])>=150000
            assert all(P['cases'][case]['unseen']!=b['effective_prompt'] for b in plan['batches'])
        assert set(range(first['batches'][0]['slot_start'],first['batches'][0]['slot_start']+16)).isdisjoint(range(repeat['batches'][0]['slot_start'],repeat['batches'][0]['slot_start']+16))
    assert total==192 and total+2*16==P['qualification']['max_policy_episodes_before_adaptation']


def test_layout_reproducibility_disjointness_and_scale():
    from weightsmith import config
    for case in P['case_order']:
        for split in layouts.SEEDS:
            stored=json.loads((HERE/'runtime/layouts'/f'{case}_{split}.json').read_text())
            assert stored==layouts.build(case,split,config.ROBOLAB/'assets/scenes')
            assert len(stored['layouts'])==(4 if split=='development' else 8)
            base=layouts.stock_poses(stored['asset'],stored['objects'])
            for layout in stored['layouts']:
                for name,pose in layout['poses'].items():
                    delta=np.array(pose['pos'])-np.array(base[name]['pos'])
                    assert np.max(abs(delta[:2]))<=.010001 and abs(delta[2])<1e-8
                    assert np.linalg.norm(pose['quat'])>.9999
        dev=json.loads((HERE/'runtime/layouts'/f'{case}_development.json').read_text())
        held=json.loads((HERE/'runtime/layouts'/f'{case}_heldout.json').read_text())
        assert {x['id'] for x in dev['layouts']}.isdisjoint({x['id'] for x in held['layouts']})


def test_each_goal_has_single_class_entrypoint_and_competitor():
    for case,cfg in P['cases'].items():
        assert cfg['original']!=cfg['teacher'] and cfg['competing']!=cfg['teacher']
        for key in ('task','competing_task'):
            source=(HERE/'runtime'/f"{cfg[key]}.py").read_text()
            assert f"class {cfg[key]}(_tasks.{cfg[key]}):" in source
