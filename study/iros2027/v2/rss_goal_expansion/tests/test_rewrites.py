import importlib.util
import json
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('expansion_rewrites',HERE/'rewrites.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
PROTOCOL=json.loads((HERE/'protocol.json').read_text())


@pytest.mark.parametrize('case',list(PROTOCOL['cases']))
def test_declared_original_unseen_and_competing(case):
    cfg=PROTOCOL['cases'][case]
    for cell in ('original','unseen'):
        assert module.route(cfg[cell])['effective_prompt']==cfg['teacher']
    assert module.route(cfg['competing'])['effective_prompt']==cfg['competing']


@pytest.mark.parametrize('prompt',[
    "Don't pick up the glasses",'Never put the mustard in the left bin',
    'Put the mustard in the right bin','Put the mustard in the left or right bin',
    'Place the bowl in front of the cube','Place the cube in front of the bowl',
    'Pick up the cube and the glasses','Place the glasses on the table',
    'Put the cube behind the bowl, then place the bowl behind the cube',
    'Put the cube in the bowl','Pick up the rubiks cube',
])
def test_competing_negated_or_ambiguous_pass_through(prompt):
    assert module.route(prompt)['effective_prompt']==prompt


def test_rules_recognize_nondeclared_lexical_variants_without_case_id():
    assert module.intent('Move a cube behind a bowl')=='cube_behind'
    assert module.intent('Raise the spectacles')=='glasses_pickup'
    assert module.intent('Place mustard inside the left bin')=='mustard_left_bin'
