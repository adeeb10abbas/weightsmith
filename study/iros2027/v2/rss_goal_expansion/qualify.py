"""Bounded development-only qualification with immutable plans and all failures kept."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from worker import digest, save

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def gate(original,teacher):
    return teacher >= 8 and teacher-original >= 4 and original <= 12


def build_plan(protocol,case,stage):
    cfg = protocol['cases'][case]
    layout = HERE/'runtime/layouts'/f'{case}_development.json'
    ids = [x['id'] for x in json.loads(layout.read_text())['layouts']]
    cells = ['original','teacher'] if stage=='initial' else (['teacher','original'] if stage=='repeat' else ['competing'])
    return {'cohort_id':f"{protocol['cohort']}-{case}-{stage}-v1",'case':case,'split':'development',
        'stage':stage,'inference_batching':'fixed16','batches':[
            {'cell':cell,'condition':'base','task':cfg['competing_task'] if cell=='competing' else cfg['task'],
             'workspace':None,'layouts':str(layout.resolve()),'layout_ids':ids,'n':16,
             'slot_start':protocol['qualification']['slot_start'][stage][case],
             'input_prompt':cfg[cell],'effective_prompt':cfg[cell]} for cell in cells]}


def read_outcomes(owner):
    owner = Path(owner)
    plan = json.loads((owner/'plan.json').read_text())
    state = json.loads((owner/'status.json').read_text())
    if state['status']!='complete' or state['completed_batches']!=len(plan['batches']):
        raise RuntimeError(f'Incomplete qualification owner {owner}')
    for path,sha in json.loads((owner/'identity.json').read_text()).items():
        if digest(path)!=sha:
            raise RuntimeError(f'Frozen source changed {path}')
    rows = {}
    for index,spec in enumerate(plan['batches']):
        batch = owner/f"batch-{index:02d}-{spec['cell']}-base"
        receipt = json.loads((batch/'completed.json').read_text())
        episodes = json.loads((batch/'episodes.json').read_text())
        if receipt['n']!=16 or len(episodes)!=16 or receipt['episodes_sha256']!=digest(batch/'episodes.json'):
            raise RuntimeError(f'Incomplete or changed receipt {batch}')
        if any(r['case']!=plan['case'] or r['cell']!=spec['cell'] or r['condition']!='base' or r['inference_batching']!='fixed16' for r in episodes):
            raise RuntimeError('Wrong evidence identity')
        successes = sum(bool(r['success']) for r in episodes)
        if successes != receipt['successes']:
            raise RuntimeError('Wrong receipt success count')
        rows[spec['cell']] = {'successes':successes,'n':16,'batch':str(batch.resolve()),
                             'receipt_sha256':digest(batch/'completed.json')}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'qualification.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = json.loads((HERE/'protocol.json').read_text())
    plans = root/'qualification-plans'
    plans.mkdir(exist_ok=True)
    for case in protocol['case_order']:
        for stage in ['initial','repeat','competing']:
            path = plans/f'{case}-{stage}.json'
            plan = build_plan(protocol,case,stage)
            if path.exists() and json.loads(path.read_text())!=plan:
                raise RuntimeError('Qualification plan changed before restart')
            save(path,plan)
    sources = [HERE/'protocol.json',HERE/'worker.py',HERE/'rewrites.py',Path(__file__),
               *sorted((HERE/'runtime').glob('*.py')),*sorted((HERE/'runtime/layouts').glob('*.json')),
               *sorted(plans.glob('*.json')),*[Path(x) for x in protocol['source_paths']]]
    seal = {str(p.resolve()):digest(p) for p in sources}
    identity_path = root/'qualification-source-freeze.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=seal:
        raise RuntimeError('Qualification source freeze changed; do not overwrite this cohort')
    save(identity_path,seal)
    if args.prepare_only:
        print(json.dumps({'status':'prepared','source_count':len(seal),'plans':len(list(plans.glob('*.json')))}))
        return
    state = {'status':'running','started_at':time.time(),'protocol_sha256':digest(HERE/'protocol.json'),
             'initial':{},'repeat':{},'selected':[],'competing':{},'max_episodes':224}
    status_path = root/'qualification-status.json'
    def run(case,stage):
        if any(digest(p)!=sha for p,sha in seal.items()):
            raise RuntimeError('Qualification sources changed after freeze')
        owner = root/'qualification'/case/stage
        state.update(active_case=case,active_stage=stage)
        save(status_path,state)
        subprocess.run([sys.executable,str(HERE/'worker.py'),'--plan',str(plans/f'{case}-{stage}.json'),
                        '--root',str(owner),'--gpu',str(args.gpu)],cwd=REPO,check=True)
        return read_outcomes(owner)
    try:
        for case in protocol['case_order']:
            row = run(case,'initial')
            row['passes_gate'] = gate(row['original']['successes'],row['teacher']['successes'])
            state['initial'][case] = row
            save(status_path,state)
        for case in protocol['case_order']:
            if not state['initial'][case]['passes_gate']:
                continue
            row = run(case,'repeat')
            row['passes_gate'] = gate(row['original']['successes'],row['teacher']['successes'])
            state['repeat'][case] = row
            save(status_path,state)
        selected = [case for case in protocol['case_order'] if state['repeat'].get(case,{}).get('passes_gate')][:2]
        state['selected'] = selected
        for case in selected:
            row = run(case,'competing')
            row['competence_qualified'] = row['competing']['successes']>=8
            state['competing'][case] = row
            rounds = [{cell:state[stage][case][cell]['batch'] for cell in ('original','teacher')} for stage in ('initial','repeat')]
            value = {'case':case,'split':'development','protocol_sha256':state['protocol_sha256'],
                     'rounds':rounds,'teacher_batch':rounds[0]['teacher'],'preservation_batch':row['competing']['batch'],
                     'preservation_competence_qualified':row['competence_qualified'],'qualification_source_freeze_sha256':digest(identity_path)}
            save(root/'qualification'/case/'qualified.json',value)
            save(status_path,state)
        state['status']='complete'
    except BaseException as exc:
        state.update(status='stopped',reason=repr(exc))
        raise
    finally:
        state['updated_at']=time.time()
        save(status_path,state)


if __name__=='__main__':
    main()
