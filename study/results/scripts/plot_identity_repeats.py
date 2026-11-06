"""Verified binary slot outcomes across chronological unchanged-weight repeats."""
from pathlib import Path
import json

PAPER=Path(__file__).resolve().parents[1]
RUNS=('adaptive-r1','nonadaptive-r1','adaptive-r2','nonadaptive-r2')
SLOTS=list(range(20032,20064))


def identity_matrix(data):
    def outcomes(rows):
        if [r['slot'] for r in rows]!=SLOTS or any(type(r['success']) is not bool for r in rows):
            raise ValueError('Heatmap requires exactly32 complete ordered boolean slot outcomes')
        return [r['success'] for r in rows]
    base=outcomes(data['base_development_slot_outcomes'])
    if sum(base)!=data['base_development']['success']:raise ValueError('Heatmap base count differs')
    columns=[{'label':'Base','run':'operator-base','proposal':None,'successes':sum(base),'outcomes':base}]
    for run in RUNS:
        complete=[p for p in data['discovery'].get(run,{}).get('proposals',[]) if p['status']=='complete']
        for p in sorted(complete,key=lambda p:p['evaluation_ordinal']):
            if p['changed_elements']!=0:continue
            values=outcomes(p['slot_outcomes']);a=p['analysis']
            fixed=[s for s,b,c in zip(SLOTS,base,values) if not b and c]
            broken=[s for s,b,c in zip(SLOTS,base,values) if b and not c]
            if (sum(values)!=a['candidate_success'] or fixed!=a['fixed_slots'] or broken!=a['broken_slots']
                    or p['changed_rows']!=0 or p['l2_delta']!=0):raise ValueError('Heatmap repeat differs from verified zero-edit analysis')
            short=('A' if run.startswith('adaptive') else 'N')+run[-1]
            columns.append({'label':short+' '+p['id'],'run':run,'proposal':p['id'],
                            'workspace_fingerprint':p['fingerprint'],'successes':sum(values),'outcomes':values})
    return {'slots':SLOTS,'columns':columns,'interpretation':'Nominal sampling streams align; inputs need not be pixel-identical. Repeats are descriptive, not pooled independent trials.'}


def render(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy as np
    matrix=identity_matrix(data);columns=matrix['columns']
    values=np.asarray([c['outcomes'] for c in columns],dtype=int).T
    fig,ax=plt.subplots(figsize=(max(5.4,.63*len(columns)+1),5.7),layout='constrained')
    cmap=ListedColormap(['#E8EAEC','#24465A'])
    ax.imshow(values,cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    ax.set(xticks=range(len(columns)),xticklabels=[c['label']+f"\n{c['successes']}/32" for c in columns],
           yticks=range(32),yticklabels=[str(s) for s in SLOTS],ylabel='Nominal sampling slot')
    ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,labelsize=8,pad=5);ax.tick_params(axis='y',length=0,labelsize=7)
    ax.set_xticks(np.arange(-.5,len(columns),1),minor=True);ax.set_yticks(np.arange(-.5,32,1),minor=True)
    ax.grid(which='major',visible=False);ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='minor',length=0)
    ax.axhline(15.5,color='#171717',lw=1.7)
    ax.legend(handles=[Patch(facecolor=cmap(0),label='Failure'),Patch(facecolor=cmap(1),label='Success')],
              loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False,fontsize=8)
    for spine in ax.spines.values():spine.set_visible(False)
    for folder,formats in ((PAPER/'figures',('pdf','svg','png')),(PAPER.parent/'blog/figures',('svg','png'))):
        folder.mkdir(parents=True,exist_ok=True)
        for ext in formats:fig.savefig(folder/f'identity_repeat_heatmap.{ext}',dpi=220,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
    path=PAPER/'data/identity_repeat_matrix.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(matrix,indent=2)+'\n')
    return len(columns)-1


if __name__=='__main__':
    print('Verified identity repeats plotted:',render(json.loads((PAPER/'data/experiment_results.json').read_text())))
