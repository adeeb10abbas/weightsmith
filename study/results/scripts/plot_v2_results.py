#!/usr/bin/env python3
"""Separate v2 curves; never combine its changed runtime/base with v1."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAPER=Path(__file__).resolve().parents[1]
BLUE='#24465A';ORANGE='#A64B2A';GREY='#72787D'


def render(data):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'axes.axisbelow':True})
    def save(fig,name):
        for folder,formats in ((PAPER/'figures',('pdf','svg','png')),(PAPER.parent/'blog/figures',('svg','png'))):
            folder.mkdir(parents=True,exist_ok=True)
            for ext in formats:fig.savefig(folder/f'{name}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.06)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,2.65),layout='constrained')
    base=data.get('base_development');run=data.get('discovery',{}).get('adaptive-r1',{})
    complete=[p for p in run.get('proposals',[]) if p['status']=='complete']
    if base is None:
        ax.axis('off');ax.text(.5,.64,'Separate fixed16 v2: no completed baseline yet',ha='center',transform=ax.transAxes,weight='bold')
        ax.text(.5,.43,'New runtime, new development streams; no pooled v1/v2 estimate.',ha='center',transform=ax.transAxes)
    else:
        rate=base['success']/base['n'];ci=base['wilson_95']
        ax.errorbar([0],[rate],yerr=[[rate-ci[0]],[ci[1]-rate]],fmt='s',color=BLUE,capsize=3,label='Fresh v2 base / incumbent')
        byname={p['id']:p for p in complete}
        if complete:
            xs=[p['round'] for p in complete];ys=[p['analysis']['candidate_rate_paired'] for p in complete]
            errors=np.array([[y-p['success_wilson_95'][0],p['success_wilson_95'][1]-y] for y,p in zip(ys,complete)]).T
            ax.errorbar(xs,ys,yerr=errors,fmt='o--',lw=1.1,color=ORANGE,capsize=3,label='Proposed policy')
            accepted=[rate]+[rate if p['incumbent_after_round']=='base' else byname[p['incumbent_after_round']]['analysis']['candidate_rate_paired'] for p in complete]
            ax.plot([0]+xs,accepted,'s-',color=BLUE,lw=1.6)
            for x,y,p in zip(xs,ys,complete):
                tag='identity' if p['changed_elements']==0 else f"{p['changed_rows']} rows"
                ax.annotate(f"{p['analysis']['candidate_success']}/32 ({tag})",(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8,color=ORANGE)
        for ordinal in (1,2,3):
            if not any(p['round']==ordinal for p in complete):ax.text(ordinal,.08,'Not completed' if run.get('status')=='stopped_partial' else 'Pending',ha='center',fontsize=8,color=GREY)
        ax.set(title='Separate fixed16 v2 exploratory discovery',xlabel='Proposal / round',ylabel='Development success fraction',
               xticks=[0,1,2,3],xticklabels=['Fresh base','1','2','3'],xlim=(-.25,3.3),ylim=(0,1))
        ax.legend(frameon=False,fontsize=8,loc='upper left')
    save(fig,'v2_discovery_curves')
    final=data.get('final_scoring',{});comparisons=final.get('comparisons',{})
    fig,ax=plt.subplots(figsize=(7,max(2.3,.48*len(comparisons)+1)),layout='constrained')
    if not comparisons:
        ax.axis('off');ax.text(.5,.64,'Separate fixed16 v2: held-out results pending',ha='center',transform=ax.transAxes,weight='bold')
        ax.text(.5,.43,'Each unique deployment has one estimate; repeated role labels share it.',ha='center',transform=ax.transAxes)
    else:
        labels=[]
        for index,(name,a) in enumerate(comparisons.items()):
            v=a['success_difference'];lo,hi=a['paired_bootstrap_95']
            ax.errorbar([v],[index],xerr=[[v-lo],[hi-v]],fmt='s',color=BLUE,capsize=3)
            labels.append(name+f" ({len(a['shared_roles'])} round labels)")
            ax.annotate(f"{a['candidate_success']}/64 vs {a['base_success']}/64; +{a['fixed']} / -{a['broken']}",(hi,index),xytext=(6,0),textcoords='offset points',va='center',fontsize=7)
        ax.axvline(0,color=GREY,lw=.8);ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='V2 candidate - fresh v2 base success fraction')
        ax.invert_yaxis();ax.margins(x=.25)
    save(fig,'v2_heldout_unique_policies')
    return len(complete),len(comparisons)


if __name__=='__main__':
    data=json.loads((PAPER/'data/experiment_results_v2.json').read_text())
    n,m=render(data);print(f'Plotted separate v2: {n} completed proposals, {m} verified final comparisons.')
