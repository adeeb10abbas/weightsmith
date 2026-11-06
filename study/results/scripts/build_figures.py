#!/usr/bin/env python3
"""Regenerate paper/blog figures from the included measured data. No simulation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
BLOG=ROOT.parent/'blog/figures';BLOG.mkdir(parents=True,exist_ok=True)
r=json.loads((ROOT/'data/smoke_report.json').read_text())
a=json.loads((ROOT/'data/analysis.json').read_text())
navy='#173348';teal='#087F8C';coral='#C15E43';pale='#EAF2F4';muted='#506878'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'ps.fonttype':42,
 'text.color':navy,'axes.labelcolor':navy,'xtick.color':muted,'ytick.color':muted,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#CBD8DF',
 'savefig.facecolor':'white','figure.facecolor':'white'})
def save(fig,name):
 for suffix in ['pdf','png','svg']:fig.savefig(FIG/f'{name}.{suffix}',dpi=250,bbox_inches='tight',pad_inches=.04)
 for suffix in ['png','svg']:fig.savefig(BLOG/f'{name}.{suffix}',dpi=250,bbox_inches='tight',pad_inches=.04)
 plt.close(fig)

fig,ax=plt.subplots(figsize=(3.5,1.85));ax.set_xlim(0,10);ax.set_ylim(0,5.5);ax.axis('off')
def box(x,y,w,h,title,body,color=pale):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',fc=color,ec='none'))
 ax.text(x+w/2,y+h-.3,title,ha='center',va='center',weight='bold',fontsize=8.2)
 ax.text(x+w/2,y+.62,body,ha='center',va='center',fontsize=6.6,linespacing=1.3)
def arrow(p,q,col=teal,connection='arc3'):
 ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,lw=1.0,color=col,connectionstyle=connection))
box(.15,2.3,2.8,1.65,'Inspect failures','Episodes\nActivations')
box(3.6,2.3,2.8,1.65,'Propose patch','Fixed editor\nSparse weight edit')
box(7.05,2.3,2.8,1.65,'Test patch','Rollout test\nKeep or revert')
arrow((2.98,3.1),(3.5,3.1));arrow((6.45,3.1),(6.95,3.1))
arrow((8.45,2.18),(1.55,2.18),connection='arc3,rad=-.23')
ax.text(5,1.15,'Adaptive arm: feedback informs the next patch',ha='center',fontsize=6.8,color=teal)
ax.plot([0,10],[.83,.83],ls='--',lw=1,color=muted)
ax.text(5,.35,'Freeze candidates  |  Operator-only held-out test',ha='center',fontsize=6.6,weight='bold')
ax.text(5,4.6,'PROPOSED PROTOCOL - NOT A MEASURED LEARNING CURVE',ha='center',fontsize=5.8,color=muted)
fig.tight_layout(pad=.1);save(fig,'edit_loop')

# Monochrome vector block diagram for the paper. Keep the colored blog export separate.
with plt.rc_context({'font.family':'DejaVu Serif', 'text.color':'black',
                     'mathtext.fontset':'dejavuserif', 'font.size':9}):
 fig,ax=plt.subplots(figsize=(7,1.48))
 ax.set_xlim(0,14);ax.set_ylim(0,3.3);ax.axis('off')
 def block(x,y,w,h,title,body):
  ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='black',linewidth=.8))
  ax.text(x+w/2,y+h*.68,title,ha='center',va='center',fontsize=9)
  ax.text(x+w/2,y+h*.28,body,ha='center',va='center',fontsize=8)
 def edge(points,style='-'):
  xs,ys=zip(*points)
  ax.plot(xs[:-1],ys[:-1],color='black',lw=.8,ls=style)
  ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                              mutation_scale=8,lw=.8,color='black',linestyle=style))
 block(.1,2.18,3.1,.88,'Development evidence',r'Episodes; activations')
 block(3.65,2.18,3.1,.88,r'Fixed editor $E_\phi$',r'Sparse weight proposal $\Delta_j$')
 block(7.2,2.18,3.1,.88,'Candidate policy',r'$Q(\theta_j+M_j\odot\Delta_j)$')
 block(10.75,2.18,3.1,.88,'Development rollouts','Accept / revert')
 for x in [3.2,6.75,10.3]:edge([(x,2.62),(x+.45,2.62)])
 edge([(12.3,2.18),(12.3,1.62),(1.65,1.62),(1.65,2.18)])
 ax.text(7,1.78,r'Outcome feedback; accepted policy becomes $\theta_{j+1}$',ha='center',fontsize=8)
 block(8.2,.13,5.65,.88,'Operator-only held-out evaluation','Frozen policies; no editor feedback')
 edge([(13.85,2.62),(13.99,2.62),(13.99,.57),(13.85,.57)],style='--')
 ax.text(.12,.66,'Proposed adaptive protocol',ha='left',fontsize=8.5)
 ax.text(.12,.24,'Editor parameters remain fixed.',ha='left',fontsize=8)
 fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
 for suffix in ['pdf','png','svg']:
  fig.savefig(FIG/f'edit_loop_paper.{suffix}',dpi=250,bbox_inches='tight',pad_inches=.025)
 plt.close(fig)

fig,(ax,bx)=plt.subplots(1,2,figsize=(3.5,1.7),gridspec_kw={'width_ratios':[.85,1.5]},layout='constrained')
ax.bar(['Success','Horizon'],[9,7],color=[teal,coral],width=.65)
ax.set_ylim(0,11);ax.set_ylabel('Episodes (n = 16)');ax.set_yticks([0,4,8])
for i,n in enumerate([9,7]):ax.text(i,n+.25,str(n),ha='center',weight='bold')
ax.set_title('Unedited stacking policy',fontsize=7.4,weight='bold')
err=a['replay_errors_authoritative_json']
bx.bar(['0','450','885'],err,color=teal,width=.6);bx.axhline(.02,ls='--',color=coral,lw=1.3)
bx.set_ylim(0,.025);bx.set_ylabel('Max. abs. action error');bx.set_xlabel('Recorded control step')
bx.set_yticks([0,.01,.02]);bx.set_title('Recorded-request replay',fontsize=7.4,weight='bold')
bx.text(2.35,.0205,'0.02 limit',ha='right',color=coral,fontsize=6.5)
for i,v in enumerate(err):bx.text(i,v+.0007,f'{v:.5f}',ha='center',fontsize=6.3,weight='bold')
for aa in [ax,bx]:aa.grid(axis='y',alpha=.18);aa.set_axisbelow(True)
save(fig,'baseline_replay')

study_path = ROOT/'data/development_results.json'
if study_path.is_file():
 study=json.loads(study_path.read_text())
 fig,ax=plt.subplots(figsize=(3.5,1.7),layout='constrained')
 for i,name in enumerate(['base','gripper_0p4','gripper_0p6']):
  item=study['development'][name]
  if 'successes' in item:
   n=item['successes']
   ax.bar(i,n,width=.55,color=muted if i==0 else teal)
   ax.text(i,n+.7,f'{n}/32',ha='center',fontsize=8,weight='bold')
  else:
   ax.text(i,12,'Pending',ha='center',fontsize=8,color=muted)
 ax.set_xticks([0,1,2],['0.5 (base)','0.4','0.6'])
 ax.set_xlim(-.6,2.6);ax.set_ylim(0,36);ax.set_yticks([0,8,16,24,32])
 ax.set_ylabel('Successful episodes');ax.set_xlabel('Runtime gripper threshold')
 ax.set_title('Development: original policy weights',fontsize=8,weight='bold')
 ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 save(fig,'development_controls')

fig,axs=plt.subplots(1,2,figsize=(7,3.5),layout='constrained')
for ax,path,title,caption in zip(axs,['success_wrist.png','failure_wrist.png'],
 ['Successful episode','Unsuccessful episode'],['Last recorded request: step 435','Replay request: step 885']):
 ax.imshow(Image.open(ROOT/'data'/path));ax.axis('off');ax.set_title(title,weight='bold',fontsize=12)
 ax.text(.5,-.06,caption,transform=ax.transAxes,ha='center',fontsize=9.5)
save(fig,'camera_examples')
print('Generated separate paper/blog loop diagrams and measured-outcomes figures.')
