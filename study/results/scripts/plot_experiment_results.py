#!/usr/bin/env python3
"""Render observed discovery and final curves from collector-validated JSON only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER=Path(__file__).resolve().parents[1]
data=json.loads((PAPER/"data/experiment_results.json").read_text())
figdir=PAPER/"figures"; blog=PAPER.parent/"blog/figures"
blue="#24465A"; orange="#A64B2A"; grey="#72787D"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
                     "grid.alpha":.16,"axes.axisbelow":True,"savefig.facecolor":"white"})
def save(fig,name):
    for suffix in ("pdf","svg","png"): fig.savefig(figdir/f"{name}.{suffix}",dpi=220,bbox_inches="tight",pad_inches=.06)
    for suffix in ("svg","png"): fig.savefig(blog/f"{name}.{suffix}",dpi=220,bbox_inches="tight",pad_inches=.06)
    plt.close(fig)

active=[(name,run) for name,run in data["discovery"].items()
        if any(p["status"]=="complete" for p in run["proposals"])]
if not active: active=[("No editor run started",{"mode":"adaptive","proposals":[]})]
ncols=min(2,len(active));nrows=(len(active)+ncols-1)//ncols
fig,axes=plt.subplots(nrows,ncols,figsize=(7,2.0*nrows),squeeze=False,layout="constrained")
base=data["base_development"];base_rate=base["success"]/base["n"]
for ax,(name,run) in zip(axes.flat,active):
    complete=[p for p in run["proposals"] if p["status"]=="complete"]
    byname={p["id"]:p for p in complete}
    ax.errorbar([0],[base_rate],yerr=[[base_rate-base["wilson_95"][0]],[base["wilson_95"][1]-base_rate]],
                fmt="s",color=blue,capsize=3,label="Pinned base / incumbent" if run['mode']=='adaptive' else 'Base / selected finalist')
    if complete:
        xs=[p["round"] for p in complete];ys=[p["analysis"]["candidate_rate_paired"] for p in complete]
        errors=np.array([[y-p["success_wilson_95"][0],p["success_wilson_95"][1]-y] for y,p in zip(ys,complete)]).T
        ax.errorbar(xs,ys,yerr=errors,fmt="o--",color=orange,capsize=3,label="Proposed policy",lw=1.1)
        for p,x,y in zip(complete,xs,ys):
            tag="no edit" if p["changed_elements"]==0 else f"{p['changed_rows']} row"+('s' if p['changed_rows']!=1 else '')
            ax.annotate(f"{p['analysis']['candidate_success']}/32\n{tag}",(x,y),xytext=(0,8),
                        textcoords="offset points",ha="center",fontsize=8,color=orange,linespacing=1.0)
        if run["mode"]=="adaptive":
            accepted=[base_rate]+[base_rate if p["incumbent_after_round"]=="base"
                                  else byname[p["incumbent_after_round"]]["analysis"]["candidate_rate_paired"] for p in complete]
            ax.plot([0]+xs,accepted,"s-",color=blue,lw=1.6)
        elif len(complete)==3:
            chosen=run["incumbent"]
            selected=base_rate if chosen=="base" else byname[chosen]["analysis"]["candidate_rate_paired"]
            ax.scatter([3.12],[selected],marker="s",color=blue,zorder=5)
    for r in (1,2,3):
        if not any(p["round"]==r for p in complete):
            ax.text(r,.07,"Not completed" if run.get("status")=="stopped_partial" else "Pending",ha="center",color=grey,fontsize=8)
    ax.set(title=name.replace("-"," "),xlabel="Proposal / round",ylabel="Development success fraction",
           xlim=(-.25,3.35),ylim=(0,1),xticks=[0,1,2,3],xticklabels=["Base","1","2","3"])
    ax.legend(loc="upper left",fontsize=7.5,frameon=False)
for ax in list(axes.flat)[len(active):]:ax.axis("off")
save(fig,"discovery_curves")

observed=[(run,p) for run,e in data["discovery"].items() for p in e["proposals"] if p["status"]=="complete"]
fig,(ax,bx)=plt.subplots(1,2,figsize=(7,max(1.8,.38*len(observed)+.8)),
                         gridspec_kw={"width_ratios":[1.4,1]},layout="constrained")
for i,(run,p) in enumerate(observed):
    a=p["analysis"];v=a["success_difference"];lo,hi=a["paired_bootstrap_95"]
    color=grey if p["changed_elements"]==0 else orange
    for j,block in enumerate(p.get("block_sensitivity",[])):
        ax.scatter([block["success_difference"]],[i+(j-.5)*.18],marker="D",color=color,alpha=.25,s=21)
    ax.errorbar([v],[i],xerr=[[v-lo],[hi-v]],fmt="o",color=color,capsize=3)
    bx.barh(i,a["fixed"],color=blue,label="Fixed" if i==0 else None)
    bx.barh(i,-a["broken"],color=orange,label="Broken" if i==0 else None)
    bx.text(a["fixed"]+.2,i,str(a["fixed"]),va="center",fontsize=8)
    bx.text(-a["broken"]-.2,i,str(a["broken"]),va="center",ha="right",fontsize=8)
labels=[f"{r} / {p['id']}"+(" (identity)" if not p["changed_elements"] else "") for r,p in observed]
ax.axvline(0,color=grey,lw=.8);ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel="Candidate - base success fraction")
bx.axvline(0,color=grey,lw=.8);bx.set(yticks=range(len(labels)),yticklabels=[],xlabel="Paired episode transitions")
for axis in (ax,bx): axis.set_ylim(len(observed)-.5,-.5)
if observed:
    bx.set_xlim(-max(p['analysis']['broken'] for _,p in observed)-1.5,max(p['analysis']['fixed'] for _,p in observed)+1.5)
bx.legend(frameon=False,fontsize=8,loc="lower center",bbox_to_anchor=(.5,1.02),ncol=2)
if not observed:ax.text(.5,.5,"No completed proposals",transform=ax.transAxes,ha="center")
save(fig,"discovery_effects")

if "evaluation_plan" in data:
    fig,ax=plt.subplots(figsize=(7,2.65),layout="constrained")
    if not data["heldout"]:
        ax.axis("off")
        ax.text(.5,.63,"Prospective v1.1: unique deployed-policy estimates",ha="center",transform=ax.transAxes,weight="bold")
        ax.text(.5,.42,"Held-out results pending; shared round labels are not independent curves.",ha="center",transform=ax.transAxes)
    else:
        for i,(name,a) in enumerate(data["heldout"].items()):
            v=a["success_difference"];lo,hi=a["paired_bootstrap_95"]
            ax.errorbar([v],[i],xerr=[[v-lo],[hi-v]],fmt="s",color=blue,capsize=3)
        ax.set(yticks=range(len(data["heldout"])),yticklabels=list(data["heldout"]),xlabel="Candidate - fresh base success fraction")
        ax.axvline(0,color=grey,lw=.8);ax.invert_yaxis()
    save(fig,"heldout_unique_policies")
else:
    fig,axes=plt.subplots(1,2,figsize=(7,2.65),layout="constrained")
    for restart,ax in zip((1,2),axes):
        found=[];bounds=[-.15,.15]
        for r in (1,2,3):
            cid=f"adaptive-r{restart}-round{r}";a=data["heldout"].get(cid)
            if a is None:
                ax.text(r,.06,"Pending",ha="center",color=grey,fontsize=8,transform=ax.get_xaxis_transform());continue
            v=a["success_difference"];lo,hi=a["paired_bootstrap_95"]
            bounds.extend([lo,hi])
            ax.errorbar([r],[v],yerr=[[v-lo],[hi-v]],fmt="s",color=blue,capsize=3)
            found.append((r,v))
            ax.annotate(a["metadata"]["incumbent"],(r,v),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8)
        if found:ax.plot(*zip(*found),color=blue,lw=1)
        ax.axhline(0,color=grey,lw=.8)
        ax.set(title=f"Adaptive restart {restart}",xlabel="Round-end incumbent",ylabel="Held-out success difference",
               xticks=[1,2,3],xlim=(.7,3.3),ylim=(min(bounds)-.08,max(bounds)+.08))
    save(fig,"heldout_incumbent_curves")
if "identity_input_diagnostic" in data:
    rows=data["identity_input_diagnostic"]["rows"]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7,2.5),layout="constrained")
    for i,(field,label) in enumerate((("base_img","Base camera"),("wrist_img","Wrist camera"))):
        values=sorted(r["differences"][field]["mean_abs_per_request"][0] for r in rows)
        jitter=np.linspace(-.12,.12,len(values))
        ax.scatter(i+jitter,values,s=15,color=blue,alpha=.65)
        ax.plot([i-.18,i+.18],[np.median(values)]*2,color=orange,lw=2)
    ax.set(xticks=[0,1],xticklabels=["Base camera","Wrist camera"],ylabel="First-image mean absolute difference\n(8-bit pixel intensity)",xlim=(-.5,1.5))
    values=sorted(r["differences"]["actions"]["max_abs_per_request"][0] for r in rows)
    bx.plot(values,np.arange(1,33)/32,color=blue,drawstyle="steps-post",lw=1.5)
    bx.set(xlabel="First raw-action array max. absolute difference\n(joint / gripper coordinates)",ylabel="Fraction of 32 paired episodes",ylim=(0,1.04))
    bx.text(.97,.16,"Same weights, seeds, and initial\nrobot joint / gripper inputs",transform=bx.transAxes,ha="right",fontsize=8)
    save(fig,"identity_input_diagnostic")
if "batch_diagnostic" in data:
    items=data["batch_diagnostic"]["report"]["items"]
    with plt.rc_context({"font.size":7}):
        fig,ax=plt.subplots(figsize=(3.5,2.2),layout="constrained")
        series=(("native_b1_vs_historical","B1 vs recorded",orange,"o"),
                ("native_b16_vs_historical","B16 vs recorded",blue,"s"),
                ("instrumented_vs_native_b1","Instrumented vs B1",grey,"^"))
        for key,label,color,marker in series:
            values=[r["comparisons"][key]["max_abs"] for r in items]
            if any(v<=0 for v in values):raise ValueError("Log diagnostic requires positive observed errors; handle zeros explicitly")
            ax.plot(range(6),values,marker=marker,markersize=3,color=color,lw=1,label=label)
        ax.axhline(.02,color=grey,lw=.8,ls="--")
        ax.axvspan(-.35,2.5,color=orange,alpha=.055)
        ax.set(yscale="log",ylabel="Raw-action array max. abs. difference",xlabel="Selected capture-item index",
               xticks=range(6),xticklabels=[str(r["capture_item"]) for r in items],xlim=(-.35,5.35),ylim=(3e-8,1))
        ax.legend(frameon=True,facecolor="white",edgecolor="none",framealpha=.96,fontsize=6.3,loc="upper right",ncol=1)
        if "coordinate_audit" in data["batch_diagnostic"]:
            ax.text(.02,.16,"B16: exact after float32 recording cast",transform=ax.transAxes,fontsize=6.2,color=blue)
        save(fig,"batch_shape_replay")
print(f"Plotted {len(observed)} verified proposals and {len(data['heldout'])} final comparisons; missing results remain pending.")
if 'followup_v2' in data:
    from plot_v2_results import render
    n,m=render(data['followup_v2'])
    print(f"Separate v2 plots: {n} completed proposals, {m} verified final comparisons.")
if 'base_development_slot_outcomes' in data:
    from plot_identity_repeats import render as render_identity
    print(f"Verified identity repeats: {render_identity(data)}; no incomplete columns plotted.")
