#!/usr/bin/env python3
"""Operator-owned, newline-JSON MCP bridge for one frozen RQ1-RQ3 editor run.

The editor receives neither a shell nor SSH credentials. All remote operations are
bound to the operator configuration; only practice data and declared tools exist.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = ("p1", "p2", "p3")
NAME = re.compile(r"^[a-zA-Z0-9_.@-]+$")
MAX_TEXT = 160_000
from study_validation import normalize_activation_spec, validate_episode_id


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def summarize_study(state, mode):
    """Separate the editor stopping from completion of the three-proposal design."""
    evaluations = state.get("evaluations", {})
    scored = [name for name in CANDIDATES if name in evaluations]
    return {
        "study_status": "complete" if len(scored) == 3 else "partial",
        "attempted_proposals": len(state.get("eval_reserved", [])),
        "completed_proposals": len(scored),
        "unscored_proposals": [name for name in CANDIDATES if name not in evaluations],
        "final_candidate": state.get("incumbent", "base"),
        "round_states": [{"round": CANDIDATES.index(name) + 1,
                          "proposal": name,
                          "incumbent": evaluations[name]["incumbent_after_round"]}
                         for name in scored] if mode == "adaptive" else [],
    }


class Broker:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.cfg = json.loads(self.config_path.read_text())
        self.agent = Path(self.cfg["agent_dir"]).resolve()
        self.state_dir = Path(self.cfg["state_dir"]).resolve()
        if self.state_dir == self.agent or self.agent in self.state_dir.parents:
            raise ValueError("operator state must be outside the editor workspace")
        if self.cfg["mode"] not in ("adaptive", "nonadaptive"):
            raise ValueError("mode must be adaptive or nonadaptive")
        self.state_path = self.state_dir / "broker_state.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.instance_lock=(self.state_dir/"broker.lock").open("a")
        try:fcntl.flock(self.instance_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.instance_lock.close();self.instance_lock=None
            raise ValueError("this run already has an active broker")
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state["config_sha256"] != digest(self.config_path):
                raise ValueError("run configuration changed after start")
        else:
            self.state = {"config_sha256": digest(self.config_path), "created": [], "committed": {},
                          "eval_reserved": [], "evaluations": {}, "acts_items_reserved": 0,
                          "incumbent": "base", "completed": False}
            self.save()
        self.log_path = self.state_dir / "broker_calls.jsonl"

    def release(self):
        if self.instance_lock is not None:
            self.instance_lock.close();self.instance_lock=None

    def save(self):
        atomic_json(self.state_path, self.state)

    def remaining(self):
        left = float(self.cfg["deadline_unix"]) - time.time()
        if left <= 0:
            raise ValueError("The 7200-second run budget is exhausted; no new tools are allowed")
        return left

    def path(self, value, *, create=False):
        if not isinstance(value, str) or not value or len(value) > 256:
            raise ValueError("relative workspace path required")
        raw = Path(value)
        if raw.is_absolute() or any(p in ("..", ".codex", ".git", ".agents") for p in raw.parts):
            raise ValueError("path must stay in the editor workspace")
        target = self.agent / raw
        # Reject symlinks before resolving, even when they point back into the workspace.
        for component in [target, *target.parents]:
            if component == self.agent:
                break
            if component.is_symlink():
                raise ValueError("workspace symlinks are not supported")
        resolved = target.resolve()
        if self.agent not in resolved.parents:
            raise ValueError("path escapes editor workspace")
        return resolved

    def invoke_remote(self, tool, args, payload=None):
        remaining=self.remaining()
        if remaining<15:raise ValueError('insufficient wall time for guarded remote cleanup')
        host=self.cfg['host']
        if not host or host.startswith('-'):raise ValueError('fixed SSH host required')
        remote_args=[self.cfg.get('remote_python','python3'),'drivers/study_remote.py',
                     '--root',self.cfg['remote_root'],'--tool',tool,'--seconds',str(remaining-10),'--',*args]
        command='cd '+shlex.quote(self.cfg['remote_repo'])+' && '+shlex.join(remote_args)
        argv=['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=2']
        if self.cfg.get('ssh_port'):argv+=['-p',str(int(self.cfg['ssh_port']))]
        argv += [host,command]
        begin=time.time()
        try:result=subprocess.run(argv,input=payload,capture_output=True,text=True,timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise ValueError('remote cleanup did not finish before local deadline; operator must inspect owned workers') from error
        with self.log_path.open('a') as stream:
            stream.write(json.dumps({'started_unix':begin,'elapsed_seconds':time.time()-begin,'tool':tool,
                                     'args':args,'input':payload,'returncode':result.returncode,
                                     'stdout':result.stdout,'stderr':result.stderr})+'\n')
        if result.returncode:raise ValueError('guarded remote operation failed: '+(result.stdout+result.stderr)[-8000:])
        return result.stdout

    def remote(self,args,stdin=None):
        return self.invoke_remote('owsp',args,stdin)

    def declarative(self,spec):
        return json.loads(self.invoke_remote('tensor',[],json.dumps(spec)))

    def candidates(self, name, allow_base=True):
        if name == "base" and allow_base:
            return
        if name not in self.state["created"]:
            raise ValueError("candidate must be one of this run's created p1, p2, p3")

    def diff(self, name):
        report = json.loads(self.remote(["diff", name]))
        if not isinstance(report.get("fingerprint"), str):
            raise ValueError("remote diff must report an exact fingerprint")
        if type(report.get("changed_elements")) is not int or report["changed_elements"]<0:
            raise ValueError("remote diff must report actual changed_elements")
        if not isinstance(report.get("l2_delta"), (int, float)):
            raise ValueError("remote diff must report actual post-bf16 l2_delta")
        return report

    def rows(self, name):
        out = self.remote(["episodes", "--cand", name, "-v"])
        rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        expected = {f"s{i}" for i in range(30016, 30048)}
        if len(rows) != 32 or {r["slot"] for r in rows} != expected:
            raise ValueError(f"{name}: require exactly one complete record for each development slot")
        if any(type(r.get("success")) is not bool for r in rows):raise ValueError("success fields must be booleans")
        if name=="base" and any(r.get("cand")!="base" or r.get("fingerprint")!="base" for r in rows):raise ValueError("original base identity required")
        if name=="base":
            identity=hashlib.sha256(json.dumps(sorted(rows,key=lambda r:r["slot"]),sort_keys=True).encode()).hexdigest()
            if self.cfg.get("expected_base_records_sha256") != identity:raise ValueError("base records do not match the operator-pinned cohort")
            if self.state.get("base_records_sha256") not in (None,identity):raise ValueError("pinned base records changed")
            self.state["base_records_sha256"]=identity;self.save()
        if len({r["fingerprint"] for r in rows}) != 1:
            raise ValueError("mixed candidate fingerprints")
        return {r["slot"]: r for r in rows}

    def gate(self, name):
        base, candidate = self.rows("base"), self.rows(name)
        expected_fp = self.state["committed"][name]["fingerprint"]
        if any(r["fingerprint"] != expected_fp for r in candidate.values()):
            raise ValueError("evaluated candidate fingerprint differs from committed candidate")
        success = sum(bool(r["success"]) for r in candidate.values())
        base_success = sum(bool(r["success"]) for r in base.values())
        fixed = sum(not base[k]["success"] and candidate[k]["success"] for k in base)
        broke = sum(base[k]["success"] and not candidate[k]["success"] for k in base)
        changed=self.state["committed"][name]["diff"]["changed_elements"]
        result = {"changed_elements":changed,"identity_no_edit":changed==0,"success": success, "base_success": base_success, "fixed": fixed, "broke": broke,
                  "eligible": changed > 0 and success > base_success and broke <= 2,
                  "l2_delta": self.state["committed"][name]["l2_delta"],
                  "ordinal": CANDIDATES.index(name)+1, "fingerprint": expected_fp}
        self.state["evaluations"][name] = result
        ranked = [( -v["success"], v["l2_delta"], v["ordinal"], n)
                  for n, v in self.state["evaluations"].items() if v["eligible"]]
        self.state["incumbent"] = min(ranked)[-1] if ranked else "base"
        result["incumbent_after_round"] = self.state["incumbent"]
        self.save()
        return result

    def call(self, tool, args):
        self.remaining()
        if not self.cfg.get("readiness_only") and not self.state.get("base_records_sha256"):
            self.rows("base")
        if self.state["completed"] and tool != "wsp_status":
            raise ValueError("run has already finished")
        if tool == "workspace_read":
            p = self.path(args["path"])
            if p.stat().st_size > MAX_TEXT:
                raise ValueError("text file exceeds 160 KB; read smaller notes")
            return p.read_text()
        if tool == "workspace_write":
            content = args["text"]
            if len(content.encode()) > MAX_TEXT:
                raise ValueError("text exceeds 160 KB")
            p = self.path(args["path"])
            if p.name in ("BRIEF.md", "PRIOR_FINDINGS.md", "MODE.md"):
                raise ValueError("operator-provided brief is immutable")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return {"written": args["path"]}
        if tool == "workspace_list":
            return [str(p.relative_to(self.agent)) for p in sorted(self.agent.rglob("*"))
                    if p.is_file() and not p.is_symlink() and not any(c.startswith(".") for c in p.relative_to(self.agent).parts)][:1000]
        if tool == "workspace_image":
            p = self.path(args["path"])
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg") or p.stat().st_size > 10_000_000:
                raise ValueError("expected PNG/JPG below 10 MB")
            return {"_image": base64.b64encode(p.read_bytes()).decode(),
                    "mimeType": "image/png" if p.suffix.lower()==".png" else "image/jpeg"}
        if tool == "wsp_status":
            return {"mode": self.cfg["mode"], "seconds_remaining": self.remaining(),
                    "state": self.state, "remote": json.loads(self.remote(["status"]))}
        if tool == "wsp_tensors":
            regex = args.get("regex", "")
            if len(regex) > 200:raise ValueError("regex too long")
            return self.remote(["tensors"] + ([regex] if regex else []))
        if tool == "wsp_new":
            name = args["name"]
            next_index = len(self.state["created"])
            if next_index >= 3 or name != CANDIDATES[next_index]:
                raise ValueError("create p1, p2, p3 in order, at most three proposals")
            if self.cfg["mode"] == "adaptive" and next_index and CANDIDATES[next_index-1] not in self.state["evaluations"]:
                raise ValueError("adaptive mode requires previous evaluation and gate before the next proposal")
            origin = args.get("from_candidate", "base")
            allowed = {"base", self.state["incumbent"]} if self.cfg["mode"] == "adaptive" else {"base"}
            if origin not in allowed:raise ValueError("proposal must start from base or the current accepted incumbent")
            out = self.remote(["new", name, "--from", origin, "--note", args["hypothesis"][:4000]])
            self.state["created"].append(name);self.save()
            return out
        if tool == "wsp_tensor_op":
            spec=dict(args['spec'])
            name=spec.get('candidate','base');self.candidates(name)
            op=spec.get('op')
            if op in ('row_scale','row_add_direction','row_project_out') and name in self.state['committed']:
                raise ValueError('committed candidates cannot be modified')
            if self.cfg['mode']=='nonadaptive' and op=='activation_contrast':spec['base_only']=True
            return self.declarative(spec)
        if tool == "wsp_diff":
            name=args["candidate"];self.candidates(name,False)
            return self.diff(name)
        if tool == "wsp_commit":
            name=args["candidate"];self.candidates(name,False)
            if name in self.state["committed"]:raise ValueError("candidate already committed")
            report=self.diff(name)
            self.state["committed"][name]={"fingerprint":report["fingerprint"],"l2_delta":report["l2_delta"],
                                            "hypothesis":args["hypothesis"],"committed_unix":time.time(),"diff":report}
            self.save();return self.state["committed"][name]
        if tool == "wsp_eval":
            name=args["candidate"]
            if name not in self.state["committed"]:raise ValueError("commit the candidate before evaluation")
            if self.cfg["mode"]=="nonadaptive" and set(self.state["committed"])!=set(CANDIDATES):
                raise ValueError("nonadaptive mode requires all three immutable commitments before any evaluation")
            if name in self.state["eval_reserved"]:raise ValueError("candidate evaluation already attempted; retries require operator audit")
            if self.diff(name)["fingerprint"]!=self.state["committed"][name]["fingerprint"]:
                raise ValueError("committed candidate has changed")
            self.state["eval_reserved"].append(name);self.save()
            out=self.remote(["eval",name,"--n","32","--offset","0"])
            if self.diff(name)["fingerprint"]!=self.state["committed"][name]["fingerprint"]:
                raise ValueError("candidate changed during evaluation")
            return {"output":out,"gate":self.gate(name)}
        if tool == "wsp_episodes":
            name=args.get("candidate")
            if name:self.candidates(name)
            return self.remote(["episodes","-v"]+(["--cand",name] if name else []))
        if tool in ("wsp_show", "wsp_frames"):
            eid=validate_episode_id(args["episode"])
            if tool=="wsp_show":return self.remote(["show",eid])
            steps=args["steps"]
            if not 1<=len(steps)<=12 or any(type(s) is not int or not 0<=s<=900 for s in steps):
                raise ValueError("use 1..12 steps between 0 and 900")
            out=self.remote(["frames",eid,"--steps",','.join(map(str,steps))])
            path=Path(out.strip().splitlines()[-1]).resolve()
            if self.agent not in path.parents:raise ValueError("unexpected local frame path")
            return {"image_path":str(path.relative_to(self.agent))}
        if tool=="wsp_activations":
            spec=normalize_activation_spec(args['spec'],{'base',*self.state['created']},self.cfg['mode'])
            name=spec.get('cand','base');self.candidates(name)
            items=spec['items']
            if self.state['acts_items_reserved']+len(items)>96:
                raise ValueError('96 activation items total, including attempted calls')
            self.state["acts_items_reserved"]+=len(items);self.save()
            return self.remote(["acts"],stdin=json.dumps(spec))
        if tool=="wsp_finish":
            self.state["completed"]=True
            self.state["finished_unix"]=time.time()
            self.state.update(summarize_study(self.state, self.cfg["mode"]))
            self.state["summary"]=args["summary"]
            self.save();return {"final_candidate":self.state["incumbent"],"state":self.state}
        raise ValueError("unknown tool")


def schema(properties=None, required=None):
    return {"type":"object","properties":properties or {},"required":required or [],"additionalProperties":False}
S={"type":"string"}
def tool(name,description,properties=None,required=None):
    return {"name":name,"description":description,"inputSchema":schema(properties,required)}
TOOLS=[
 tool("workspace_read","Read a text file in this editor's workspace.",{"path":S},["path"]),
 tool("workspace_write","Write notes or scripts in this editor's workspace; no parent paths.",{"path":S,"text":S},["path","text"]),
 tool("workspace_list","List files in this editor's workspace."),
 tool("workspace_image","View a PNG/JPG returned by wsp_frames.",{"path":S},["path"]),
 tool("wsp_status","Read budget, proposal commitments, accepted incumbent and remaining wall time."),
 tool("wsp_tensors","List exposed action-MLP down-projection tensors and statistics.",{"regex":S}),
 tool("wsp_new","Create p1, p2, p3 in order. Adaptive: base/current incumbent; nonadaptive: base only.",{"name":S,"from_candidate":S,"hypothesis":S},["name","hypothesis"]),
 tool("wsp_tensor_op","Typed NumPy only: row_stats, row_scale, row_add_direction, row_project_out, activation_contrast, observation, observation_summary. No arbitrary Python. See BRIEF for JSON schema/examples.",{"spec":{"type":"object"}},["spec"]),
 tool("wsp_diff","Validate a proposal, report actual bf16 changes, row/layer support, L2 norm and exact fingerprint.",{"candidate":S},["candidate"]),
 tool("wsp_commit","Freeze a proposal and hypothesis. Nonadaptive: commit all 3 before any evaluation.",{"candidate":S,"hypothesis":S},["candidate","hypothesis"]),
 tool("wsp_eval","Evaluate one committed candidate on the fixed 32 development slots, at most once, and apply the declared acceptance gate.",{"candidate":S},["candidate"]),
 tool("wsp_episodes","Read practice episode summaries, optionally filtered to a candidate.",{"candidate":S}),
 tool("wsp_show","Read events for a practice episode.",{"episode":S},["episode"]),
 tool("wsp_activations","Capture activations on recorded practice requests; total96 attempted items. No prompt override. Nonadaptive: base only.",{"spec":{"type":"object"}},["spec"]),
 tool("wsp_finish","Finish the run. The operator gate fixes the final candidate; base/no-edit is allowed.",{"summary":S},["summary"]),
]


def serve(broker):
    for line in sys.stdin:
        try:
            request=json.loads(line);method=request.get("method");ident=request.get("id")
            if method=="initialize":
                result={"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"weightsmith-study","version":"1.0"}}
            elif method=="tools/list":result={"tools":[t for t in TOOLS if not broker.cfg.get("readiness_only") or t["name"] in {"workspace_list","workspace_read","workspace_write"}]}
            elif method=="tools/call":
                params=request["params"]
                try:
                    if params["name"] not in {t["name"] for t in TOOLS}:raise ValueError("unknown exposed tool")
                    if broker.cfg.get("readiness_only") and params["name"] not in {"workspace_list","workspace_read","workspace_write"}:raise ValueError("readiness-only probe has no remote or image tools")
                    value=broker.call(params["name"],params.get("arguments",{}))
                    if isinstance(value,dict) and "_image" in value:
                        content=[{"type":"image","data":value["_image"],"mimeType":value["mimeType"]}]
                    else:content=[{"type":"text","text":value if isinstance(value,str) else json.dumps(value)}]
                    result={"content":content,"isError":False}
                except Exception as error:result={"content":[{"type":"text","text":str(error)}],"isError":True}
            elif method=="ping":result={}
            elif ident is None:continue
            else:
                print(json.dumps({"jsonrpc":"2.0","id":ident,"error":{"code":-32601,"message":"method not found"}}),flush=True);continue
            if ident is not None:print(json.dumps({"jsonrpc":"2.0","id":ident,"result":result}),flush=True)
        except Exception as error:
            print(json.dumps({"jsonrpc":"2.0","id":None,"error":{"code":-32600,"message":str(error)}}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True)
    serve(Broker(parser.parse_args().config))
