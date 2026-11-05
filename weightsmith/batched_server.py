"""Batched, hot-swappable pi0.5 policy server.

Protocol (msgpack over websocket, openpi-compatible):
  * plain obs dict              -> single inference (identical to openpi Policy.infer)
  * {"__batch__": [obs, ...]}   -> {"__batch__": [out, ...]} one fused forward pass
  * {"__cmd__": "load", "workspace": path|None} -> reset to base, apply edits
  * {"__cmd__": "info"}        -> current edit report
Per-obs "sampling_seed" gives noise = normal(key(seed), (1,H,D)), exactly the
single-request path, so batched == unbatched given seeds.
"""
from __future__ import annotations

import argparse, asyncio, hashlib, http, json, logging, sys, time, traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `weightsmith` importable
from weightsmith import config as C  # noqa: E402

CHECKPOINT = C.CHECKPOINT
log = logging.getLogger("batched_server")


class Server:
    def __init__(self, config_name="pi05_droid_jointpos"):
        import jax
        import flax.nnx as nnx
        from openpi.policies import policy_config
        from openpi.training import config as training_config
        from openpi.models import model as _model

        self.jax, self.nnx, self._model = jax, nnx, _model
        self.policy = policy_config.create_trained_policy(training_config.get_config(config_name), CHECKPOINT)
        model = self.policy._model
        self.graphdef, self.state = nnx.split(model)
        self.H, self.D = model.action_horizon, model.action_dim
        graphdef = self.graphdef

        def fun(state, rng, observation, noise):
            module = nnx.merge(graphdef, state)
            return module.sample_actions(rng, observation, noise=noise)

        self.jitted = jax.jit(fun)
        self.edited = {}  # name -> (entry, workspace) currently applied
        self.report = {"workspace": None, "edited_tensors": {}}

    # ---------- weights ----------
    def _variable(self, path):
        node = self.state
        for key in path:
            node = node[key]
        return node

    def _set(self, entry, value):
        import jax.numpy as jnp
        var = self._variable(entry["model_path"])
        idx = entry["scan_index"]
        v = jnp.asarray(value, dtype=var.value.dtype)
        var.value = v if idx is None else var.value.at[idx].set(v)

    def load(self, workspace):
        from weightsmith.lab.openpi_vlm_full import validate_workspace
        # restore previously edited tensors to base
        for name, (entry, ws) in self.edited.items():
            self._set(entry, np.load(Path(ws) / "base" / entry["file"]))
        self.edited = {}
        self.report = {"workspace": None, "edited_tensors": {}}
        if workspace:
            ws = Path(workspace)
            cfg = json.loads((ws / "workspace.json").read_text())
            rep = validate_workspace(ws)
            h = hashlib.sha256()
            for name in sorted(rep["edited_tensors"]):
                entry = cfg["tensors"][name]
                arr = np.load(ws / "edits" / entry["file"])
                base = np.load(ws / "base" / entry["file"])
                var = self._variable(entry["model_path"])
                idx = entry["scan_index"]
                live = np.asarray(var.value if idx is None else var.value[idx]).astype(np.float32)
                if not np.array_equal(live, base):  # same guard as the activation-replay path (_apply_edits)
                    raise ValueError(f"Loaded checkpoint does not match workspace base for {name}")
                h.update(name.encode()); h.update(arr.tobytes())
                self._set(entry, arr)
                applied = np.asarray(self._variable(entry["model_path"]).value if idx is None
                                     else self._variable(entry["model_path"]).value[idx]).astype(np.float32)
                if not np.array_equal(applied, arr):
                    raise ValueError(f"Applied value does not match the edit for {name}")
                self.edited[name] = (entry, str(ws))
            rep["fingerprint"] = h.hexdigest()
            rep["workspace"] = str(ws)
            self.report = rep
        log.info("loaded %s", json.dumps({k: v for k, v in self.report.items() if k != "edited_tensors"}, default=str))
        return self.report

    def warmup(self):
        dummy = {"observation/exterior_image_1_left": np.zeros((224, 224, 3), np.uint8),
                 "observation/wrist_image_left": np.zeros((224, 224, 3), np.uint8),
                 "observation/joint_position": np.zeros(7, np.float32),
                 "observation/gripper_position": np.zeros(1, np.float32),
                 "prompt": "warm up", "sampling_seed": 1}
        for b in (1, 2, 4, 8, 16):
            t = time.monotonic()
            self.infer_batch([dummy] * b)
            log.info("warmup batch=%d %.1fs", b, time.monotonic() - t)

    # ---------- inference ----------
    def infer_batch(self, obs_list):
        jax, jnp = self.jax, self.jax.numpy
        p = self.policy
        inputs, noises = [], []
        for obs in obs_list:
            obs = dict(obs)
            seed = obs.pop("sampling_seed", None)
            if seed is None:
                seed = int(np.random.randint(0, 2**31 - 1))
            inputs.append(p._input_transform(obs))
            noises.append(jax.random.normal(jax.random.key(int(seed)), (1, self.H, self.D)))
        n = len(inputs)
        bucket = 1
        while bucket < n:
            bucket *= 2
        pad = bucket - n
        stacked = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs + xs[:1] * pad]), *inputs)
        noise = jnp.concatenate(noises + noises[:1] * pad, axis=0)
        observation = self._model.Observation.from_dict(stacked)
        t = time.monotonic()
        actions = np.asarray(self.jitted(self.state, jax.random.key(0), observation, noise))
        ms = (time.monotonic() - t) * 1000
        state = np.asarray(stacked["state"])
        outs = []
        for i in range(n):
            o = p._output_transform({"state": state[i], "actions": actions[i]})
            o["policy_timing"] = {"infer_ms": ms, "batch": n}
            outs.append(o)
        return outs


async def handler(websocket, srv):
    from openpi_client import msgpack_numpy
    import websockets
    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack(dict(srv.policy.metadata, weight_space_edit=srv.report)))
    loop = asyncio.get_running_loop()
    while True:
        try:
            msg = msgpack_numpy.unpackb(await websocket.recv())
            if "__cmd__" in msg:
                async with srv.alock:
                    if msg["__cmd__"] == "load":
                        rep = await loop.run_in_executor(srv.pool, srv.load, msg.get("workspace"))
                        out = {"ok": True, "report": json.loads(json.dumps(rep, default=str))}
                    else:
                        out = {"ok": True, "report": json.loads(json.dumps(srv.report, default=str))}
            else:
                batch = msg["__batch__"] if "__batch__" in msg else [msg]
                fut = loop.create_future()
                await srv.queue.put((batch, fut))
                res = await fut
                out = {"__batch__": res} if "__batch__" in msg else res[0]
            await websocket.send(packer.pack(out))
        except websockets.ConnectionClosed:
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close()
            raise


async def batcher(srv):
    """Merge requests from all connected sims into one fused forward pass."""
    loop = asyncio.get_running_loop()
    while True:
        items = [await srv.queue.get()]
        await asyncio.sleep(0.004)
        while not srv.queue.empty() and sum(len(b) for b, _ in items) < srv.max_batch:
            items.append(srv.queue.get_nowait())
        obs = [o for b, _ in items for o in b]
        try:
            async with srv.alock:
                outs = await loop.run_in_executor(srv.pool, srv.infer_batch, obs)
            i = 0
            for b, fut in items:
                fut.set_result(outs[i:i + len(b)])
                i += len(b)
        except Exception as e:
            for _, fut in items:
                if not fut.done():
                    fut.set_exception(e)


def health(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def main_async(srv, port):
    import concurrent.futures
    import websockets.asyncio.server as _server
    srv.queue = asyncio.Queue()
    srv.alock = asyncio.Lock()
    srv.pool = concurrent.futures.ThreadPoolExecutor(1)
    asyncio.get_running_loop().create_task(batcher(srv))
    async with _server.serve(lambda ws: handler(ws, srv), "0.0.0.0", port, compression=None,
                             max_size=None, process_request=health, ping_interval=None,
                             open_timeout=120) as s:
        await s.serve_forever()


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--workspace", default=None)
    a = ap.parse_args()
    srv = Server()
    srv.max_batch = 16
    srv.load(a.workspace)
    srv.warmup()
    asyncio.run(main_async(srv, a.port))


if __name__ == "__main__":
    main()
