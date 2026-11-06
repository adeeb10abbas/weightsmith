"""Batched, hot-swappable pi0.5 policy server.

Protocol (msgpack over websocket, openpi-compatible):
  * plain obs dict              -> single API request evaluated in the fixed16 runtime
  * {"__batch__": [obs, ...]}   -> {"__batch__": [out, ...]} one fused forward pass
  * {"__cmd__": "load", "workspace": path|None} -> reset to base, apply edits
  * {"__cmd__": "info"}        -> current edit report
Per-obs "sampling_seed" gives noise = normal(key(seed), (1,H,D)), exactly the
single-request path, v2 always executes batch16; cross-shape equality is NOT assumed.
"""
from __future__ import annotations

import argparse, asyncio, http, json, logging, sys, time, traceback, uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `weightsmith` importable
from weightsmith import config as C  # noqa: E402
from weightsmith.integrity import workspace_fingerprint  # noqa: E402

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
        self.edited = {}  # name -> (entry, original checkpoint array)
        self.report = {"workspace": None, "fingerprint": "base", "edited_tensors": {}}
        self.instance_id = uuid.uuid4().hex

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
        from weightsmith.lab.openpi_vlm_full import _apply_edits, validate_workspace
        # JAX array leaves are immutable. Clone the state containers so failed
        # validation never changes the currently served model or its report.
        old_state = self.state
        self.state = self.jax.tree.map(lambda value: value, old_state)
        originals = {}
        report = {"workspace": None, "fingerprint": "base", "edited_tensors": {}}
        try:
            for entry, original in self.edited.values():
                self._set(entry, original)
            if workspace:
                ws = Path(workspace).resolve()
                fingerprint = workspace_fingerprint(ws)
                cfg = json.loads((ws / "workspace.json").read_text())
                validated = validate_workspace(ws)
                for name in validated["edited_tensors"]:
                    entry = cfg["tensors"][name]
                    variable = self._variable(entry["model_path"])
                    index = entry["scan_index"]
                    value = variable.value if index is None else variable.value[index]
                    originals[name] = (entry, np.asarray(value).copy())
                model = self.nnx.merge(self.graphdef, self.state)
                report = _apply_edits(model, ws)
                if workspace_fingerprint(ws) != fingerprint:
                    raise ValueError("Workspace changed while the server loaded it")
                _, self.state = self.nnx.split(model)
                report.update(fingerprint=fingerprint, workspace=str(ws))
        except BaseException:
            self.state = old_state
            raise
        self.edited = originals
        report.update(capture_protocol_id="wsp-batch-capture-v2-fixed16", runtime_variant="fixed16_new_runtime")
        self.report = report
        log.info("loaded %s", json.dumps({k: v for k, v in self.report.items() if k != "edited_tensors"}, default=str))
        return self.report

    def warmup(self):
        dummy = {"observation/exterior_image_1_left": np.zeros((224, 224, 3), np.uint8),
                 "observation/wrist_image_left": np.zeros((224, 224, 3), np.uint8),
                 "observation/joint_position": np.zeros(7, np.float32),
                 "observation/gripper_position": np.zeros(1, np.float32),
                 "prompt": "warm up", "sampling_seed": 1}
        for b in (16,):
            t = time.monotonic()
            self.infer_batch([dummy] * b)
            log.info("warmup batch=%d %.1fs", b, time.monotonic() - t)

    # ---------- inference ----------
    def infer_context(self, obs_list):
        from weightsmith.lab.batch_runtime_v2 import native_forward
        return native_forward(self, obs_list)

    def infer_batch(self, obs_list):
        import uuid
        from weightsmith.lab.batch_runtime_v2 import project_identity
        from weightsmith.lab.batch_context_v2 import write_packet, pack_requests, unpack_requests
        recording = getattr(self, "record_actual_batches", False)
        if recording:
            arrays, _ = pack_requests(obs_list)
            original_requests = unpack_requests(arrays)  # snapshot before any input transform
            identity_before = project_identity(self.report["fingerprint"])
        outputs, context = self.infer_context(obs_list)
        if getattr(self, "record_actual_batches", False):
            packet_id = uuid.uuid4().hex
            identity = project_identity(self.report["fingerprint"])
            if identity != identity_before: raise RuntimeError("runtime/policy changed during live inference")
            packet = write_packet(C.PROJECT_ROOT / "data/batch_packets" / packet_id,
                original_requests, np.stack([o["actions"] for o in outputs]), identity,
                variant="fixed16_new_runtime")
            for position, output in enumerate(outputs):
                output["_wsp_batch_v2"] = {"packet_id": packet_id,
                    "packet_sha256": packet["packet_sha256"], "position": position,
                    "real_count": len(obs_list), "padded_count": 16,
                    "protocol_id": "wsp-batch-capture-v2-fixed16"}
        return outputs


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


def health(connection, request, srv):
    if request.path == "/healthz":
        report = {k: v for k, v in srv.report.items() if k != "edited_tensors"}
        return connection.respond(http.HTTPStatus.OK, json.dumps({"instance_id": srv.instance_id,
                                                                 "report": report}, default=str) + "\n")
    return None


async def main_async(srv, port):
    import concurrent.futures
    import websockets.asyncio.server as _server
    srv.queue = asyncio.Queue()
    srv.alock = asyncio.Lock()
    srv.pool = concurrent.futures.ThreadPoolExecutor(1)
    asyncio.get_running_loop().create_task(batcher(srv))
    async with _server.serve(lambda ws: handler(ws, srv), "0.0.0.0", port, compression=None,
                             max_size=None, process_request=lambda c, r: health(c, r, srv), ping_interval=None,
                             open_timeout=120) as s:
        await s.serve_forever()


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--instance-id", default=None)
    a = ap.parse_args()
    from weightsmith.lab.batch_runtime_v2 import project_identity
    project_identity("base")  # CPU-only binding validation before model initialization
    srv = Server()
    if a.instance_id:
        srv.instance_id = a.instance_id
    srv.max_batch = 16
    srv.load(a.workspace)
    srv.warmup()
    srv.record_actual_batches = True
    asyncio.run(main_async(srv, a.port))


if __name__ == "__main__":
    main()
