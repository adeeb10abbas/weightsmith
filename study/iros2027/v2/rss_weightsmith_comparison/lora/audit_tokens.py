"""Check archived and current token correspondence without loading policy weights."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("historical_lora", Path(__file__).with_name("historical_lora.py"))
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from openpi import transforms as T
    from openpi.training import config, checkpoints
    from weightsmith import config as C
    out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    dataset = Path(args.dataset)
    meta = json.loads((dataset / "dataset.json").read_text())
    index_path = H.SOURCE / "data/acts/c2-multi/index.json"
    index = json.loads(index_path.read_text())
    cfg = config.get_config("pi05_droid_jointpos")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    stats = checkpoints.load_norm_stats(C.CHECKPOINT / "assets", data.asset_id)
    transform = T.compose([*data.data_transforms.inputs, T.Normalize(stats, use_quantiles=data.use_quantile_norm),
                           *data.model_transforms.inputs])
    slots = np.array(index["prompt_token_slots"])
    rows = []
    for item in meta["records"]:
        with np.load(item["obs_path"], allow_pickle=False) as z:
            raw = {k: z[k][item["decision_index"]].copy() for k in ("base_img", "wrist_img", "joint", "gripper")}
        for label, prompt in (("original", H.ORIGINAL), ("teacher", H.TEACHER)):
            current = transform({**H.raw_request(raw), "prompt": prompt})
            archived = index["items"][item[label+"_row"]]
            same_ids = bool(np.array_equal(current["tokenized_prompt"][slots], archived["prompt_token_ids_at_slots"]))
            same_mask = bool(np.array_equal(current["tokenized_prompt_mask"][slots], archived["prompt_token_valid_at_slots"]))
            rows.append({"item_id": item["item_id"], "instruction": label, "token_ids_match": same_ids,
                         "token_mask_matches": same_mask, "captured_positions": len(slots)})
    passed = all(r["token_ids_match"] and r["token_mask_matches"] for r in rows)
    H.dump(out, {"status": "pass" if passed else "fail", "rows": rows,
                 "dataset_sha256": H.sha(dataset / "dataset.json"), "capture_index_sha256": H.sha(index_path),
                 "code_sha256": H.sha(__file__), "policy_forward_calls": 0,
                 "interpretation": "Token correspondence alone does not prove numerical action parity."})
    print(json.dumps({"status": "pass" if passed else "fail", "paired_inputs": len(rows), "output": str(out)}))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
