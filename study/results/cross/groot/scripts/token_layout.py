"""Token layout of GR00T N1.7's backbone prompt for given wordings (CPU only; Isaac-GR00T venv).

Uses the checkpoint's own processor (eval transforms, Qwen3-VL chat template) on two dummy 180x320 frames and prints the
token sequence, image-token span, the instruction span and the LCS alignment used by groot_build.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_build import align  # noqa: E402

CKPT = str(Path.home() / ".cache/huggingface/hub/models--nvidia--GR00T-N1.7-DROID/snapshots/05e7cc97e40dbd33b0890c35cc0214fcb0547ab5")


def main():
    from transformers import AutoProcessor
    import gr00t.model  # noqa: F401  (registers the processor class)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData
    proc = AutoProcessor.from_pretrained(CKPT)
    proc.eval()
    tag = EmbodimentTag.resolve("OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    wordings = json.loads(sys.argv[1])
    tok = proc.collator.processor.tokenizer
    out = {}
    for name, text in wordings.items():
        img = np.zeros((1, 180, 320, 3), np.uint8)
        step = VLAStepData(images={"exterior_image_1_left": img, "wrist_image_left": img},
                           states={"eef_9d": np.zeros((1, 9), np.float32), "gripper_position": np.zeros((1, 1), np.float32),
                                   "joint_position": np.zeros((1, 7), np.float32)},
                           actions={}, text=text, embodiment=tag)
        feat = proc([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        batch = proc.collator([feat])["inputs"]
        ids = batch["input_ids"][0].tolist()
        out[name] = ids
        img_id = proc.collator.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        n_img = sum(i == img_id for i in ids)
        last_img = max(i for i, t in enumerate(ids) if t == img_id)
        print(f"{name}: {len(ids)} tokens, {n_img} image tokens (last at {last_img}), pixel_values {tuple(batch['pixel_values'].shape)}")
        print("  after images:", [tok.decode([t]) for t in ids[last_img + 1:]])
        print("  before images:", [tok.decode([t]) for t in ids[:min(8, len(ids))]])
    names = list(wordings)
    if len(names) == 2:
        d, pmap = align(out[names[0]], out[names[1]])
        tokn = [tok.decode([out[names[0]][p]]) for p in pmap]
        print("first divergence", d, "aligned", len(pmap), list(zip(pmap.keys(), pmap.values(), tokn)))


if __name__ == "__main__":
    main()
