#!/usr/bin/env python3
"""Wrist-camera strip: a cap grasp that slips (s60000) vs one that holds (s60013), MustardInLeftBinTask base policy.
Input: frame sheets rendered by `owsp frames` (6 rows x [base | wrist], 640x360 per tile)."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

STUDY = Path(__file__).resolve().parents[2]
slip_sheet, hold_sheet = Path(sys.argv[1]), Path(sys.argv[2])
SLIP_STEPS, HOLD_STEPS = [40, 55, 70, 85, 95, 100], [55, 75, 100, 130, 160, 180]
TW, TH = 320, 180


def tiles(sheet, steps, keep):
    im = Image.open(sheet).convert("RGB")
    w, h = im.size; rh = h // len(steps)
    return [(steps[i], im.crop((w // 2, i * rh, w, (i + 1) * rh)).resize((TW, TH), Image.LANCZOS)) for i in keep]


rows = [("Slips: base.base.s60000 (grab at step 48, dropped at 98)", tiles(slip_sheet, SLIP_STEPS, [1, 2, 3, 4, 5])),
        ("Holds: base.base.s60013 (grab at step 60, released over bin at 168)", tiles(hold_sheet, HOLD_STEPS, [0, 1, 2, 3, 5]))]
pad, head = 8, 26
out = Image.new("RGB", (pad + 5 * (TW + pad), len(rows) * (head + TH + pad) + pad), (252, 252, 251))
d = ImageDraw.Draw(out)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
except OSError:
    font = small = ImageFont.load_default()
y = pad
for title, ts in rows:
    d.text((pad, y + 4), title, fill=(11, 11, 11), font=font)
    y += head
    for j, (step, t) in enumerate(ts):
        x = pad + j * (TW + pad)
        out.paste(t, (x, y))
        d.rectangle((x + 4, y + 4, x + 78, y + 22), fill=(252, 252, 251))
        d.text((x + 8, y + 6), f"step {step}", fill=(11, 11, 11), font=small)
    y += TH + pad
for folder in (STUDY / "results/figures", STUDY / "blog/figures"):
    out.save(folder / "opus_mustard_slip.jpg", quality=88, optimize=True)
print("saved", out.size)
