#!/usr/bin/env python3
"""Replace the labels burned into the two world-action model clips with plain ones.

The clips were rendered with working labels ("unedited, RF wording (reference first)", "edited t8und ..."). This
script covers each old label with a solid chip and writes a new label on it, on the video and on its poster image.
It always starts from the clips as they were at SOURCE_REV, so running it twice gives the same files.

    python3 site/scripts/post/relabel_clips.py

Needs ffmpeg, Pillow and matplotlib (for its copy of DejaVu Sans Bold, the font of the other clips' labels).
"""
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
VIDEO = REPO / "site" / "cosmos" / "video"
SOURCE_REV = "8d020f4"   # the last commit whose clips still carry the working labels
FONT = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                                       "DejaVuSans-Bold.ttf"), 15)
DARK, TEAL = (26, 25, 21, 255), (20, 125, 146, 255)
ROW_H = 180   # each clip is two 960 x 180 rows

# clip -> one entry per row: the new label, the chip colour, and the right edge of the old label's box in pixels
CLIPS = {
    "cosmos_t8_wording": [("Unedited model, the original wording", DARK, 356),
                          ("Unedited model, the clear wording", DARK, 326)],
    "cosmos_before_after": [("Unedited model, original wording, a held-out run", DARK, 419),
                            ("Edited model, the same wording and the same run", TEAL, 416)],
}


def original(name, suffix, to):
    data = subprocess.check_output(["git", "show", f"{SOURCE_REV}:site/cosmos/video/{name}{suffix}"], cwd=REPO)
    Path(to).write_bytes(data)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for name, rows in CLIPS.items():
            chips = Image.new("RGBA", (960, ROW_H * len(rows)), (0, 0, 0, 0))
            draw = ImageDraw.Draw(chips)
            for r, (text, colour, old_right) in enumerate(rows):
                y = ROW_H * r
                right = max(old_right + 4, 10 + draw.textlength(text, font=FONT) + 10)   # hide all of the old box
                draw.rounded_rectangle((2, y + 2, right, y + 30), radius=4, fill=colour)
                draw.text((10, y + 16), text, font=FONT, fill=(255, 255, 255, 255), anchor="lm")
            src, png, jpg = f"{tmp}/{name}.mp4", f"{tmp}/{name}.png", f"{tmp}/{name}.jpg"
            original(name, ".mp4", src)
            original(name, ".jpg", jpg)
            chips.save(png)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", png, "-filter_complex",
                            "[0:v][1:v]overlay=0:0,format=yuv420p", "-c:v", "libx264", "-crf", "20", "-preset", "slow",
                            "-movflags", "+faststart", "-an", str(VIDEO / f"{name}.mp4")], check=True)
            Image.alpha_composite(Image.open(jpg).convert("RGBA"), chips).convert("RGB").save(
                VIDEO / f"{name}.jpg", quality=88)
            print(f"relabelled {name}")


if __name__ == "__main__":
    main()
