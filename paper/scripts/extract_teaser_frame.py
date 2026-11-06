#!/usr/bin/env python3
"""Recreate paper/figures/src/ctb_heldout90000_wrist.png, the image used in Table 1.

Source: the side-by-side clip of held-out slot 90000 (cube then banana), committed on the site/v1 branch as
site/assets/video/cube_then_banana.mp4 (sha256 e7c537d8eeb6b3f498c53ed63910bc7c44493192d616f68cc4b355ee60950291).
The clip is 960x540: the top half is the unedited policy, the bottom half the edited policy; in each half the left
480 px are the exterior camera and the right 480 px the wrist camera. We take frame 210 (t = 7.0 s at 30 fps) and keep
the top 225 rows of each half's wrist view, which avoids the caption burned into the bottom of the clip.

Usage:
  git show site/v1:site/assets/video/cube_then_banana.mp4 > /tmp/cube_then_banana.mp4
  python3 paper/scripts/extract_teaser_frame.py /tmp/cube_then_banana.mp4
Requires opencv-python and numpy.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures/src/ctb_heldout90000_wrist.png"
FRAME = 210


def main(src):
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, fr = cap.read()
    assert ok, "could not read frame"
    assert fr.shape[:2] == (540, 960), fr.shape
    unedited = fr[0:225, 480:960]
    edited = fr[270:495, 480:960]
    out = np.vstack([unedited, np.full((6, 480, 3), 255, np.uint8), edited])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), out, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print("wrote", OUT, out.shape)


if __name__ == "__main__":
    main(sys.argv[1])
