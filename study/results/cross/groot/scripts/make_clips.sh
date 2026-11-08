#!/usr/bin/env bash
# Before/after clips of the same start for the page (runs on ali-ms; ffmpeg). Each source video is RoboLab's sensor
# recording of one slot (exterior | wrist, 1280x360, 15 fps). Output: H.264, 2x speed, 960x270, no audio, <= 2 MB.
# Usage: make_clips.sh OUTDIR TASK SLOT LABEL1=ARM1 [LABEL2=ARM2 ...]
#   e.g. make_clips.sh ~/cross/groot/clips RubiksCubeThenBananaTask 30003 unedited=unedited edited=ed-bb
set -eu
OUT="$1"; TASK="$2"; SLOT="$3"; shift 3
mkdir -p "$OUT"
for pair in "$@"; do
  lab="${pair%%=*}"; arm="${pair#*=}"
  src=~/cross/groot/videos/$TASK/$arm/$SLOT.mp4
  dst="$OUT/${TASK}_${SLOT}_${lab}.mp4"
  for crf in 28 32 36 40; do
    ffmpeg -loglevel error -y -i "$src" -an -vf "setpts=0.5*PTS,scale=960:270:flags=area,format=yuv420p" -r 15 \
      -c:v libx264 -preset slow -crf $crf -movflags +faststart "$dst"
    [[ $(stat -c %s "$dst") -le 2000000 ]] && break
  done
  echo "$dst $(stat -c %s "$dst") bytes (crf $crf) from $src"
done
