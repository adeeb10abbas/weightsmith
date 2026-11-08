#!/usr/bin/env bash
# Before/after clips of one start for the page: the unedited and the edited policy on the same slot, stacked
# (unedited on top), each row = left shoulder | right shoulder | wrist camera, with a label burned in; H.264, 2x speed,
# capped at 2 MB. Runs on the GPU host (ffmpeg with libx264), from WS.
#   scripts/make_clips.sh TASK_CLASS SLOT BEFORE_ARM AFTER_ARM OUT.mp4 "before label" "after label"
# e.g. scripts/make_clips.sh MustardRightOfRaisinRFTask 96003 base Et8und clips/t8_96003.mp4 "unedited" "edited"
set -euo pipefail
TASK=$1; SLOT=$2; A=$3; B=$4; OUT=$5; LA=${6:-$3}; LB=${7:-$4}
VA=videos/$TASK/$A/$SLOT.mp4; VB=videos/$TASK/$B/$SLOT.mp4
[ -f "$VA" ] && [ -f "$VB" ] || { echo "missing $VA or $VB"; exit 1; }
mkdir -p "$(dirname "$OUT")"
FONT=$(fc-match -f '%{file}' 'DejaVu Sans:bold' 2>/dev/null || echo /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf)
lab() { echo "drawtext=fontfile=$FONT:text='$1':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=4"; }
for crf in 26 30 34 38; do
  ffmpeg -v error -y -i "$VA" -i "$VB" -filter_complex \
    "[0:v]setpts=PTS/2,$(lab "$LA")[a];[1:v]setpts=PTS/2,$(lab "$LB")[b];[a][b]vstack=inputs=2:shortest=0,format=yuv420p[v]" \
    -map "[v]" -r 15 -c:v libx264 -preset slow -crf $crf -movflags +faststart -an "$OUT"
  [ "$(stat -c %s "$OUT")" -le 2000000 ] && break
done
echo "$OUT $(stat -c %s "$OUT") bytes, crf $crf"
