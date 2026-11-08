#!/usr/bin/env python3
"""Build the weightsmith LinkedIn video.

    python3 src/build.py                 # both formats + poster  (run from anywhere)
    python3 src/build.py --fmt 4x5       # just the 4:5 feed version
    python3 src/build.py --layers-only   # just re-render the text layers (build/<fmt>/*.png) to look at them

Outputs (next to src/):
    weightsmith_linkedin_4x5.mp4    1080x1350, H.264 yuv420p, 30 fps, silent AAC, +faststart
    weightsmith_linkedin_16x9.mp4   1920x1080, same settings
    poster.png                      the drop, from the 4:5 cut (lossless, before encoding)
    check/<fmt>_NN_<time>_<what>.png  frames pulled from each finished video for review

Needs: Google Chrome (CHROME=... to override), ffmpeg/ffprobe on PATH (or FFMPEG=/FFPROBE=), network access to
Google Fonts, Python 3 (Pillow optional, for a sanity check of the layers).

The story (about 35 s): title; the fix (the cube-then-banana clip, before/after captions, the drop slowed down);
21% -> 83% on unseen starts; the surprise (the same edit fixed two tasks it was never built for) and those two clips
with their numbers; the seven blind Claude agents; links.

How it works
    1. Every still layer (cards and captions) is a full frame drawn by src/scenes.html and screenshotted with
       headless Chrome. Layers that sit over footage leave a transparent rounded window.
    2. Each scene is composed losslessly in RGB with ffmpeg: paper, the footage in its window, then the layers, each
       new layer fading in over the previous one (that is how captions change).
    3. The scenes are joined with short cross-fades, converted to BT.709 yuv420p and encoded with a silent AAC track.

The footage is the site's own clips (copied into src/clips): a 2x2 grid, top row unedited, bottom row after the
weight edit; left over-the-shoulder camera, right wrist camera; the 15 Hz simulation at 2x speed, so clip frame n is
control step n (site/data/hero_90002.json: the unedited policy releases the cube at step 76, the edited one drops it
into the bowl at step 119). The clips were encoded with the BT.601 matrix (their teal labels decode to #147d92 that
way), so they are decoded as BT.601 and re-encoded as tagged BT.709.
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

SRC = pathlib.Path(__file__).resolve().parent
OUT = SRC.parent
BUILD = OUT / 'build'
CLIPS = SRC / 'clips'
PAGE = SRC / 'scenes.html'
CHROME = os.environ.get('CHROME', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
FFMPEG = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
FFPROBE = os.environ.get('FFPROBE') or shutil.which('ffprobe')

FPS = 30
XF = 12            # cross-fade between scenes, frames (0.4 s)
PAPER = '0xfaf9f5'

# ---------------------------------------------------------------------------------------------------------------
# Timeline (frames at 30 fps). Clip frame numbers are the source clip's own frames (= control steps).
# ---------------------------------------------------------------------------------------------------------------
TITLE_N = 87       # 2.9 s; the last 0.4 s is the cross-fade into the clip

# The fix: cube_then_banana.mp4, frames 10..277, with the drop (62..95) slowed 3x so it reads on a phone. The edited
# run is done by frame 257 (cube and banana in the bowl); the last frame is held under the cross-fade out.
FIX_START, SLOW_A, SLOW_B, FIX_END, SLOW_K = 10, 62, 96, 278, 3


def fix_out(src):
    """Output frame (from the start of the fix scene) at which clip frame `src` first appears."""
    if src < SLOW_A:
        return src - FIX_START
    if src < SLOW_B:
        return (SLOW_A - FIX_START) + (src - SLOW_A) * SLOW_K
    return (SLOW_A - FIX_START) + (SLOW_B - SLOW_A) * SLOW_K + (src - SLOW_B)


FIX_CLIP_N = fix_out(FIX_END)          # frames of footage
FIX_N = FIX_CLIP_N + XF                # + a held last frame under the cross-fade out
FIX_STATES = [                         # (layer state, first frame, fade-in frames)
    ('before', 0, 0),
    ('before,tag', fix_out(SLOW_A), 9),        # "Slow motion" appears as the clip slows
    ('after', fix_out(SLOW_B), 12),            # the cube is gone: switch caption, drop the tag, back to speed
]

RESULT_N = 108     # 3.6 s
RESULT_STATES = [('a', 0, 0), ('b', 15, 12)]   # the sub-line follows the number after half a second

SURPRISE_N = 81    # 2.7 s: "The surprise: the same edit fixed two tasks it was never built for"

# The two tasks it was never built for: the last 3.5 s of each clip (the edited policy succeeds at the end of each),
# one after the other with a short dissolve; each gets its own task line and number.
TRANSFER_CLIPS = [('eyeglasses.mp4', 287, 392), ('cube_left_of_bowl.mp4', 257, 362)]
TX = 9             # dissolve between the two clips, frames
TRANSFER_CLIP_N = sum(b - a for _, a, b in TRANSFER_CLIPS) - TX
TRANSFER_N = TRANSFER_CLIP_N + XF
_A = TRANSFER_CLIPS[0][2] - TRANSFER_CLIPS[0][1]
TRANSFER_STATES = [('a', 0, 0), ('b', _A - TX, TX)]

AGENTS_N = 210     # 7.0 s: three sentences, revealed one after another
AGENTS_STATES = [('a', 0, 0), ('b', 24, 9), ('c', 54, 9)]

END_N = 108        # 3.6 s

SCENES = ['title', 'fix', 'result', 'surprise', 'transfer', 'end']
LENGTHS = {'title': TITLE_N, 'fix': FIX_N, 'result': RESULT_N, 'surprise': SURPRISE_N, 'transfer': TRANSFER_N,
           'end': END_N}
TOTAL_N = sum(LENGTHS.values()) - XF * (len(SCENES) - 1)

# Frames pulled from each finished video into check/ for review: (scene, frame within the scene, label).
CHECKS = [
    ('title', 0, 'title-first-frame'),
    ('fix', 6, 'transition-midpoint-paper'),
    ('fix', 45, 'before-caption'),
    ('fix', fix_out(80), 'drop-slow-motion'),
    ('fix', fix_out(90), 'drop-clear'),
    ('fix', fix_out(119), 'after-cube-into-bowl'),
    ('fix', fix_out(260), 'after-success'),
    ('result', 60, 'result-card'),
    ('surprise', 40, 'surprise-card'),
    ('transfer', 50, 'eyeglasses-22-44'),
    ('transfer', 160, 'cube-left-of-bowl-8-36'),
    ('end', 60, 'end-card'),
]

# The poster: the drop at its clearest (clip frame 90: the unedited gripper has let the cube go; the edited one
# still holds it), from the lossless fix scene.
POSTER_SRC_FRAME = 90

# ---------------------------------------------------------------------------------------------------------------
# Geometry per format: frame size and the footage window (x, y, w, h). The footage is 960x540.
# ---------------------------------------------------------------------------------------------------------------
FORMATS = {
    '4x5': dict(W=1080, H=1350, video=(60, 436, 960, 540), video_transfer=(60, 404, 960, 540)),
    '16x9': dict(W=1920, H=1080, video=(384, 228, 1152, 648)),
}


def video_rect(fmt, scene):
    f = FORMATS[fmt]
    return f.get(f'video_{scene}', f['video'])
BAR = {'title': False, 'fix': True, 'result': True, 'surprise': True, 'transfer': True, 'agents': True, 'end': False}


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write('\n$ ' + ' '.join(map(str, cmd)) + '\n' + r.stdout[-4000:] + r.stderr[-4000:] + '\n')
        raise SystemExit(f'command failed ({r.returncode}): {cmd[0]}')
    return r


def page_url(fmt, scene, state='', with_video=False):
    q = {'fmt': fmt, 'scene': scene, 'state': state, 'bar': '1' if BAR[scene] else '0'}
    if with_video:
        q.update(dict(zip(['vx', 'vy', 'vw', 'vh'], map(str, video_rect(fmt, scene)))))
    return PAGE.as_uri() + '?' + urllib.parse.urlencode(q)


def chrome(args, url):
    with tempfile.TemporaryDirectory(prefix='ws-chrome-') as prof:
        return run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
                    '--no-default-browser-check', f'--user-data-dir={prof}', '--force-device-scale-factor=1',
                    '--virtual-time-budget=8000', *args, url])


def check_fonts():
    """Abort unless Inter and Source Serif 4 really load from Google Fonts (a fallback font would pass silently)."""
    r = chrome(['--dump-dom', '--window-size=1080,1350'], page_url('4x5', 'title'))
    if 'data-fonts="ok"' not in r.stdout:
        i = r.stdout.find('data-fonts=')
        raise SystemExit('web fonts did not load: ' + (r.stdout[i:i + 160] if i >= 0 else 'no font report') +
                         '\n(the layers need https://fonts.googleapis.com; check the network)')
    print('fonts: Inter and Source Serif 4 loaded from Google Fonts')


def layer_jobs(fmt):
    jobs = [('title', '', False), ('surprise', '', False), ('end', '', False)]
    jobs += [('fix', s, True) for s, _, _ in FIX_STATES]
    jobs += [('result', s, False) for s, _, _ in RESULT_STATES]
    jobs += [('transfer', s, True) for s, _, _ in TRANSFER_STATES]
    jobs += [('agents', s, False) for s, _, _ in AGENTS_STATES]
    return jobs


def layer_path(fmt, scene, state):
    return BUILD / fmt / f"{scene}{'_' + state.replace(',', '+') if state else ''}.png"


def render_layers(fmt):
    f = FORMATS[fmt]
    (BUILD / fmt).mkdir(parents=True, exist_ok=True)

    def one(job):
        scene, state, vid = job
        out = layer_path(fmt, scene, state)
        chrome([f'--screenshot={out}', f"--window-size={f['W']},{f['H']}", '--default-background-color=00000000'],
               page_url(fmt, scene, state, vid))
        return out

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(one, layer_jobs(fmt)))
    try:  # sanity: right size, and not the red "fonts missing" sheet
        from PIL import Image
        for p in outs:
            im = Image.open(p)
            assert im.size == (f['W'], f['H']), (p, im.size)
            assert im.convert('RGBA').getpixel((4, f['H'] - 4))[:3] != (221, 0, 0), f'{p}: fonts missing'
    except ImportError:
        pass
    print(f'{fmt}: rendered {len(outs)} layers')


# ---------------------------------------------------------------------------------------------------------------
# ffmpeg: scenes
# ---------------------------------------------------------------------------------------------------------------
DECODE = 'scale=in_color_matrix=bt601:in_range=tv{size}:flags=lanczos+accurate_rnd+full_chroma_int,format=rgb24'


def layered_scene(fmt, name, n, states, footage=None):
    """paper -> footage (optional filter chain producing [clip]) -> layers fading in one after another."""
    f = FORMATS[fmt]
    W, H = f['W'], f['H']
    inputs, graph = [], []
    clip_inputs, clip_chain = footage or ([], None)
    inputs += clip_inputs
    graph.append(f'color=c={PAPER}:s={W}x{H}:r={FPS},format=rgb24[base]')
    cur = 'base'
    if clip_chain:
        graph.append(clip_chain)   # ends in [clip]
        x, y, _, _ = video_rect(fmt, name)
        graph.append(f'[{cur}][clip]overlay=x={x}:y={y}:format=rgb:eof_action=repeat[v0]')
        cur = 'v0'
    for i, (state, start, fade) in enumerate(states):
        idx = inputs.count('-i')
        inputs += ['-loop', '1', '-framerate', str(FPS), '-i', str(layer_path(fmt, name, state))]
        chain = f'[{idx}:v]format=rgba'
        if fade:
            chain += f',fade=t=in:start_frame={start}:nb_frames={fade}:alpha=1'
        elif start:
            raise ValueError('a later layer needs a fade')
        graph.append(chain + f'[l{i}]')
        graph.append(f'[{cur}][l{i}]overlay=0:0:format=rgb[o{i}]')
        cur = f'o{i}'
    graph.append(f'[{cur}]format=rgb24[out]')
    out = BUILD / fmt / f'scene_{name}.mkv'
    run([FFMPEG, '-hide_banner', '-y', *inputs, '-filter_complex', ';'.join(graph), '-map', '[out]',
         '-frames:v', str(n), '-c:v', 'libx264rgb', '-qp', '0', '-preset', 'ultrafast', '-pix_fmt', 'rgb24',
         str(out)])
    return out


def size_opt(fmt):
    _, _, w, h = FORMATS[fmt]['video']
    return '' if (w, h) == (960, 540) else f':w={w}:h={h}'


def fix_footage(fmt):
    # Retime with an explicit frame map: setpts puts clip frame n at its output frame, fps=30 repeats frames to fill
    # the slowed stretch (each slowed frame is held 3 frames, no interpolation), tpad holds the last frame.
    n0 = SLOW_A - FIX_START
    n1 = SLOW_B - FIX_START
    a = SLOW_A - FIX_START
    b = a + (SLOW_B - SLOW_A) * SLOW_K
    expr = f'if(lt(N,{n0}),N,if(lt(N,{n1}),{a}+(N-{n0})*{SLOW_K},{b}+(N-{n1})))'
    chain = (f"[0:v]trim=start_frame={FIX_START}:end_frame={FIX_END},settb=1/{FPS},setpts='{expr}',"
             f"fps={FPS},{DECODE.format(size=size_opt(fmt))},tpad=stop_mode=clone:stop={XF + 2}[clip]")
    return ['-i', str(CLIPS / 'cube_then_banana.mp4')], chain


def transfer_footage(fmt):
    (n0, a0, b0), (n1, a1, b1) = TRANSFER_CLIPS
    parts = [f'[{i}:v]trim=start_frame={a}:end_frame={b},settb=1/{FPS},setpts=N,fps={FPS},'
             f'{DECODE.format(size=size_opt(fmt))},format=gbrp[t{i}]'
             for i, (_, a, b) in enumerate(TRANSFER_CLIPS)]
    chain = ';'.join(parts) + (f';[t0][t1]xfade=transition=fade:duration={TX / FPS:.6f}:offset={(b0 - a0 - TX) / FPS:.6f},'
                               f'format=rgb24,tpad=stop_mode=clone:stop={XF + 2}[clip]')
    return ['-i', str(CLIPS / n0), '-i', str(CLIPS / n1)], chain


def build_scenes(fmt):
    paths = {
        'title': layered_scene(fmt, 'title', TITLE_N, [('', 0, 0)]),
        'fix': layered_scene(fmt, 'fix', FIX_N, FIX_STATES, fix_footage(fmt)),
        'result': layered_scene(fmt, 'result', RESULT_N, RESULT_STATES),
        'surprise': layered_scene(fmt, 'surprise', SURPRISE_N, [('', 0, 0)]),
        'transfer': layered_scene(fmt, 'transfer', TRANSFER_N, TRANSFER_STATES, transfer_footage(fmt)),
        'end': layered_scene(fmt, 'end', END_N, [('', 0, 0)]),
    }
    for k, p in paths.items():
        got = int(run([FFPROBE, '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                       'stream=nb_read_frames', '-of', 'csv=p=0', str(p)]).stdout.strip())
        assert got == LENGTHS[k], f'{fmt} {k}: {got} frames, expected {LENGTHS[k]}'
    print(f'{fmt}: scenes ' + ', '.join(f'{k} {LENGTHS[k] / FPS:.2f}s' for k in SCENES))
    return paths


# ---------------------------------------------------------------------------------------------------------------
# ffmpeg: final encode
# ---------------------------------------------------------------------------------------------------------------
# Scene change = a dip through the paper colour, not a cross-dissolve (no double exposure of text over footage):
# the outgoing scene eases out to paper in the first half and the new one eases in over the second half. Inside the
# running header's band, pixels identical in both scenes stay put, so the header does not blink between scenes.
# (Only there: elsewhere two different texts can share identical ink pixels, which must still fade.)
# xfade custom expr: P runs 1 -> 0; A/B are the outgoing/incoming samples; gbrp planes are G, B, R.
_PAPER_PLANE = 'if(eq(PLANE,0),249,if(eq(PLANE,1),245,250))'    # #faf9f5 as G, B, R
HEADER_BAND = {'4x5': 104, '16x9': 88}                          # = --barH in scenes.html


def dip_expr(fmt):
    return (f'if(lt(Y,{HEADER_BAND[fmt]})*eq(a0(X,Y),b0(X,Y))*eq(a1(X,Y),b1(X,Y))*eq(a2(X,Y),b2(X,Y)),A,'
            f'st(0,{_PAPER_PLANE});st(1,if(gt(P,0.5),2*P-1,1-2*P));st(1,ld(1)*ld(1)*(3-2*ld(1)));'
            'ld(0)+(if(gt(P,0.5),A,B)-ld(0))*ld(1))')


def assemble(fmt, paths):
    out = OUT / f'weightsmith_linkedin_{fmt}.mp4'
    inputs, graph = [], []
    for i, k in enumerate(SCENES):
        inputs += ['-i', str(paths[k])]
        graph.append(f'[{i}:v]settb=1/{FPS},setpts=N,fps={FPS},format=gbrp[s{i}]')   # xfade wants a known CFR
    cur, t = 's0', LENGTHS[SCENES[0]]
    for i in range(1, len(SCENES)):
        off = (t - XF) / FPS
        graph.append(f"[{cur}][s{i}]xfade=transition=custom:duration={XF / FPS:.6f}:offset={off:.6f}:"
                     f"expr='{dip_expr(fmt)}'[x{i}]")
        cur, t = f'x{i}', t - XF + LENGTHS[SCENES[i]]
    assert t == TOTAL_N
    graph.append(f'[{cur}]scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,'
                 f'format=yuv420p,setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709[v]')
    dur = TOTAL_N / FPS
    run([FFMPEG, '-hide_banner', '-y', *inputs,
         '-f', 'lavfi', '-t', f'{dur:.6f}', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
         '-filter_complex', ';'.join(graph), '-map', '[v]', '-map', f'{len(SCENES)}:a',
         '-frames:v', str(TOTAL_N), '-r', str(FPS),
         '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-profile:v', 'high', '-level:v', '4.0',
         '-pix_fmt', 'yuv420p', '-g', str(2 * FPS), '-bf', '2',
         '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
         '-c:a', 'aac', '-b:a', '96k', '-ar', '48000', '-ac', '2', '-shortest',
         '-movflags', '+faststart', str(out)])
    return out


def scene_starts():
    starts, t = {}, 0
    for k in SCENES:
        starts[k] = t
        t += LENGTHS[k] - XF
    return starts


def extract_checks(fmt, mp4):
    starts = scene_starts()
    pts = sorted((starts[s] + i, label) for s, i, label in CHECKS)
    d = OUT / 'check'
    d.mkdir(exist_ok=True)
    for old in d.glob(f'{fmt}_*.png'):
        old.unlink()
    sel = '+'.join(f'eq(n\\,{n})' for n, _ in pts)
    with tempfile.TemporaryDirectory() as tmp:
        run([FFMPEG, '-hide_banner', '-y', '-i', str(mp4), '-vf', f'select={sel}', '-fps_mode', 'passthrough',
             f'{tmp}/%02d.png'])
        names = []
        for k, (n, label) in enumerate(pts, 1):
            dst = d / f'{fmt}_{k:02d}_{n / FPS:05.2f}s_{label}.png'
            shutil.move(f'{tmp}/{k:02d}.png', dst)
            names.append(dst)
    return names


def poster(paths):
    frame = fix_out(POSTER_SRC_FRAME)
    out = OUT / 'poster.png'
    run([FFMPEG, '-hide_banner', '-y', '-i', str(paths['fix']), '-vf', f'select=eq(n\\,{frame})',
         '-frames:v', '1', '-fps_mode', 'passthrough', str(out)])
    return out


def describe(p):
    r = json.loads(run([FFPROBE, '-v', 'error', '-show_entries',
                        'format=duration,size:stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames',
                        '-of', 'json', str(p)]).stdout)
    v = [s for s in r['streams'] if s['codec_name'] == 'h264'][0]
    a = [s['codec_name'] for s in r['streams'] if s['codec_name'] != 'h264']
    mb = int(r['format']['size']) / 1e6
    return (f"{p.name}: {v['width']}x{v['height']} {v['codec_name']} {v['pix_fmt']} {v['avg_frame_rate']} fps, "
            f"{float(r['format']['duration']):.2f} s, audio {a}, {mb:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fmt', nargs='+', choices=list(FORMATS), default=list(FORMATS))
    ap.add_argument('--layers-only', action='store_true', help='render the Chrome layers and stop')
    ap.add_argument('--keep', action='store_true', help='keep the lossless scene files in build/ (~0.5 GB)')
    args = ap.parse_args()
    for tool, path in [('Chrome', CHROME), ('ffmpeg', FFMPEG), ('ffprobe', FFPROBE)]:
        if not path or not pathlib.Path(path).exists():
            raise SystemExit(f'{tool} not found (set {tool.upper()}=/path/to/{tool.lower()})')
    print(f'timeline: {TOTAL_N} frames = {TOTAL_N / FPS:.2f} s; '
          f'fix scene {FIX_N / FPS:.2f} s, drop slowed {SLOW_K}x over clip frames {SLOW_A}-{SLOW_B - 1}')
    check_fonts()
    for fmt in args.fmt:
        render_layers(fmt)
        if args.layers_only:
            continue
        paths = build_scenes(fmt)
        out = assemble(fmt, paths)
        print(describe(out))
        print(f'{fmt}: check frames in', extract_checks(fmt, out)[0].parent)
        if fmt == '4x5':
            print('poster:', poster(paths))
        if not args.keep:
            for p in paths.values():
                p.unlink()


if __name__ == '__main__':
    main()
