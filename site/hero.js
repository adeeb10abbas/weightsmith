/* The opening video's timeline: the commanded gripper of both policies on held-out start 90002 (filled where the
   gripper is commanded closed), with a playhead synced to the video. Click or drag to scrub; arrow keys step one
   control step (Shift: one 15-step action chunk); Space plays or pauses. The clip shows the 15 Hz simulation at 2x
   speed at 30 fps from step 0, so frame n, shown from n/30 s to (n+1)/30 s, is control step n (checked against the
   logged events: the cube leaves the unedited gripper at frames 76-77 and drops into the bowl in the edited run at
   frame 119). Data: data/hero_90002.json, written by scripts/make_hero_timeline.py from the paper's Fig. 1 source. */
(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var video = document.getElementById('hero-video');
  var root = document.getElementById('scrub');
  var svg = document.getElementById('scrub-svg');
  var readout = document.getElementById('scrub-readout');
  var hint = document.querySelector('.scrub-hint');
  var slow = document.getElementById('scrub-slow');
  if (!video || !root || !svg) return;

  var D = null;          // timeline data
  var steps = 288;       // control steps in the clip (one per frame)
  var fps = 30;
  var geom = null;       // current drawing geometry
  var shown = -1;        // step currently drawn
  var head, dots = [], labels = [];
  var stopAt = null;     // slow replay: pause when this step is reached

  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(parent, x, y, s, cls, anchor) {
    var t = el('text', { x: x, y: y, 'class': cls, 'text-anchor': anchor || 'start' }, parent);
    t.textContent = s;
    return t;
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function phase(arm, s) {
    var p = arm.phases, out = p[0][1];
    for (var i = 0; i < p.length; i++) if (p[i][0] <= s) out = p[i][1];
    return out;
  }

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    labels = [];
    var W = Math.max(280, Math.round(root.clientWidth));
    var narrow = W < 560;
    var lw = narrow ? 64 : 100;
    var x0 = lw, x1 = W - 2;
    var H = 96;
    var sx = function (s) { return x0 + (s / steps) * (x1 - x0); };
    geom = { W: W, H: H, x0: x0, x1: x1, sx: sx };
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.setAttribute('height', H);
    // layers: tracks and marks, then the playhead, then text; labels sit above and below the playhead's reach
    var gTrack = el('g', {}, svg), gHead = el('g', {}, svg), gText = el('g', {}, svg);

    var rows = [
      { key: 'unedited', name: 'Unedited', y: 32, cls: 'u' },
      { key: 'edited', name: narrow ? 'Edited' : 'After the edit', y: 64, cls: 'e' }
    ];
    rows.forEach(function (row) {
      var arm = D.arms[row.key];
      txt(gText, 0, row.y + 4, row.name, 'tl-name');
      el('rect', { x: x0, y: row.y - 5, width: x1 - x0, height: 10, rx: 2, 'class': 'tl-track' }, gTrack);
      if (arm.last_step + 1 < steps) {  // the episode's log ends inside the clip: leave that stretch empty
        el('rect', { x: sx(arm.last_step + 1), y: row.y - 5, width: x1 - sx(arm.last_step + 1), height: 10, rx: 2, 'class': 'tl-over' }, gTrack);
      }
      arm.closed.forEach(function (sp) {
        var a = clamp(sp[0], 0, steps), b = clamp(sp[1], 0, steps);
        if (b > a) el('rect', { x: sx(a), y: row.y - 5, width: Math.max(1.5, sx(b) - sx(a)), height: 10, rx: 1.5, 'class': 'tl-closed ' + row.cls }, gTrack);
      });
      arm.marks.forEach(function (m) {
        var x = sx(m.step);
        if (m.kind === 'release') {
          el('path', { d: 'M' + (x - 4) + ' ' + (row.y - 4) + 'l8 8M' + (x + 4) + ' ' + (row.y - 4) + 'l-8 8', 'class': 'tl-x' }, gTrack);
          var t = txt(gText, x + 9, row.y - 13, m.label, 'tl-label');
          if (t.getComputedTextLength() + x + 9 > W) { t.setAttribute('x', W); t.setAttribute('text-anchor', 'end'); }
          labels.push({ step: m.step, node: t });
        } else {
          el('circle', { cx: x, cy: row.y, r: 4.2, 'class': 'tl-ring' }, gTrack);
          var lab = txt(gText, x, row.y + 24, m.label, 'tl-label', 'middle');
          var len = lab.getComputedTextLength();
          if (x + len / 2 > W) { lab.setAttribute('x', W); lab.setAttribute('text-anchor', 'end'); }
          labels.push({ step: m.step, node: lab });
        }
      });
    });
    head = el('line', { x1: x0, x2: x0, y1: rows[0].y - 9, y2: rows[1].y + 9, 'class': 'tl-head' }, gHead);
    dots = rows.map(function (row) { return el('circle', { cx: x0, cy: row.y, r: 3.6, 'class': 'tl-dot' }, gHead); });
    shown = -1;
    update(currentStep(), true);
  }

  // the frame on screen at playback position t (frame n covers [n/fps, (n+1)/fps))
  function currentStep() { return clamp(Math.floor(video.currentTime * fps + 1e-3), 0, steps - 1); }

  function update(s, force) {
    if (!geom || (!force && s === shown)) return;
    shown = s;
    var x = geom.sx(s + 0.5);
    head.setAttribute('x1', x); head.setAttribute('x2', x);
    dots.forEach(function (d) { d.setAttribute('cx', x); });
    labels.forEach(function (l) { l.node.classList.toggle('on', s >= l.step); });
    var u = phase(D.arms.unedited, s), e = phase(D.arms.edited, s);
    root.setAttribute('aria-valuenow', s);
    root.setAttribute('aria-valuetext', 'Step ' + s + '. Unedited: ' + u + '. After the edit: ' + e + '.');
    if (readout) {
      readout.children[0].textContent = 'Step ' + s;
      readout.children[1].textContent = 'Unedited: ' + u;
      readout.children[2].textContent = 'After the edit: ' + e;
    }
    if (stopAt !== null && s >= stopAt) endSlow(true);
  }

  // keep the playhead on the frame being shown: per presented frame where the browser reports frames (mediaTime is
  // the frame's own timestamp, n/fps), else on every animation frame while playing
  var perFrame = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  if (perFrame) {
    var onFrame = function (now, meta) {
      if (D) update(clamp(Math.round(meta.mediaTime * fps), 0, steps - 1));
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  } else {
    var tick = function () {
      if (D) update(currentStep());
      if (!video.paused) requestAnimationFrame(tick);
    };
    video.addEventListener('play', function () { requestAnimationFrame(tick); });
  }
  ['seeked', 'loadedmetadata', 'pause'].forEach(function (ev) {
    video.addEventListener(ev, function () { if (D) update(currentStep()); });
  });
  video.addEventListener('timeupdate', function () { if (D && (!perFrame || video.paused)) update(currentStep()); });

  function seek(s) {
    s = clamp(s, 0, steps - 1);
    video.currentTime = (s + 0.5) / fps;  // the middle of frame s
    update(s);
  }
  function toggle() {
    if (video.paused) { var p = video.play(); if (p && p.catch) p.catch(function () {}); } else video.pause();
  }

  // pointer: click or drag anywhere on the timeline
  var dragging = false, wasPlaying = false;
  function stepAt(clientX) {
    var r = svg.getBoundingClientRect();
    var x = (clientX - r.left) * (geom.W / r.width);
    return Math.round(((x - geom.x0) / (geom.x1 - geom.x0)) * steps - 0.5);
  }
  root.addEventListener('pointerdown', function (e) {
    if (!geom || e.button > 0) return;
    endSlow(false);
    dragging = true;
    wasPlaying = !video.paused;
    video.pause();
    root.setPointerCapture(e.pointerId);
    root.classList.add('dragging');
    seek(stepAt(e.clientX));  // the browser's own focus on mousedown keeps the arrow keys working afterwards
  });
  root.addEventListener('pointermove', function (e) { if (dragging) seek(stepAt(e.clientX)); });
  function release() {
    if (!dragging) return;
    dragging = false;
    root.classList.remove('dragging');
    if (wasPlaying) toggle();
  }
  root.addEventListener('pointerup', release);
  root.addEventListener('pointercancel', release);

  // keyboard: a slider over control steps
  root.addEventListener('keydown', function (e) {
    var s = shown < 0 ? currentStep() : shown, d = e.shiftKey ? 15 : 1;
    switch (e.key) {
      case 'ArrowRight': case 'ArrowUp': video.pause(); seek(s + d); break;
      case 'ArrowLeft': case 'ArrowDown': video.pause(); seek(s - d); break;
      case 'PageUp': video.pause(); seek(s + 15); break;
      case 'PageDown': video.pause(); seek(s - 15); break;
      case 'Home': video.pause(); seek(0); break;
      case 'End': video.pause(); seek(steps - 1); break;
      case ' ': case 'Enter': case 'k': endSlow(false); toggle(); break;
      default: return;
    }
    e.preventDefault();
  });

  // the release, slowed down: steps 60 to 96 at quarter speed
  function endSlow(pause) {
    if (stopAt === null) return;
    stopAt = null;
    if (pause) video.pause();
    video.playbackRate = 1;
    if (slow) slow.setAttribute('aria-pressed', 'false');
  }
  if (slow) slow.addEventListener('click', function () {
    if (!D) return;
    video.pause();
    seek(60);
    video.playbackRate = 0.25;
    stopAt = 96;
    slow.setAttribute('aria-pressed', 'true');
    var p = video.play();
    if (p && p.catch) p.catch(function () { endSlow(false); });
  });

  fetch(root.getAttribute('data-src'))
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function (data) {
      D = data;
      fps = data.fps;
      steps = data.clip_steps;
      root.setAttribute('aria-valuemax', steps - 1);
      root.hidden = false;
      if (hint) hint.hidden = false;
      if (readout) readout.hidden = false;
      draw();
      var w = root.clientWidth;
      window.addEventListener('resize', function () {
        if (root.clientWidth !== w) { w = root.clientWidth; draw(); }
      });
    })
    .catch(function () { /* the video and its caption stand on their own */ });
})();
