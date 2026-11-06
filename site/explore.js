/* Results explorer: a forest of paired differences (one row per comparison) and the same data as a table,
   both drawn from data/comparisons.json. Colour: the accent marks the distilled edit, greys mark the rest. */
(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var DATA_URL = 'data/comparisons.json';

  // Failure types in the article's order and words; unknown levels are appended as they are.
  var LEVELS = [
    ['Language binding', 'Misread instruction: cube then banana'],
    ['Binding edit on other tasks', 'The same edit on other tasks'],
    ['Perception: wrong target object', 'Perception: lookalike object'],
    ['Motor: missing competence', 'Missing skill: flat box'],
    ['Motor: grasp physics', 'Contact physics: mustard'],
    ['Gripper commitment (all tasks)', 'Gripper threshold'],
    ['Unchanged weights (noise controls)', 'Unchanged weights, rerun']
  ];
  var TYPE_NAME = {
    weight: 'weight edit, practice seeds',
    heldout: 'weight edit, held-out seeds',
    prompt: 'reworded instruction, no weight edit',
    runtime: 'runtime change, no weight edit',
    identity: 'unchanged weights, rerun'
  };

  var fLevel = document.getElementById('f-level');
  var fEditor = document.getElementById('f-editor');
  var fSeeds = document.getElementById('f-seeds');
  var svg = document.getElementById('chart');
  var box = document.getElementById('chart-box');
  var tbody = document.querySelector('#tbl tbody');
  var countEl = document.getElementById('count');
  var ALL = [];
  var DOMAIN = [-25, 75];
  var levelName = {};
  var levelOrder = {};
  LEVELS.forEach(function (l, i) { levelName[l[0]] = l[1]; levelOrder[l[0]] = i; });

  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(parent, x, y, s, attrs) {
    var t = el('text', Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function signed(v, digits) {
    var r = Number(v.toFixed(digits === undefined ? 1 : digits));
    if (r === 0) return '0';
    return (r > 0 ? '+' : '−') + Math.abs(r).toFixed(digits === undefined ? 1 : digits);
  }
  function lname(level) { return levelName[level] || level; }

  // Deterministic PRNG, so intervals do not move between visits.
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // Paired bootstrap: resample n seed pairs from {fixed, broke, unchanged} and take the 2.5/97.5 percentiles.
  function bootCI(n, fixed, broke, seed) {
    if (!n || (fixed === 0 && broke === 0)) return [0, 0];
    var rnd = mulberry32(seed), pf = fixed / n, pb = (fixed + broke) / n, B = 2000, out = new Array(B);
    for (var b = 0; b < B; b++) {
      var d = 0;
      for (var i = 0; i < n; i++) {
        var u = rnd();
        if (u < pf) d++;
        else if (u < pb) d--;
      }
      out[b] = (100 * d) / n;
    }
    out.sort(function (x, y) { return x - y; });
    return [out[Math.floor(0.025 * B)], out[Math.ceil(0.975 * B) - 1]];
  }

  function kind(r) {
    if (r.type === 'identity') return 'noise';
    if (r.type === 'prompt' || r.type === 'runtime') return 'change';
    if (r.type === 'heldout') return 'held';
    if (r.level === 'Language binding' || r.level === 'Binding edit on other tasks') return 'fix';
    if (r.editor === 'GPT-6 Astra') return 'astra';
    return 'claude';
  }
  // The flat-box row pools 10 edits and carries its own display string ("0/16 each vs 0/64").
  function withoutStr(r) {
    if (r.display && r.display.indexOf(' vs ') > 0) return r.display.split(' vs ')[1];
    return r.base + '/' + r.n;
  }
  function withStr(r) {
    if (r.display && r.display.indexOf(' vs ') > 0) return r.display.split(' vs ')[0];
    return r.cand + '/' + r.n;
  }
  function editorName(e) { return e === 'operator' ? 'Operator (no editor)' : e; }
  function seedsStr(r) { return r.n + (r.type === 'heldout' ? ' held-out' : ''); }

  function tipText(r) {
    var lines = [
      signed(r.diff) + ' points · 95% CI ' + signed(r.ci[0]) + ' to ' + signed(r.ci[1]),
      'Without ' + withoutStr(r) + ' · with ' + withStr(r) + ' · ' + r.fixed + ' fixed, ' + r.broke + ' broke',
      r.task + ' · ' + r.label,
      editorName(r.editor) + ' · ' + (TYPE_NAME[r.type] || r.type)
    ];
    if (r.values_changed) lines.push(r.values_changed.toLocaleString('en-US') + ' values changed');
    if (r.note) lines.push(r.note);
    if (r.source) lines.push(r.source);
    return lines.join('|').replace(/\s+/g, ' ');
  }

  function filtered() {
    var lv = fLevel.value, ed = fEditor.value, sd = fSeeds.value;
    return ALL.filter(function (r) {
      if (lv && r.level !== lv) return false;
      if (ed && r.editor !== ed) return false;
      if (sd === 'heldout' && r.type !== 'heldout') return false;
      if (sd === 'practice' && r.type === 'heldout') return false;
      return true;
    });
  }

  // Shorten a text node with an ellipsis until it fits maxW pixels.
  function fit(node, maxW) {
    var s = node.textContent;
    if (maxW <= 0) { node.textContent = ''; return; }
    if (node.getComputedTextLength() <= maxW) return;
    var lo = 0, hi = s.length;
    while (lo < hi) {
      var mid = (lo + hi + 1) >> 1;
      node.textContent = s.slice(0, mid) + '…';
      if (node.getComputedTextLength() <= maxW) lo = mid; else hi = mid - 1;
    }
    node.textContent = lo > 0 ? s.slice(0, lo).replace(/[\s,·:(]+$/, '') + '…' : '';
  }

  function setActive(i) {
    tbody.querySelectorAll('tr').forEach(function (tr) { tr.classList.toggle('active', tr.dataset.i === String(i)); });
    svg.querySelectorAll('.row').forEach(function (g) { g.classList.toggle('active', g.dataset.i === String(i)); });
  }

  function drawMark(g, r, x, y) {
    var k = r.kind;
    if (k === 'held') {
      var s = 11;
      el('rect', { x: x - s / 2, y: y - s / 2, width: s, height: s, rx: 1.5, 'class': 'mark m-held ring', transform: 'rotate(45 ' + x + ' ' + y + ')' }, g);
    } else if (k === 'astra') {
      el('rect', { x: x - 5, y: y - 5, width: 10, height: 10, rx: 2, 'class': 'mark m-astra ring' }, g);
    } else if (k === 'change') {
      el('circle', { cx: x, cy: y, r: 7.5, 'class': 'm-pad' }, g);
      el('circle', { cx: x, cy: y, r: 4.6, 'class': 'mark m-change' }, g);
    } else {
      el('circle', { cx: x, cy: y, r: k === 'fix' ? 6 : 5.5, 'class': 'mark m-' + k + ' ring' }, g);
    }
  }

  function drawChart(rows) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var W = Math.max(280, Math.floor(box.clientWidth));
    var narrow = W < 600;
    var countW = narrow ? 0 : 118;
    var labelW = narrow ? 0 : Math.round(Math.min(360, Math.max(220, W * 0.36)));
    var x0 = narrow ? 8 : labelW + 14;
    var x1 = W - (narrow ? 8 : countW + 10);
    var rowH = narrow ? 64 : 42, groupH = 38, axisH = narrow ? 46 : 26;
    var sx = function (v) { return x0 + ((v - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * (x1 - x0); };
    var ticks = [];
    for (var v = DOMAIN[0]; v <= DOMAIN[1]; v += 25) ticks.push(v);

    var groups = [];
    rows.forEach(function (r) {
      var g = groups[groups.length - 1];
      if (!g || g.level !== r.level) groups.push(g = { level: r.level, rows: [] });
      g.rows.push(r);
    });
    var H = axisH + groups.length * groupH + rows.length * rowH + 10;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.setAttribute('width', W);
    svg.setAttribute('height', H);

    // axis title, tick labels and (on wide screens) full-height grid lines
    var ax = el('g', { 'aria-hidden': 'true' }, svg);
    var tickY = narrow ? 36 : 14;
    if (narrow) txt(ax, 0, 13, 'Change in success, points', { 'class': 't-axis' });
    else txt(ax, labelW, 14, 'Change in success, points', { 'class': 't-axis', 'text-anchor': 'end' });
    ticks.forEach(function (t) {
      if (!narrow) el('line', { x1: sx(t), x2: sx(t), y1: axisH - 4, y2: H - 4, 'class': t === 0 ? 'zero' : 'grid' }, ax);
      if (narrow && t % 50 !== 0) return;
      var anchor = t === DOMAIN[0] ? 'start' : t === DOMAIN[1] ? 'end' : 'middle';
      txt(ax, sx(t), tickY, t === 0 ? '0' : signed(t, 0), { 'class': 't-axis', 'text-anchor': anchor });
    });

    var y = axisH;
    groups.forEach(function (gr) {
      txt(svg, 0, y + 25, lname(gr.level), { 'class': 't-group', 'aria-hidden': 'true' });
      y += groupH;
      gr.rows.forEach(function (r) {
        var g = el('g', {
          'class': 'row', tabindex: '0', role: 'listitem', 'data-i': r.i, 'data-tip': tipText(r),
          'aria-label': r.task + ', ' + r.label + ' (' + editorName(r.editor) + '): ' + withoutStr(r) + ' without, ' + withStr(r) +
            ' with, change ' + signed(r.diff) + ' points, 95% interval ' + signed(r.ci[0]) + ' to ' + signed(r.ci[1])
        }, svg);
        el('rect', { 'class': 'hit', x: 0, y: y + 1, width: W, height: rowH - 2, rx: 6 }, g);
        var cy, counts = withoutStr(r) + ' → ' + withStr(r);
        if (narrow) {
          // task and counts, then the intervention, then the mark; grid lines only behind the mark
          var c = txt(g, W - 8, y + 17, counts, { 'class': 't-count', 'text-anchor': 'end' });
          var t = txt(g, 8, y + 17, r.task, { 'class': 't-task' });
          fit(t, W - 24 - c.getComputedTextLength());
          var l = txt(g, 8, y + 32, r.label, { 'class': 't-label' });
          fit(l, W - 16);
          cy = y + 49;
          ticks.forEach(function (tk) {
            el('line', { x1: sx(tk), x2: sx(tk), y1: cy - 10, y2: cy + 10, 'class': tk === 0 ? 'zero' : 'grid' }, g);
          });
        } else {
          var tt = txt(g, labelW, y + 18, r.task, { 'class': 't-task', 'text-anchor': 'end' });
          fit(tt, labelW - 8);
          var tl = txt(g, labelW, y + 33, r.label, { 'class': 't-label', 'text-anchor': 'end' });
          fit(tl, labelW - 8);
          txt(g, W - 4, y + rowH / 2 + 4, counts, { 'class': 't-count', 'text-anchor': 'end' });
          cy = y + rowH / 2;
        }
        if (r.ci[1] > r.ci[0]) {
          el('line', { x1: sx(r.ci[0]), x2: sx(r.ci[1]), y1: cy, y2: cy, 'class': 'ci ci-' + r.kind }, g);
        }
        drawMark(g, r, sx(r.diff), cy);
        g.addEventListener('mouseenter', function () { setActive(r.i); });
        g.addEventListener('mouseleave', function () { setActive(null); });
        g.addEventListener('focus', function () { setActive(r.i); });
        g.addEventListener('blur', function () { setActive(null); });
        y += rowH;
      });
    });
    if (!rows.length) txt(svg, W / 2, axisH + 28, 'No comparisons match these filters.', { 'class': 't-label', 'text-anchor': 'middle' });
  }

  function td(tr, text, cls) {
    var c = document.createElement('td');
    if (cls) c.className = cls;
    c.textContent = text;
    tr.appendChild(c);
    return c;
  }

  function drawTable(rows) {
    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
    rows.forEach(function (r) {
      var tr = document.createElement('tr');
      tr.dataset.i = r.i;
      td(tr, lname(r.level));
      td(tr, r.task);
      td(tr, r.label);
      td(tr, editorName(r.editor));
      td(tr, seedsStr(r), 'nw');
      td(tr, withoutStr(r));
      td(tr, withStr(r));
      td(tr, r.fixed + ' / ' + r.broke);
      var d = td(tr, signed(r.diff), 'd');
      var ci = document.createElement('span');
      ci.className = 'ci';
      ci.textContent = '[' + signed(r.ci[0]) + ', ' + signed(r.ci[1]) + ']';
      d.appendChild(ci);
      tr.addEventListener('mouseenter', function () { setActive(r.i); });
      tr.addEventListener('mouseleave', function () { setActive(null); });
      tbody.appendChild(tr);
    });
  }

  function render() {
    if (window.wsTip) window.wsTip.hide();
    var rows = filtered();
    drawChart(rows);
    drawTable(rows);
    countEl.textContent = 'Showing ' + rows.length + ' of ' + ALL.length + ' comparisons';
  }

  fetch(DATA_URL)
    .then(function (res) {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    })
    .then(function (data) {
      ALL = data.rows.map(function (r, i) {
        var o = Object.assign({}, r);
        o.i = i;
        o.diff = (100 * (r.cand - r.base)) / r.n;
        o.ci = bootCI(r.n, r.fixed, r.broke, 1000 + i);
        o.kind = kind(r);
        return o;
      });
      ALL.forEach(function (r) {
        if (!(r.level in levelOrder)) { levelOrder[r.level] = LEVELS.length + Object.keys(levelOrder).length; }
      });
      // stable sort by failure type, keeping the file's order within each type
      ALL.sort(function (a, b) { return levelOrder[a.level] - levelOrder[b.level] || a.i - b.i; });

      var lo = 0, hi = 0;
      ALL.forEach(function (r) { lo = Math.min(lo, r.ci[0], r.diff); hi = Math.max(hi, r.ci[1], r.diff); });
      DOMAIN = [Math.floor(lo / 25) * 25, Math.ceil(hi / 25) * 25];

      var seen = {};
      ALL.forEach(function (r) {
        if (!seen['l' + r.level]) { seen['l' + r.level] = 1; fLevel.add(new Option(lname(r.level), r.level)); }
      });
      ['Claude Opus', 'GPT-6 Astra', 'operator'].concat(ALL.map(function (r) { return r.editor; })).forEach(function (e) {
        if (!seen['e' + e] && ALL.some(function (r) { return r.editor === e; })) { seen['e' + e] = 1; fEditor.add(new Option(editorName(e), e)); }
      });
      [fLevel, fEditor, fSeeds].forEach(function (s) { s.addEventListener('change', render); });

      render();
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(render);
      var lastW = box.clientWidth, raf = 0;
      window.addEventListener('resize', function () {
        if (box.clientWidth === lastW) return;
        lastW = box.clientWidth;
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(render);
      });
    })
    .catch(function (err) {
      countEl.textContent = 'Could not load ' + DATA_URL + ' (' + err.message + '). Serve the site over HTTP rather than opening the file directly.';
    });
})();
