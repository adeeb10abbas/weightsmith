/* The cross-model draft page.
   1. Renders the TL;DR and the comparison tables from metrics.json. The page ships the same tables pre-rendered (no-JS fallback,
      written by site/scripts/make_cosmos_metrics.py), so this only matters when metrics.json is newer than the HTML.
   2. Clips: a placeholder becomes a player as soon as its MP4 exists in video/, and players run only while in view,
      never for readers who ask for reduced motion.
   3. Contents: the sticky list marks the section being read; the folded list closes after a pick.
   4. Citation: a copy button for the BibTeX. */
(function () {
  'use strict';

  var REPO = 'https://github.com/adeeb10abbas/weightsmith';

  function el(tag, cls) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }

  // text with `code` spans, as the generator writes them
  function inline(parent, s) {
    String(s || '').split('`').forEach(function (part, i) {
      if (!part) return;
      if (i % 2) {
        var c = el('code');
        c.textContent = part;
        parent.appendChild(c);
      } else {
        parent.appendChild(document.createTextNode(part));
      }
    });
    return parent;
  }

  /* ---------- tables ---------- */
  function table(t, models, order) {
    var tb = el('table', 'ctable');
    var head = el('thead');
    var hr = el('tr');
    var h0 = el('th');
    h0.scope = 'col';
    var vh = el('span', 'vh');
    vh.textContent = 'Row';
    h0.appendChild(vh);
    hr.appendChild(h0);
    order.forEach(function (id) {
      var th = el('th');
      th.scope = 'col';
      var mn = el('span', 'mn');
      mn.textContent = models[id].name;
      var mk = el('span', 'mk');
      mk.textContent = models[id].short;
      th.appendChild(mn);
      th.appendChild(mk);
      hr.appendChild(th);
    });
    head.appendChild(hr);
    tb.appendChild(head);
    var body = el('tbody');
    t.rows.forEach(function (r) {
      var tr = el('tr', r.sub ? 'sub' : '');
      var th = inline(el('th'), r.label);
      th.scope = 'row';
      tr.appendChild(th);
      order.forEach(function (id) {
        var c = r.cells[id] || { v: '—', s: 'na' };
        var td = el('td', 'c' + (c.s === 'running' ? ' st-run' : c.s === 'na' ? ' st-na' : '') + (c.prov ? ' prov' : ''));
        td.setAttribute('data-model', models[id].name);
        var v = inline(el('span', 'v'), c.v);
        if (c.prov) {
          var tag = el('span', 'tag');
          tag.textContent = 'earlier slots';
          v.appendChild(document.createTextNode(' '));
          v.appendChild(tag);
        }
        if (c.prac) {
          var ptag = el('span', 'tag');
          ptag.textContent = 'practice starts';
          v.appendChild(document.createTextNode(' '));
          v.appendChild(ptag);
        }
        td.appendChild(v);
        if (c.d) td.appendChild(inline(el('span', 'd'), c.d));
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    tb.appendChild(body);
    return tb;
  }

  function sources(list) {
    var p = el('p', 'src');
    p.appendChild(document.createTextNode('Records: '));
    list.forEach(function (s, i) {
      if (i) p.appendChild(document.createTextNode(', '));
      var a = el('a');
      var at = s.lastIndexOf(':');
      var branch = at > 0 ? s.slice(0, at) : 'main';
      var path = at > 0 ? s.slice(at + 1) : s;
      a.href = REPO + (/\.[a-z]+$/i.test(path) ? '/blob/' : '/tree/') + branch + '/' + path;
      var c = el('code');
      c.textContent = path.replace('study/results/', '');
      a.appendChild(c);
      p.appendChild(a);
    });
    return p;
  }

  function render(m) {
    var models = {};
    var order = [];
    m.models.forEach(function (x) {
      models[x.id] = x;
      order.push(x.id);
    });
    var byId = { summary: m.summary };
    m.tables.forEach(function (t) {
      byId[t.id] = t;
    });
    Array.prototype.forEach.call(document.querySelectorAll('[data-table]'), function (b) {
      var t = byId[b.getAttribute('data-table')];
      if (!t) return;
      var frag = document.createDocumentFragment();
      frag.appendChild(table(t, models, order));
      if (t.sources && t.sources.length) frag.appendChild(sources(t.sources));
      if (t.note) frag.appendChild(inline(el('p', 'tnote'), t.note));
      b.textContent = '';
      b.appendChild(frag);
    });
    // TL;DR: bullets read from the tables by the generator; "{running}" marks what is not measured yet
    var tl = document.querySelector('[data-tldr]');
    if (tl && m.tldr && m.tldr.items) {
      tl.textContent = '';
      m.tldr.items.forEach(function (it) {
        var li = el('li');
        li.setAttribute('data-id', it.id);
        var lead = el('span', 'tl');
        var b = el('b');
        b.textContent = it.lead;
        lead.appendChild(b);
        if (it.kind) {
          var k = el('span', 'tk');
          k.textContent = it.kind;
          lead.appendChild(document.createTextNode(' '));
          lead.appendChild(k);
        }
        if (it.tag) {
          var tg = el('span', 'tag');
          tg.textContent = it.tag;
          lead.appendChild(document.createTextNode(' '));
          lead.appendChild(tg);
        }
        li.appendChild(lead);
        li.appendChild(document.createTextNode(' '));
        String(it.text).split('{running}').forEach(function (part, i) {
          if (i) {
            var r = el('span', 'run');
            r.textContent = 'running';
            li.appendChild(r);
          }
          if (part) li.appendChild(document.createTextNode(part));
        });
        tl.appendChild(li);
      });
      var tn = document.querySelector('[data-tldr-note]');
      if (tn && m.tldr.note) tn.textContent = m.tldr.note;
    }
    var st = document.getElementById('stamp');
    if (st && m.status_text) st.textContent = m.status_text + ' · updated ' + m.updated_text;
  }

  if (window.fetch && document.querySelector('[data-table]')) {
    fetch('metrics.json', { cache: 'no-cache' })
      .then(function (r) {
        if (!r.ok) throw new Error('metrics.json ' + r.status);
        return r.json();
      })
      .then(render)
      .catch(function () {
        /* keep the pre-rendered tables */
      });
  }

  /* ---------- clips ---------- */
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var io = !reduce && 'IntersectionObserver' in window
    ? new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            var p = e.target.play();
            if (p && p.catch) p.catch(function () {});
          } else {
            e.target.pause();
          }
        });
      }, { threshold: 0.5 })
    : null;

  function watch(v) {
    if (io) io.observe(v);
  }

  function exists(url, kind) {
    return fetch(url, { method: 'HEAD', cache: 'no-cache' }).then(function (r) {
      return r.ok && (r.headers.get('content-type') || '').indexOf(kind) === 0;
    });
  }

  function upgrade(fig) {
    var src = fig.getAttribute('data-clip');
    var v = el('video');
    v.controls = true;
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.preload = 'metadata';
    v.setAttribute('muted', '');
    v.setAttribute('playsinline', '');
    v.setAttribute('aria-label', fig.getAttribute('data-title') + '. ' + fig.getAttribute('data-caption'));
    var s = el('source');
    s.src = src;
    s.type = 'video/mp4';
    v.appendChild(s);
    var poster = src.replace(/\.mp4$/, '.jpg');
    exists(poster, 'image').then(function (ok) {
      if (ok) v.poster = poster;
    }).catch(function () {});
    fig.replaceChild(v, fig.querySelector('.ph'));
    var cap = fig.querySelector('figcaption');
    if (cap) {
      cap.textContent = '';
      var b = el('b');
      b.textContent = fig.getAttribute('data-title') + '.';
      cap.appendChild(b);
      cap.appendChild(document.createTextNode(' ' + fig.getAttribute('data-caption')));
    }
    fig.classList.remove('clip-pending');
    watch(v);
  }

  Array.prototype.forEach.call(document.querySelectorAll('.clip video'), watch);
  if (window.fetch) {
    Array.prototype.forEach.call(document.querySelectorAll('.clip-pending[data-clip]'), function (fig) {
      exists(fig.getAttribute('data-clip'), 'video').then(function (ok) {
        if (ok) upgrade(fig);
      }).catch(function () {});
    });
  }

  /* ---------- contents: mark the section being read ---------- */
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc-side a[href^="#"]'));
  var marks = tocLinks.map(function (a) {
    return { a: a, t: document.getElementById(a.getAttribute('href').slice(1)) };
  }).filter(function (m) { return m.t; });
  var current = null;
  var queued = false;
  function spy() {
    queued = false;
    var line = Math.min(160, window.innerHeight * 0.3);
    var hit = null;
    for (var i = 0; i < marks.length; i++) {
      if (marks[i].t.getBoundingClientRect().top <= line) hit = marks[i];
    }
    // at the very bottom the last sections cannot reach the line: the last one on screen is the one being read
    var doc = document.documentElement;
    if (window.innerHeight + window.scrollY >= doc.scrollHeight - 2) {
      for (var k = marks.length - 1; k >= 0; k--) {
        if (marks[k].t.getBoundingClientRect().top < window.innerHeight) { hit = marks[k]; break; }
      }
    }
    if (hit !== current) {
      if (current) current.a.removeAttribute('aria-current');
      if (hit) hit.a.setAttribute('aria-current', 'true');
      current = hit;
    }
  }
  if (marks.length) {
    window.addEventListener('scroll', function () {
      if (!queued) {
        queued = true;
        window.requestAnimationFrame(spy);
      }
    }, { passive: true });
    window.addEventListener('resize', spy);
    spy();
  }

  // the folded contents list closes once a reader picks a section
  var mob = document.querySelector('.toc-mob');
  if (mob) {
    mob.addEventListener('click', function (e) {
      if (e.target.closest && e.target.closest('a')) mob.open = false;
    });
  }

  /* ---------- citation: copy the BibTeX ---------- */
  var copy = document.querySelector('.bib-copy');
  var bib = document.getElementById('bibtex');
  if (copy && bib && navigator.clipboard && window.isSecureContext) {
    copy.hidden = false;
    copy.addEventListener('click', function () {
      navigator.clipboard.writeText(bib.textContent).then(function () {
        copy.textContent = 'Copied';
        setTimeout(function () { copy.textContent = 'Copy'; }, 1600);
      }).catch(function () {});
    });
  }
})();

/* Animated block diagrams (figure.anim). Each figure is a story in steps (data-dur: seconds per step). Parts with
   data-s hold one state per step (opacity, translation, scale), eased in; tokens (.tk) move along their paths
   (data-p) in the steps listed in data-on; joint-state bars (.gb) tick. Only transform and opacity are animated, with
   the Web Animations API, and everything belonging to a figure pauses together: on the button, off screen and while
   the tab is hidden. With prefers-reduced-motion the steps are still frames behind a manual stepper. Without the
   script, and in print, each figure is its still drawing (the last step) with the steps listed under it. */
(function () {
  if (!('animate' in Element.prototype) || typeof KeyframeEffect === 'undefined') return;
  var EASE = 'cubic-bezier(.65,0,.35,1)';
  var reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var all = [];

  function num(v, d) { var n = parseFloat(v); return isNaN(n) ? d : n; }
  function tf(s) { return 'translate(' + s.x + 'px,' + s.y + 'px) scale(' + (s.s * s.sx) + ',' + s.s + ')'; }
  function same(a, b) { return a.o === b.o && a.x === b.x && a.y === b.y && a.s === b.s && a.sx === b.sx; }
  function parseSpec(str, n) {
    var out = {}, last = { o: 1, x: 0, y: 0, s: 1, sx: 1 };
    (str || '').trim().split(/\s+/).forEach(function (part) {
      var eq = part.indexOf('=');
      if (eq < 0) return;
      var st = { o: 1, x: 0, y: 0, s: 1, sx: 1, at: 0, d: 0.45, pop: false, re: false };
      part.slice(eq + 1).split(',').forEach(function (kv) {
        if (kv === 'pop' || kv === 're') st[kv] = true;
        else if (kv.charAt(0) === '@') st.at = num(kv.slice(1), 0);
        else { var i = kv.indexOf(':'); st[kv.slice(0, i)] = num(kv.slice(i + 1), 0); }
      });
      out[part.slice(0, eq)] = st;
    });
    for (var k = 1; k <= n; k++) {   // a step not listed: hidden, where it was
      if (out[k]) last = out[k];
      else out[k] = { o: 0, x: last.x, y: last.y, s: last.s, sx: last.sx, at: 0, d: 0.45, pop: false, re: false };
    }
    return out;
  }

  function Story(fig) {
    var self = this;
    this.fig = fig;
    this.dur = (fig.getAttribute('data-dur') || '4').trim().split(/\s+/).map(function (v) { return num(v, 4) * 1000; });
    this.n = this.dur.length;
    this.k = 1;
    this.running = [];        // token, gauge, bar and clock animations of the current step
    this.userPaused = false;
    this.visible = false;
    this.started = false;
    this.playing = false;
    this.parts = Array.prototype.map.call(fig.querySelectorAll('[data-s]'), function (el) {
      var still = !el.classList.contains('sh');
      return { el: el, spec: parseSpec(el.getAttribute('data-s'), self.n),
               cur: { o: still ? 1 : 0, x: 0, y: 0, s: 1, sx: 1 }, anims: [] };
    });
    this.tokens = Array.prototype.map.call(fig.querySelectorAll('.tk'), function (el) {
      var pts = [[0, 0]];
      (el.getAttribute('data-p') || '').trim().split(/\s+/).forEach(function (p) {
        if (!p) return;
        var xy = p.split(',');
        pts.push([num(xy[0], 0), num(xy[1], 0)]);
      });
      return { el: el, on: ' ' + (el.getAttribute('data-on') || '') + ' ', t: num(el.getAttribute('data-t'), 0) * 1000,
               dur: num(el.getAttribute('data-dur'), 1) * 1000, kf: tokenFrames(pts) };
    });
    this.nudges = Array.prototype.map.call(fig.querySelectorAll('.nd'), function (el) {
      var m = (el.getAttribute('data-n') || '').split('@');
      return { el: el, k: m[0], t: num(m[1], 0) * 1000 };
    });
    this.gauges = fig.querySelectorAll('.gb');
    this.btn = fig.querySelector('.anim-play');
    this.label = fig.querySelector('.anim-label');
    this.labels = fig.querySelectorAll('.anim-label span');
    this.dots = fig.querySelectorAll('.anim-dot');
    this.bar = fig.querySelector('.anim-bar i');
    fig.classList.add('anim-ready');
    if (reduce) fig.classList.add('anim-rm');
    if (this.btn) this.btn.addEventListener('click', function () { self.userPaused = self.playing; self.update(); });
    Array.prototype.forEach.call(this.dots, function (d) {
      d.addEventListener('click', function () { self.jump(+d.getAttribute('data-k')); });
    });
    var ctl = fig.querySelector('.anim-ctl');
    if (ctl) ctl.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        var k = (self.k + (e.key === 'ArrowRight' ? 0 : self.n - 2)) % self.n + 1;
        self.jump(k);
        var d = self.dots[k - 1];
        if (d) d.focus();
      }
    });
    this.go(1, true);
  }

  // keyframes for a token: eased along its polyline as a whole, fading in at the start and out at the end
  function tokenFrames(pts) {
    var L = 0, segs = [];
    for (var i = 1; i < pts.length; i++) {
      var l = Math.sqrt(Math.pow(pts[i][0] - pts[i - 1][0], 2) + Math.pow(pts[i][1] - pts[i - 1][1], 2));
      segs.push(l); L += l;
    }
    var offs = [0, 0.16, 0.78, 1], acc = 0;
    for (i = 0; i < segs.length - 1; i++) { acc += segs[i]; if (L) offs.push(acc / L); }
    offs.sort(function (a, b) { return a - b; });
    function at(f) {   // the point a fraction f along the path
      if (!L) return pts[0];
      var d = f * L;
      for (var j = 0; j < segs.length; j++) {
        if (d <= segs[j] || j === segs.length - 1) {
          var u = segs[j] ? Math.min(1, d / segs[j]) : 1;
          return [pts[j][0] + (pts[j + 1][0] - pts[j][0]) * u, pts[j][1] + (pts[j + 1][1] - pts[j][1]) * u];
        }
        d -= segs[j];
      }
      return pts[pts.length - 1];
    }
    var out = [], seen = {};
    offs.forEach(function (f) {
      var key = f.toFixed(4);
      if (seen[key]) return;
      seen[key] = 1;
      var p = at(f);
      out.push({ offset: f, transform: 'translate(' + p[0].toFixed(2) + 'px,' + p[1].toFixed(2) + 'px)',
                 opacity: f < 0.16 ? f / 0.16 : f > 0.78 ? (1 - f) / 0.22 : 1 });
    });
    return out;
  }

  Story.prototype.setState = function (p, st, instant) {
    var el = p.el, prev = p.cur, old = p.anims, fresh = [];
    var to = { opacity: st.o, transform: tf(st) };
    if (instant) {
      fresh.push(el.animate([to, to], { duration: 0, fill: 'forwards' }));
    } else if (st.o && (st.pop || st.re) && (prev.o || !same(prev, st))) {
      var from = { opacity: prev.o, transform: tf(prev) };
      if (prev.o) fresh.push(el.animate([from, { opacity: 0, transform: tf(prev) }], { duration: 260, easing: EASE, fill: 'forwards' }));
      if (st.pop) {
        var small = { o: 0, x: st.x, y: st.y, s: st.s * 0.6, sx: st.sx };
        var big = { o: 1, x: st.x, y: st.y, s: st.s * 1.07, sx: st.sx };
        fresh.push(el.animate([{ opacity: 0, transform: tf(small) }, { opacity: 1, transform: tf(big), offset: 0.6 }, to],
          { duration: 480, delay: Math.max(st.at * 1000, prev.o ? 260 : 0), easing: 'cubic-bezier(.3,0,.25,1)', fill: prev.o ? 'forwards' : 'both' }));
      } else {
        fresh.push(el.animate([{ opacity: 0, transform: tf(st) }, to],
          { duration: st.d * 1000, delay: Math.max(st.at * 1000, prev.o ? 260 : 0), easing: EASE, fill: prev.o ? 'forwards' : 'both' }));
      }
    } else if (!same(prev, st)) {
      fresh.push(el.animate([{ opacity: prev.o, transform: tf(prev) }, to],
        { duration: st.d * 1000, delay: st.at * 1000, easing: EASE, fill: 'both' }));
    } else {
      return;
    }
    old.forEach(function (a) { a.cancel(); });
    p.anims = fresh;
    p.cur = { o: st.o, x: st.x, y: st.y, s: st.s, sx: st.sx };
  };

  Story.prototype.clear = function () {
    this.running.forEach(function (a) { a.cancel(); });
    this.running = [];
  };

  // step k: every part eases to its state, the step's tokens set off, the clock runs the step
  Story.prototype.go = function (k, instant) {
    var self = this;
    this.k = k;
    this.clear();
    var stepping = !instant && !this.playing;   // stepping while paused: the step's still frame, without waiting
    this.parts.forEach(function (p) {
      var st = p.spec[k];
      if (stepping && st.at) {
        st = { o: st.o, x: st.x, y: st.y, s: st.s, sx: st.sx, at: 0, d: st.d, pop: st.pop, re: st.re };
      }
      self.setState(p, st, instant || reduce);
      // a part that was paused before it had appeared would stay hidden: take it to the end of its move
      if (stepping) p.anims.forEach(function (a) { if (a.playState === 'paused') a.finish(); });
    });
    Array.prototype.forEach.call(this.dots, function (d) {
      if (+d.getAttribute('data-k') === k) d.setAttribute('aria-current', 'step'); else d.removeAttribute('aria-current');
    });
    Array.prototype.forEach.call(this.labels, function (l) { l.classList.toggle('on', +l.getAttribute('data-k') === k); });
    if (reduce || instant) return;
    var key = ' ' + k + ' ';
    this.tokens.forEach(function (t) {
      if (t.on.indexOf(key) < 0) return;
      self.running.push(t.el.animate(t.kf, { duration: t.dur, delay: t.t, easing: EASE }));
    });
    this.nudges.forEach(function (n) {
      if (n.k !== String(k)) return;
      self.running.push(n.el.animate([{ transform: 'translate(0px,0px)' }, { transform: 'translate(-34px,0px)', offset: 0.42 },
        { transform: 'translate(-26px,0px)', offset: 0.6 }, { transform: 'translate(0px,0px)' }],
        { duration: 1100, delay: n.t, easing: EASE }));
    });
    if (this.bar) this.running.push(this.bar.animate([{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }],
      { duration: this.dur[k - 1], fill: 'forwards' }));
    var clock = new Animation(new KeyframeEffect(null, null, { duration: this.dur[k - 1] }), document.timeline);
    clock.onfinish = function () { if (self.k === k) self.go(k % self.n + 1); };
    clock.play();
    this.running.push(clock);
    if (!this.playing) this.running.forEach(function (a) { a.pause(); });
  };


  // the joint-state readout ticks all the time the figure plays, across steps
  Story.prototype.startLoops = function () {
    var self = this;
    this.loops = Array.prototype.map.call(this.gauges, function (g, i) {
      var v = (g.getAttribute('data-g') || '1').split(' ').map(function (x) { return num(x, 1); });
      var kf = v.concat([v[0]]).map(function (s) { return { transform: 'scale(1,' + s + ')', easing: EASE }; });
      var a = g.animate(kf, { duration: 560 * v.length, iterations: Infinity });
      a.currentTime = i * 137;
      if (!self.playing) a.pause();
      return a;
    });
  };

  Story.prototype.jump = function (k) {
    if (!reduce) this.userPaused = true;
    if (this.label) this.label.setAttribute('aria-live', 'polite');
    this.update();
    this.go(k, false);
  };

  // play while on screen, the tab is visible and nobody paused it
  Story.prototype.update = function () {
    var want = !reduce && !this.userPaused && this.visible && !document.hidden;
    var self = this;
    if (want && !this.started) {   // the first time it plays: what appears later in the step starts hidden
      this.started = true;
      this.playing = true;
      this.parts.forEach(function (p) {
        var st = p.spec[self.k];
        if (st.o && (st.pop || st.at)) self.setState(p, { o: 0, x: st.x, y: st.y, s: st.s, sx: st.sx }, true);
      });
      this.startLoops();
      this.go(this.k, false);
    }
    if (want !== this.playing) {
      this.playing = want;
      var parts = [];
      this.parts.forEach(function (q) { parts = parts.concat(q.anims); });
      this.running.concat(this.loops || [], parts).forEach(function (a) {
        if (want && a.playState === 'paused') a.play();
        else if (!want && a.playState === 'running') a.pause();
      });
    }
    if (this.btn) {
      var paused = !want;
      this.btn.setAttribute('aria-pressed', paused ? 'true' : 'false');
      this.btn.lastChild.textContent = this.userPaused ? 'Play' : 'Pause';
      this.btn.setAttribute('aria-label', this.userPaused ? 'Play the animation' : 'Pause the animation');
    }
  };

  // print and the still image: drop every animation, so the drawing's own still state shows
  Story.prototype.still = function () {
    this.clear();
    (this.loops || []).forEach(function (a) { a.cancel(); });
    this.parts.forEach(function (p) { p.anims.forEach(function (a) { a.cancel(); }); p.anims = []; });
  };

  Array.prototype.forEach.call(document.querySelectorAll('figure.anim'), function (fig) { all.push(new Story(fig)); });
  if (!all.length) return;
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        all.forEach(function (s) { if (s.fig === e.target) { s.visible = e.isIntersecting; s.update(); } });
      });
    }, { threshold: 0.2 });
    all.forEach(function (s) { io.observe(s.fig); });
  } else {
    all.forEach(function (s) { s.visible = true; s.update(); });
  }
  document.addEventListener('visibilitychange', function () { all.forEach(function (s) { s.update(); }); });
  window.addEventListener('beforeprint', function () { all.forEach(function (s) { s.still(); }); });
  window.addEventListener('afterprint', function () {
    all.forEach(function (s) {
      s.parts.forEach(function (p) { p.cur = { o: p.el.classList.contains('sh') ? 0 : 1, x: 0, y: 0, s: 1, sx: 1 }; });
      s.go(s.k, !s.playing);
      if (s.started) s.startLoops();
    });
  });
})();

/* A link to a place inside a closed section ("All numbers and records") opens that section first. */
(function () {
  function reveal(id) {
    var el = id && document.getElementById(id);
    var d = el && el.closest ? el.closest('details') : null;
    while (d) { d.open = true; d = d.parentElement ? d.parentElement.closest('details') : null; }
  }
  document.addEventListener('click', function (e) {
    var a = e.target && e.target.closest ? e.target.closest('a[href^="#"]') : null;
    if (a) reveal(decodeURIComponent(a.getAttribute('href').slice(1)));
  });
  window.addEventListener('hashchange', function () { reveal(decodeURIComponent(location.hash.slice(1))); });
  if (location.hash) reveal(decodeURIComponent(location.hash.slice(1)));
})();
