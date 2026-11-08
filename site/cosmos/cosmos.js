/* The cross-model draft page.
   1. Renders the comparison tables from metrics.json. The page ships the same tables pre-rendered (no-JS fallback,
      written by site/scripts/make_cosmos_metrics.py), so this only matters when metrics.json is newer than the HTML.
   2. Clips: a placeholder becomes a player as soon as its MP4 exists in video/, and players run only while in view,
      never for readers who ask for reduced motion. */
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
})();
