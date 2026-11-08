/* Renders the comparison tables from metrics.json. The page ships the same tables pre-rendered (no-JS fallback,
   written by site/scripts/make_cosmos_metrics.py), so this only matters when metrics.json is newer than the HTML. */
(function () {
  'use strict';
  var blocks = document.querySelectorAll('[data-table]');
  if (!blocks.length || !window.fetch) return;

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
        td.appendChild(inline(el('span', 'v'), c.v));
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
      a.href = REPO + (/\.[a-z]+$/i.test(s) ? '/blob/main/' : '/tree/main/') + s;
      var c = el('code');
      c.textContent = s.replace('study/results/', '');
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
    Array.prototype.forEach.call(blocks, function (b) {
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

  fetch('metrics.json', { cache: 'no-cache' })
    .then(function (r) {
      if (!r.ok) throw new Error('metrics.json ' + r.status);
      return r.json();
    })
    .then(render)
    .catch(function () {
      /* keep the pre-rendered tables */
    });
})();
