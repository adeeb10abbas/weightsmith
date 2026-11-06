/* Shared behaviour for both pages: one tooltip for every [data-tip] mark, and the BibTeX copy button.
   Tooltips only repeat what the page already says in text, so nothing depends on them. */
(function () {
  'use strict';

  var tip = document.createElement('div');
  tip.className = 'tooltip';
  tip.setAttribute('role', 'tooltip');
  tip.setAttribute('aria-hidden', 'true');
  document.body.appendChild(tip);
  var current = null;

  // "value|detail|detail" -> first line emphasised; built with textContent, never innerHTML.
  function fill(text) {
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    String(text).split('|').forEach(function (line, i) {
      var d = document.createElement('div');
      d.className = i === 0 ? 'tt-v' : i === 1 ? '' : 'tt-s';
      d.textContent = line;
      tip.appendChild(d);
    });
  }

  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight, vw = window.innerWidth, pad = 12;
    var left = Math.min(Math.max(pad, x - w / 2), vw - w - pad);
    var top = y - h - 14;
    if (top < pad) top = y + 22;
    tip.style.transform = 'translate(' + Math.round(left) + 'px,' + Math.round(top) + 'px)';
  }

  function show(el, x, y) {
    if (current !== el) fill(el.getAttribute('data-tip'));
    current = el;
    tip.classList.add('show');
    if (x === undefined) {
      var r = el.getBoundingClientRect();
      x = r.left + r.width / 2;
      y = r.top;
    }
    place(x, y);
  }

  function hide() {
    current = null;
    tip.classList.remove('show');
  }

  window.wsTip = { show: show, hide: hide, fill: fill, place: place, el: tip };

  document.addEventListener('pointermove', function (e) {
    if (e.pointerType === 'touch') return;
    var el = e.target.closest && e.target.closest('[data-tip]');
    if (el) show(el, e.clientX, e.clientY);
    else if (current && !current.contains(document.activeElement)) hide();
  });
  document.addEventListener('click', function (e) {
    var el = e.target.closest && e.target.closest('[data-tip]');
    if (el) {
      var r = el.getBoundingClientRect();
      show(el, e.clientX || r.left + r.width / 2, e.clientY || r.top);
    } else hide();
  });
  document.addEventListener('focusin', function (e) {
    var el = e.target.closest && e.target.closest('[data-tip]');
    if (el) show(el);
  });
  document.addEventListener('focusout', function (e) {
    if (current && e.target === current) hide();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') hide();
  });
  window.addEventListener('scroll', function () { if (current && document.activeElement !== current) hide(); }, { passive: true });

  // Short muted loops (the earlier-work card) play only while they are on screen, and never on their own for readers
  // who ask for reduced motion: they get the poster and the player's controls instead.
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelectorAll('video.lazyplay').forEach(function (v) {
    function play() { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
    if (reduce) { v.controls = true; return; }
    if (!('IntersectionObserver' in window)) { play(); return; }
    new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) play(); else v.pause(); });
    }, { threshold: 0.25 }).observe(v);
  });

  // Copy BibTeX, falling back to selecting the text.
  document.querySelectorAll('[data-copy]').forEach(function (btn) {
    var src = document.getElementById(btn.getAttribute('data-copy'));
    var scope = btn.closest('section, .cite');
    var status = scope && scope.querySelector('.copied');
    var label = btn.textContent;
    function done(msg, ok) {
      if (status) status.textContent = msg;
      btn.textContent = ok ? 'Copied' : label;
      setTimeout(function () {
        btn.textContent = label;
        if (status) status.textContent = '';
      }, 2400);
    }
    function selectIt() {
      var range = document.createRange();
      range.selectNodeContents(src);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      done('Selected. Press Ctrl+C or ⌘C to copy.', false);
    }
    btn.addEventListener('click', function () {
      var text = src.textContent;
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function () { done('BibTeX copied to the clipboard.', true); }, selectIt);
      } else selectIt();
    });
  });
})();
