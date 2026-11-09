/* Charts draw themselves when they scroll into view. A figure.reveal that is still below the screen gets .rv-wait,
   which holds its bars at zero and its marks at the unedited value (post.css). The class comes off when the figure
   arrives, and the CSS transitions do the rest. Without IntersectionObserver, in print, and for readers who ask for
   reduced motion, nothing is held back. */
(function () {
  'use strict';

  var figs = Array.prototype.slice.call(document.querySelectorAll('figure.reveal'));
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!figs.length || reduce || !('IntersectionObserver' in window)) return;

  // bars in the imported SVG charts take their turn in drawing order
  figs.forEach(function (fig) {
    Array.prototype.forEach.call(fig.querySelectorAll('svg rect[class^="dg-bar"], svg circle'), function (el, i) {
      el.style.setProperty('--i', Math.min(i, 12));
    });
  });

  function show(fig) {
    fig.classList.remove('rv-wait');
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        show(e.target);
        io.unobserve(e.target);
      }
    });
  }, { rootMargin: '0px 0px -18% 0px' });

  figs.forEach(function (fig) {
    // a figure already on screen (a reload half way down the page) is left drawn
    var r = fig.getBoundingClientRect();
    if (r.top < window.innerHeight && r.bottom > 0) return;
    fig.classList.add('rv-wait');
    io.observe(fig);
  });

  window.addEventListener('beforeprint', function () { figs.forEach(show); });
})();
