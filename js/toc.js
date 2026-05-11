// Builds a table of contents from h2/h3 headings in .post-content.
// Activation rules:
//   data-toc-mode="force" : render whenever there's >=1 heading
//   data-toc-mode="auto"  : render only when there are more than 3 headings
// Assigns ids to headings that don't have one, so the TOC links scroll correctly.
(function () {
  function slugify(text) {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-');
  }

  function ensureUniqueId(el, used) {
    let id = el.id || slugify(el.textContent || 'section');
    if (!id) id = 'section';
    let candidate = id;
    let n = 1;
    while (
      used.has(candidate) ||
      (document.getElementById(candidate) && document.getElementById(candidate) !== el)
    ) {
      n += 1;
      candidate = id + '-' + n;
    }
    el.id = candidate;
    used.add(candidate);
    return candidate;
  }

  document.addEventListener('DOMContentLoaded', function () {
    const tocEl = document.getElementById('post-toc');
    if (!tocEl) return;

    const content = document.querySelector('.post-content');
    if (!content) return;

    const headings = Array.from(content.querySelectorAll('h2, h3'));
    const mode = tocEl.getAttribute('data-toc-mode') || 'auto';
    const threshold = 3;

    if (mode === 'auto' && headings.length <= threshold) return;
    if (headings.length === 0) return;

    const list = tocEl.querySelector('.post-toc-list');
    const used = new Set();
    headings.forEach(function (h) {
      const id = ensureUniqueId(h, used);
      const li = document.createElement('li');
      li.className = 'toc-' + h.tagName.toLowerCase();
      const a = document.createElement('a');
      a.href = '#' + id;
      a.textContent = h.textContent;
      li.appendChild(a);
      list.appendChild(li);
    });

    tocEl.hidden = false;
  });
})();
