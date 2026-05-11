// Builds a nested table of contents from h2/h3 headings in .post-content.
// Activation rules (set via data-toc-mode on #post-toc):
//   "force" : render whenever there's >=1 heading
//   "auto"  : render only when there are more than 3 headings
// Assigns ids to headings that don't have one, so the TOC anchors scroll correctly.
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

  function makeItem(heading, id) {
    const li = document.createElement('li');
    li.className = 'toc-' + heading.tagName.toLowerCase();
    const a = document.createElement('a');
    a.href = '#' + id;
    a.textContent = heading.textContent;
    li.appendChild(a);
    return li;
  }

  document.addEventListener('DOMContentLoaded', function () {
    const tocEl = document.getElementById('post-toc');
    if (!tocEl) return;

    const content = document.querySelector('.post-content');
    if (!content) return;

    const headings = Array.from(content.querySelectorAll('h2, h3'));
    const mode = tocEl.getAttribute('data-toc-mode') || 'auto';
    const threshold = 3;

    if (headings.length === 0) return;
    if (mode === 'auto' && headings.length <= threshold) return;

    const list = tocEl.querySelector('.post-toc-list');
    const used = new Set();
    let currentH2Li = null;

    headings.forEach(function (h) {
      const id = ensureUniqueId(h, used);
      const li = makeItem(h, id);

      if (h.tagName === 'H2') {
        list.appendChild(li);
        currentH2Li = li;
      } else {
        // H3: nest under the most recent H2. If none yet (post starts with h3),
        // fall back to the top level so we still show something useful.
        if (currentH2Li) {
          let sub = currentH2Li.querySelector(':scope > ul.post-toc-sublist');
          if (!sub) {
            sub = document.createElement('ul');
            sub.className = 'post-toc-sublist';
            currentH2Li.appendChild(sub);
          }
          sub.appendChild(li);
        } else {
          list.appendChild(li);
        }
      }
    });

    tocEl.hidden = false;
  });
})();
