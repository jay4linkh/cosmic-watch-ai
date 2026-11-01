// "Teach me": the galaxy CNN shows you the objects it is least sure about; your answers become
// training labels, and active learning retrains it in the background once enough have arrived.
import { esc } from './fx.js';

const ANSWERS = [
  ['spiral', 'Spiral', '1'], ['elliptical', 'Elliptical', '2'], ['edge-on', 'Edge-on', '3'], ['merger', 'Merger', '4'],
  ['star', 'Star', '5'], ['gas', 'Gas / nebula', '6'], ['artifact', 'Artifact', '7'],
];

export function createTeacher({ toast, onLabel }) {
  const root = document.createElement('div');
  root.id = 'teach';
  root.hidden = true;
  root.innerHTML = `<div class="teach-card glass">
    <button class="icon-btn small teach-close" title="Close (Esc)">×</button>
    <div class="dz-agent-head"><i class="dz-orb"></i>COSMIC AGENT · TEACH ME <span class="teach-count"></span></div>
    <div class="teach-body"></div></div>`;
  document.body.appendChild(root);
  const $ = (s) => root.querySelector(s);
  let items = [], i = 0, done = 0;

  function show() {
    const it = items[i];
    if (!it) {
      $('.teach-body').innerHTML = `<p class="teach-q">That's all for now: thank you! You taught me ${done} object${done === 1 ? '' : 's'}.</p>
        <p class="feed-dim">Once ${'enough'} new labels have arrived I retrain myself on this laptop's GPU, and keep the new model only if it scores better.</p>`;
      $('.teach-count').textContent = '';
      return;
    }
    $('.teach-count').textContent = `${i + 1} / ${items.length}`;
    $('.teach-body').innerHTML = `
      <div class="teach-grid">
        <div class="teach-img"><img src="${it.crop_url}" alt="object to classify"><span>${esc(it.source.toUpperCase())} · ${esc(it.target)}</span></div>
        <div>
          <p class="teach-q">${esc(it.reason)}</p>
          <div class="cnn-box"><div class="cnn-head">MY CURRENT GUESS</div>${it.top.map(([k, p]) =>
            `<div class="cnn-row"><span>${esc(k)}</span><i style="--p:${Math.round(p * 100)}%"></i><b>${Math.round(p * 100)}%</b></div>`).join('')}</div>
          <p class="feed-dim small">What is it? (keys 1-7, S to skip)</p>
        </div>
      </div>
      <div class="teach-answers">${ANSWERS.map(([k, l, key]) => `<button data-label="${k}"><kbd>${key}</kbd>${l}</button>`).join('')}
        <button data-label="skip" class="ghost"><kbd>S</kbd>Skip</button></div>`;
  }

  async function answer(label) {
    const it = items[i];
    if (!it) return;
    if (label !== 'skip') {
      try {
        const r = await fetch('/api/labels', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key: it.key, kind: 'galaxy', label, ai_label: it.ai_label }) });
        const j = await r.json();
        done++;
        onLabel?.(j.total);
      } catch { toast?.('Could not save that answer', '#fb7185'); return; }
    }
    i++;
    show();
  }

  root.addEventListener('click', (e) => {
    if (e.target === root || e.target.closest('.teach-close')) return close();
    const b = e.target.closest('button[data-label]');
    if (b) answer(b.dataset.label);
  });
  addEventListener('keydown', (e) => {
    if (root.hidden || e.target?.matches?.('input, select')) return;
    const a = ANSWERS.find(([, , key]) => key === e.key);
    if (a) { e.stopImmediatePropagation(); answer(a[0]); }
    else if (e.key === 's' || e.key === 'S') { e.stopImmediatePropagation(); answer('skip'); }
    else if (e.key === 'Escape') { e.stopImmediatePropagation(); close(); }
  }, true);

  async function open() {
    root.hidden = false;
    i = 0; done = 0;
    $('.teach-body').innerHTML = '<p class="feed-dim">Finding the objects I am least sure about…</p>';
    try { items = await fetch('/api/learn/queue?n=20').then((r) => r.json()); } catch { items = []; }
    show();
  }
  function close() { root.hidden = true; }
  return { open, close, get isOpen() { return !root.hidden; } };
}
