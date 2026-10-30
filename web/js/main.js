import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { api, makeStarfield, makeNebula, timeAgo, fmt, clamp } from './fx.js';
import { createObservatory } from './views/observatory.js';
import { createDeep } from './views/deep.js';
import { createSun } from './views/sun.js';
import { createBlackHoles } from './views/blackholes.js';
import { createInspector } from './inspect.js';

const $ = (s) => document.querySelector(s);
const SOURCE_COLOR = { lsst: '#67e8f9', ztf: '#5eead4', jwst: '#fcd34d', hst: '#d8b4fe', sdo: '#fdba74', alerts: '#a78bfa', mast: '#60a5fa' };
const KIND_COLOR = { blackhole: '#f472b6', transient: '#fbbf24', error: '#fb7185' };

// ---------- renderer ----------
const canvas = $('#stage');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
const SPACE = 0x000002;   // deep, neutral black (no blue cast)
renderer.setClearColor(SPACE, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const labelRenderer = new CSS2DRenderer({ element: $('#labels') });
labelRenderer.setSize(innerWidth, innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 3000);
camera.position.set(0, 30, 180);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 12;
controls.maxDistance = 220;
controls.autoRotateSpeed = 0.35;

scene.add(new THREE.AmbientLight(0x8899cc, 0.6));
const key = new THREE.PointLight(0xffffff, 1.4, 0, 0);
key.position.set(30, 40, 60);
scene.add(key);

const bg = new THREE.Group();
bg.add(makeStarfield(), makeNebula());
scene.add(bg);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.6, 0.5, 0.32);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---------- state ----------
const state = { explode: 0, target: 0, handMode: false, inspecting: false, summary: null, seen: new Set(), firstEvents: true };
let current = null;
let hands = null;

const ctx = {
  camera,
  get handMode() { return state.handMode; },
  goTo: (v) => switchView(v),
  inspect: (t) => openInspect(t),
  flyTo: (home, ms) => flyTo(home, ms),
  onInfoChange: () => { renderInfo(); renderItems(); },
  showDetail(html) { $('#detail-body').innerHTML = html; $('#detail').classList.add('open'); },
};

const views = {
  observatory: createObservatory(ctx),
  deep: createDeep(ctx),
  bh: createBlackHoles(ctx),
  sun: createSun(ctx),
};
// Only the active view's group is in the scene. (CSS2DRenderer ignores parent visibility,
// so a merely hidden group would still draw its labels.)
function hideLabels(group) {
  group.traverse((o) => { if (o.isCSS2DObject) o.element.style.display = 'none'; });
}

// ---------- inspect mode ----------
const inspector = createInspector(ctx);
const INSPECT_HOME = { pos: [-2, 7, 44], target: [6, 0, 0] };
const INSPECT_FAR = 46, INSPECT_NEAR = 13;

async function openInspect(target) {
  if (!state.inspecting) {
    $('#fade').style.opacity = 1;
    await new Promise((r) => setTimeout(r, 260));
    scene.remove(current.group);
    hideLabels(current.group);
    current.unhover?.();
    hideTip();
    scene.add(inspector.group);
    state.inspecting = true;
    document.body.classList.add('inspecting');
    controls.enableZoom = false;
    controls.autoRotate = false;
    camera.position.set(0, 20, 110);
    flyTo(INSPECT_HOME, 1600);
  }
  state.explode = state.target = 0;
  $('#explode-label').textContent = 'Zoom';
  $('#inspect-back b').textContent = current.title;
  await inspector.open(target);
  renderInfo();
  renderItems();
  $('#fade').style.opacity = 0;
}

async function closeInspect({ fly = true } = {}) {
  if (!state.inspecting) return;
  $('#fade').style.opacity = 1;
  await new Promise((r) => setTimeout(r, 240));
  inspector.close();
  scene.remove(inspector.group);
  hideLabels(inspector.group);
  state.inspecting = false;
  document.body.classList.remove('inspecting');
  controls.enableZoom = true;
  controls.enableRotate = true;
  scene.add(current.group);
  state.explode = state.target = 0;
  $('#explode-label').textContent = current.explodeLabel || 'Exploded view';
  if (fly) flyTo(current.home, 1100);
  renderInfo();
  renderItems();
  $('#fade').style.opacity = 0;
}
$('#inspect-back').addEventListener('click', () => closeInspect());
canvas.addEventListener('wheel', (e) => {
  if (!state.inspecting) return;
  e.preventDefault();
  setExplodeTarget(state.target - e.deltaY * 0.0012);
}, { passive: false });

// ---------- camera flight ----------
let flight = null;
function flyTo(home, ms = 1400) {
  flight = {
    fromPos: camera.position.clone(), toPos: new THREE.Vector3(...home.pos),
    fromT: controls.target.clone(), toT: new THREE.Vector3(...home.target),
    t0: performance.now(), ms,
  };
}
const ease = (t) => 1 - Math.pow(1 - t, 3);

// ---------- views ----------
async function switchView(name) {
  const next = views[name];
  if (!next) return;
  if (state.inspecting) await closeInspect({ fly: next === current });
  if (next === current) return;
  $('#fade').style.opacity = 1;
  await new Promise((r) => setTimeout(r, 260));
  if (current) { scene.remove(current.group); hideLabels(current.group); current.unhover?.(); }
  current = next;
  document.querySelectorAll('#views button').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
  $('#detail').classList.remove('open');
  hideTip();
  state.explode = state.target = 0;
  $('#explode-label').textContent = current.explodeLabel || 'Exploded view';
  try { await current.enter?.(); } catch (e) { toast(`Could not load ${current.title}: ${e.message}`, '#fb7185'); }
  scene.add(current.group);
  controls.autoRotate = !!current.autoRotate && !state.handMode;
  const h = current.home;
  camera.position.set(h.pos[0] * 1.5, h.pos[1] * 1.5 + 8, h.pos[2] * 1.6);
  flyTo(h);
  renderInfo();
  renderItems();
  $('#fade').style.opacity = 0;
}

function renderInfo() {
  if (!current) return;
  const src = state.inspecting ? inspector : current;
  const i = src.info();
  $('#info-kicker').textContent = i.kicker;
  $('#info-title').textContent = i.title;
  $('#info-sub').textContent = i.sub;
  $('#info-stats').innerHTML = (i.stats || []).slice(0, 6).map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');
  $('#info-legend').innerHTML = (i.legend || []).map(([k, c]) => `<div><i style="background:${c};color:${c}"></i>${k}</div>`).join('');
  renderClassify();
}

// ---------- your classification ----------
const CLASSES = {
  galaxy: [['spiral', 'Spiral'], ['elliptical', 'Elliptical'], ['edge-on', 'Edge-on'], ['merger', 'Merger'], ['star', 'Star'], ['artifact', 'Artifact']],
  bh: [['real', 'Real event'], ['bogus', 'Bogus'], ['TDE', 'TDE'], ['AGN', 'AGN'], ['QSO', 'Quasar'], ['unsure', 'Not sure']],
};
const myLabels = new Map();
let labelCount = 0;

function renderClassify() {
  const box = $('#classify');
  const t = state.inspecting ? inspector.target : null;
  if (!t) { box.innerHTML = ''; return; }
  const mine = myLabels.get(t.key);
  const ai = t.kind === 'bh' ? t.cls : t.aiLabel;
  box.innerHTML = `
    <div class="control-head"><span>Your classification</span><b>${labelCount ? `${labelCount} labelled` : ''}</b></div>
    <div class="classify-ai">AI says <b>${ai}</b>${mine ? ` · you said <b class="mine">${mine}</b>` : ' · what do you think?'}</div>
    <div class="classify-grid">${CLASSES[t.kind].map(([k, l]) => `<button data-label="${k}" class="${mine === k ? 'on' : ''}">${l}</button>`).join('')}</div>`;
}

$('#classify').addEventListener('click', async (e) => {
  const b = e.target.closest('button[data-label]');
  const t = inspector.target;
  if (!b || !t) return;
  const lbl = b.dataset.label;
  myLabels.set(t.key, lbl);
  renderClassify();
  try {
    const r = await fetch('/api/labels', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: t.key, kind: t.kind, label: lbl, ai_label: t.kind === 'bh' ? t.cls : t.aiLabel }),
    });
    const j = await r.json();
    labelCount = j.total ?? labelCount;
    toast(`Saved: <b>${lbl}</b> · ${labelCount} object${labelCount === 1 ? '' : 's'} labelled (training data for the next model)`, '#4ade80');
  } catch { toast('Could not save your label', '#fb7185'); }
  // the 3D model follows your call
  if (t.kind === 'galaxy' && ['spiral', 'elliptical', 'edge-on', 'star'].includes(lbl)) inspector.reshape(lbl);
  renderClassify();
});

function renderItems() {
  const box = $('#items');
  const src = state.inspecting ? inspector : current;
  const items = src?.items() || [];
  box.innerHTML = items.map((it, i) => `<button data-i="${i}" class="${i === src.itemIndex ? 'active' : ''}">${it.label}<small>${it.sub ?? ''}</small></button>`).join('');
  box.querySelector('.active')?.scrollIntoView({ block: 'nearest', inline: 'center' });
}
$('#items').addEventListener('click', async (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  if (state.inspecting) { await inspector.select(+b.dataset.i); renderInfo(); }
  else await current.selectItem(+b.dataset.i);
  renderItems();
});

document.querySelectorAll('#views button').forEach((b) => b.addEventListener('click', () => switchView(b.dataset.view)));

// ---------- explode control ----------
const slider = $('#explode');
slider.addEventListener('input', () => { state.target = slider.value / 100; });
function setExplodeTarget(v) { state.target = clamp(v, 0, 1); }

// ---------- picking ----------
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.45;
raycaster.params.Line.threshold = 0.3;
const ndc = new THREE.Vector2();
let hoverObj = null;

function pickAt(x, y) {
  if (!current || state.inspecting) return null;
  const targets = current.hoverables;
  if (!targets?.length) return null;
  raycaster.params.Points.threshold = state.handMode ? 1.0 : 0.45;   // fingers are less precise than a mouse
  ndc.set((x / innerWidth) * 2 - 1, -(y / innerHeight) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(targets, false);
  return hits[0] || null;
}

function hoverAt(x, y) {
  const hit = pickAt(x, y);
  const html = hit ? current.hover?.(hit.object, hit) : null;
  if (!html) { hoverObj = null; current?.unhover?.(); hideTip(); canvas.style.cursor = ''; return null; }
  hoverObj = hit;
  showTip(html, x, y);
  canvas.style.cursor = 'pointer';
  return hit;
}

function showTip(html, x, y) {
  const t = $('#tip');
  t.innerHTML = html;
  const w = t.offsetWidth, h = t.offsetHeight;
  t.style.left = `${Math.min(x + 18, innerWidth - w - 12)}px`;
  t.style.top = `${Math.min(y + 18, innerHeight - h - 12)}px`;
  t.classList.add('show');
}
function hideTip() { $('#tip').classList.remove('show'); }

let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
let lastDrag = null;
canvas.addEventListener('pointermove', (e) => {
  if (state.inspecting && inspector.fullness > 0.8 && e.buttons) {     // drag turns the photo in full view
    if (lastDrag) inspector.tiltBy((e.clientX - lastDrag[0]) / innerWidth, (e.clientY - lastDrag[1]) / innerHeight);
    lastDrag = [e.clientX, e.clientY];
    return;
  }
  lastDrag = null;
  if (!state.handMode) hoverAt(e.clientX, e.clientY);
});
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const hit = pickAt(e.clientX, e.clientY);
  if (hit) current.pick?.(hit.object, hit);
});
controls.addEventListener('start', () => { controls.autoRotate = false; });

// ---------- hand mode ----------
function rotateCamera(dx, dy) {
  const off = camera.position.clone().sub(controls.target);
  const sph = new THREE.Spherical().setFromVector3(off);
  sph.theta -= dx * 4.5;
  sph.phi = clamp(sph.phi - dy * 3.5, 0.15, Math.PI - 0.15);
  off.setFromSpherical(sph);
  camera.position.copy(controls.target).add(off);
}
function zoomCamera(f) {
  const off = camera.position.clone().sub(controls.target);
  const len = clamp(off.length() * Math.pow(f, 1.6), controls.minDistance, controls.maxDistance);
  camera.position.copy(controls.target).add(off.setLength(len));
}

const VIEW_ORDER = ['observatory', 'deep', 'bh', 'sun'];
async function toggleHandMode() {
  if (state.handMode) {
    hands?.stop();
    state.handMode = false;
    document.body.classList.remove('hand-mode', 'hands-active');
    controls.autoRotate = !!current?.autoRotate && !state.inspecting;
    renderer.setClearColor(SPACE, 1);
    toast('Hand mode off');
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    toast('Camera needs HTTPS (or localhost) and a browser with camera access', '#fb7185');
    return;
  }
  try {
    const { HandController } = await import('./hands.js');
    const hc = $('#hands');
    hands = hands || new HandController({
      video: $('#cam'), canvas: hc,
      on: {
        explode: (v) => setExplodeTarget(v),
        rotate: (dx, dy) => (state.inspecting && inspector.fullness > 0.8 ? inspector.tiltBy(dx, dy) : rotateCamera(dx, dy)),
        zoom: (f) => zoomCamera(f),
        point: (x, y) => { handHoverDom(x, y); hoverAt(x, y); },
        pointEnd: () => { handHoverDom(-1, -1); if (hoverObj) { hoverObj = null; current?.unhover?.(); hideTip(); } },
        select: (x, y) => {
          // Buttons and tabs are clickable by pointing too.
          const el = domTarget(x, y);
          if (el) { el.click(); return; }
          const hit = pickAt(x, y);
          if (hit) current.pick?.(hit.object, hit);
        },
        snap: async () => {
          if (state.inspecting) { await inspector.next(); renderInfo(); renderItems(); toast('Snap · next object', '#fbbf24'); }
          else if (current.next) { await current.next(); renderItems(); toast('Snap · next item', '#fbbf24'); }
          else { switchView(VIEW_ORDER[(VIEW_ORDER.indexOf(current.name) + 1) % VIEW_ORDER.length]); }
        },
        fist: () => {
          if (state.inspecting) closeInspect();
          else if (current.name !== 'observatory') switchView('observatory');
        },
        presence: (p) => document.body.classList.toggle('hands-active', p),
        status: (g, n, open) => { handStatus = { g, n: n ?? 0, open: open ?? '—' }; },
      },
    });
    hands.setSensitivity($('#sens').value);
    resizeHands();
    state.handMode = true;
    document.body.classList.add('hand-mode');
    controls.autoRotate = false;
    renderer.setClearColor(0x000000, 1);
    toast('Hand mode: show your hand to the camera');
    await hands.start();
  } catch (e) {
    state.handMode = false;
    document.body.classList.remove('hand-mode');
    renderer.setClearColor(SPACE, 1);
    toast(`Hand mode failed: ${e.message || e}`, '#fb7185');
  }
}
let handStatus = { g: '—', n: 0, open: '—' };

// Pointing at a button highlights it; holding clicks it.
const CLICKABLE = 'button, a, select';
function domTarget(x, y) {
  if (x < 0) return null;
  const el = document.elementFromPoint(x, y);
  return el && el !== canvas ? el.closest(CLICKABLE) : null;
}
let handHovered = null;
function handHoverDom(x, y) {
  const el = domTarget(x, y);
  if (el === handHovered) return;
  handHovered?.classList.remove('hand-hover');
  handHovered = el;
  el?.classList.add('hand-hover');
}
$('#hand-btn').addEventListener('click', toggleHandMode);
$('#sens').addEventListener('change', (e) => hands?.setSensitivity(e.target.value));
function resizeHands() { const c = $('#hands'); c.width = innerWidth; c.height = innerHeight; }

// ---------- live data ----------
async function refreshSummary() {
  try {
    const s = await api('/api/summary');
    state.summary = s;
    views.observatory.refresh(s);
    renderSources(s);
    if (current) renderInfo();
    const ok = s.sources.filter((x) => x.state === 'ok' || x.state === 'running').length;
    const newest = s.sources.map((x) => x.last_ok).filter(Boolean).sort().pop();
    const pill = $('#live');
    pill.className = `live-pill ${ok === s.sources.length ? 'ok' : 'warn'}`;
    pill.querySelector('span').textContent = `LIVE · updated ${timeAgo(newest)}`;
  } catch (e) {
    $('#live').className = 'live-pill warn';
    $('#live span').textContent = 'reconnecting…';
  }
}

function renderSources(s) {
  $('#sources').innerHTML = s.sources.map((x) => {
    const when = x.state === 'running' ? 'fetching now' : x.last_ok ? timeAgo(x.last_ok) : 'never';
    return `<div class="src ${x.state}" title="${x.message || ''}"><span><i></i>${x.name}</span><em>${when} · /${x.interval_min}m</em></div>`;
  }).join('');
}

async function refreshEvents() {
  let ev;
  try { ev = await api('/api/events?limit=40'); } catch { return; }
  const fresh = ev.filter((e) => !state.seen.has(e.key));
  ev.forEach((e) => state.seen.add(e.key));
  $('#events').innerHTML = ev.map((e) => {
    const c = KIND_COLOR[e.kind] || SOURCE_COLOR[e.source] || '#67e8f9';
    const isNew = !state.firstEvents && fresh.includes(e);
    return `<li style="--c:${c}" class="${isNew ? 'fresh' : ''}"><time>${timeAgo(e.ts)}</time><b>${e.title}</b><span>${e.detail}</span></li>`;
  }).join('');
  if (!state.firstEvents) {
    const sources = new Set(fresh.map((e) => e.source));
    sources.forEach((s) => views.observatory.pulse(s, 8));
    fresh.filter((e) => e.kind === 'blackhole' || e.kind === 'image').slice(0, 3)
      .forEach((e) => toast(`<b>${e.title}</b> · ${e.detail}`, KIND_COLOR[e.kind] || SOURCE_COLOR[e.source]));
  } else {
    // first load: a celebratory burst along every lane
    ['alerts', 'mast', 'sdo'].forEach((s, i) => setTimeout(() => views.observatory.pulse(s, 6), 600 + i * 500));
  }
  state.firstEvents = false;
}

function toast(html, color = '#67e8f9') {
  const t = document.createElement('div');
  t.className = 'toast';
  t.style.setProperty('--c', color);
  t.innerHTML = html;
  $('#toasts').append(t);
  setTimeout(() => t.remove(), 4600);
  while ($('#toasts').children.length > 3) $('#toasts').firstChild.remove();
}

// ---------- agent ----------
// The agent watches the anomaly feed, suggests what to look at, and learns from
// what you open or skip (a Thompson-sampling bandit on the server).
const agentState = { current: null, shown: new Set(), typing: null };

function typeOut(el, text) {
  clearInterval(agentState.typing);
  el.textContent = '';
  let i = 0;
  agentState.typing = setInterval(() => {
    el.textContent = text.slice(0, (i += 3));
    if (i >= text.length) clearInterval(agentState.typing);
  }, 16);
}

async function agentNext() {
  const box = $('#agent');
  box.classList.add('thinking');
  try {
    const r = await api(`/api/agent/suggestion?exclude=${[...agentState.shown].join(',')}`);
    const s = r.suggestion;
    const kinds = Object.values(r.stats.interest || {});
    $('#agent-sub').textContent = `${r.stats.new} anomalies waiting · ${kinds.length ? 'learning your taste' : 'learning'}`;
    if (!s) { agentState.current = null; box.hidden = true; return; }
    if (agentState.current?.id === s.id) return;
    agentState.current = s;
    agentState.shown.add(s.id);
    box.hidden = false;
    $('#agent-title').textContent = s.title;
    typeOut($('#agent-msg'), s.message);
    $('#agent-hyp').innerHTML = (s.hypotheses || []).map((h) => `<span>${h}</span>`).join('');
    $('#agent-why').textContent = s.why_me;
  } catch { /* offline: try again next poll */ } finally { box.classList.remove('thinking'); }
}

async function agentFeedback(action) {
  const s = agentState.current;
  if (!s) return;
  try {
    await fetch('/api/agent/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: s.id, action }) });
  } catch {}
}

async function agentLook() {
  const s = agentState.current;
  if (!s) return;
  agentFeedback('look');
  const ref = s.ref || {};
  try {
    if (ref.view === 'deep') { await switchView('deep'); await views.deep.openAt(ref.image_id, ref.u, ref.v); }
    else if (ref.view === 'bh') { await switchView('bh'); await views.bh.openOid(ref.survey, ref.oid); }
    else if (ref.view === 'sun') { await switchView('sun'); await views.sun.focusOn(ref.channel, ref.x, ref.y); renderItems(); renderInfo(); }
  } catch (e) { toast(`Couldn't open it: ${e.message}`, '#fb7185'); }
  agentState.current = null;
  setTimeout(agentNext, 12000);       // give you time to look before the next idea
  $('#agent').hidden = true;
}

$('#agent-look').addEventListener('click', agentLook);
$('#agent-skip').addEventListener('click', async () => { await agentFeedback('dismiss'); agentState.current = null; agentNext(); });
$('#agent-min').addEventListener('click', () => {
  const a = $('#agent');
  a.classList.toggle('min');
  $('#agent-min').textContent = a.classList.contains('min') ? '+' : '–';
});

// ---------- detail / trace ----------
$('#detail-close').addEventListener('click', () => $('#detail').classList.remove('open'));
$('#trace-toggle').addEventListener('click', () => {
  const t = $('#trace');
  t.classList.toggle('collapsed');
  $('#trace-toggle').textContent = t.classList.contains('collapsed') ? '+' : '–';
});

// ---------- keyboard ----------
addEventListener('keydown', async (e) => {
  if (e.target?.matches?.('input, select')) return;
  if (['1', '2', '3', '4'].includes(e.key)) switchView(VIEW_ORDER[+e.key - 1]);
  else if (e.key === 'h' || e.key === 'H') toggleHandMode();
  else if (e.key === ' ') { e.preventDefault(); setExplodeTarget(state.target > 0.5 ? 0 : 1); }
  else if (state.inspecting && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) { await inspector.next(e.key === 'ArrowRight' ? 1 : -1); renderInfo(); renderItems(); }
  else if (e.key === 'ArrowRight' && current?.next) { await current.next(); renderItems(); }
  else if (e.key === 'ArrowLeft' && current?.selectItem) { await current.selectItem(current.itemIndex - 1); renderItems(); }
  else if (e.key === 'Escape') { if (state.inspecting) closeInspect(); else $('#detail').classList.remove('open'); }
});

// ---------- tour ----------
const TOUR = [
  ['Welcome to Cosmic Watch 🔭', 'Real telescopes, live. Rubin, ZTF, JWST, Hubble and NASA\'s SDO stream data here, and computer vision analyses every new image.', null],
  ['The observatory map', 'Each planet is a telescope, an AI detector or a discovery. The glowing pulses are data flowing through the pipeline. Click a planet to dive in.', '#stage'],
  ['Live event trace', 'Every new image, black-hole candidate and pipeline run lands here as it happens.', '#trace'],
  ['Four worlds', 'Deep Field pulls a fresh JWST/Hubble image apart into the galaxies the AI found. Black Holes shows live TDE and quasar candidates. The Sun shows active regions right now.', '#views'],
  ['Explode anything', 'This slider (or Space) pulls the current scene apart in 3D: galaxies by depth, magnetic loops off the Sun, orbits around the black hole.', '.control'],
  ['Inspect & classify 🔍', 'Click any galaxy or black-hole card to dive in: a 3D model next to the real telescope pixels. Zoom as deep as you like, then classify it yourself; your labels train the next model.', null],
  ['Use your hands ✋', 'Turn on Hand mode. While your hand is in view the room fades to black and only space remains. Open hand = explode / zoom, pinch = rotate, point and hold = select (buttons too), snap = next, fist = back. Video never leaves your browser.', '#hand-btn'],
];
let tourI = 0, spot = null;
function tourShow() {
  const [title, text, sel] = TOUR[tourI];
  $('#tour').hidden = false;
  $('#tour-title').textContent = title;
  $('#tour-text').textContent = text;
  $('#tour-step').textContent = `${tourI + 1} of ${TOUR.length}`;
  $('#tour-progress').style.width = `${((tourI + 1) / TOUR.length) * 100}%`;
  $('#tour-next').textContent = tourI === TOUR.length - 1 ? 'Start exploring' : 'Next';
  spot?.classList.remove('spotlight');
  spot = sel && sel !== '#stage' ? $(sel) : null;
  spot?.classList.add('spotlight');
}
function tourEnd() {
  $('#tour').hidden = true;
  spot?.classList.remove('spotlight');
  try { localStorage.setItem('cw-tour', '1'); } catch {}
}
$('#tour-next').addEventListener('click', () => { if (++tourI >= TOUR.length) tourEnd(); else tourShow(); });
$('#tour-skip').addEventListener('click', tourEnd);
$('#tour-btn').addEventListener('click', () => { tourI = 0; tourShow(); });

// ---------- loop ----------
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
  labelRenderer.setSize(innerWidth, innerHeight);
  resizeHands();
});

const clock = new THREE.Clock();
let fps = 60, fpsAcc = 0, fpsN = 0, hudT = 0;
function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(clock.getDelta(), 0.05);
  const t = clock.elapsedTime;

  if (flight) {
    const k = ease(Math.min(1, (performance.now() - flight.t0) / flight.ms));
    camera.position.lerpVectors(flight.fromPos, flight.toPos, k);
    controls.target.lerpVectors(flight.fromT, flight.toT, k);
    if (k >= 1) flight = null;
  }
  controls.update();

  state.explode += (state.target - state.explode) * Math.min(1, dt * 5);
  if (document.activeElement !== slider) slider.value = Math.round(state.target * 100);
  $('#explode-val').textContent = `${Math.round(state.explode * 100)}%`;

  if (state.inspecting) {
    inspector.update(dt, t, state.explode);
    controls.enableRotate = inspector.fullness < 0.8;
    // zoom = camera distance: open hand / wheel / slider all drive it
    if (!flight) {
      const off = camera.position.clone().sub(controls.target);
      const want = INSPECT_FAR + (INSPECT_NEAR - INSPECT_FAR) * state.explode;
      camera.position.copy(controls.target).add(off.setLength(off.length() + (want - off.length()) * Math.min(1, dt * 4)));
    }
  } else if (current) {
    if (current.name === 'observatory') current.setExplode(state.explode);
    current.update(dt, t, state.explode);
  }
  // Hand mode: while a hand is interacting the camera fades out and full space comes back.
  const immersed = !state.handMode || document.body.classList.contains('hands-active');
  bg.rotation.y += dt * 0.004;
  bg.children[0].material.uniforms.uOpacity.value = immersed ? 0.9 : 0.35;
  bg.children[1].visible = immersed;
  bloom.strength = state.handMode ? 0.8 : 0.6;

  composer.render();
  labelRenderer.render(scene, camera);

  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = Math.round(fpsN / fpsAcc); fpsAcc = 0; fpsN = 0; }
  if ((hudT += dt) > 0.25) {
    hudT = 0;
    let particles = 0;
    scene.traverseVisible((o) => { if (o.isPoints) particles += o.geometry.attributes.position.count; });
    $('#hud').innerHTML = `<b>${fps}</b> fps · <b>${(particles / 1000).toFixed(1)}k</b> particles` +
      (state.handMode ? `<br>hands <b>${handStatus.n}</b> · cam <b>${hands?.fps ?? 0}</b> fps · open <b>${handStatus.open}</b><br>${handStatus.g}` : `<br>press <b>H</b> for hand mode`);
  }
}

// ---------- boot ----------
(async function boot() {
  const msg = $('#splash-msg');
  try {
    msg.textContent = 'Reading the live pipeline…';
    await refreshSummary();
    msg.textContent = 'Tuning in to Rubin, ZTF, JWST, Hubble and SDO…';
    await refreshEvents();
  } catch (e) {
    msg.textContent = `Offline: ${e.message}`;
  }
  frame();
  await switchView('observatory');
  setTimeout(() => $('#splash').classList.add('gone'), 250);
  // warm the other views in the background so switching is instant
  setTimeout(() => { views.deep.load().catch(() => {}); views.bh.load().catch(() => {}); views.sun.load().catch(() => {}); }, 1500);
  setInterval(refreshSummary, 20000);
  setTimeout(agentNext, 4000);
  setInterval(() => { if (!agentState.current) agentNext(); }, 30000);
  setInterval(refreshEvents, 15000);
  let seenTour = false;
  try { seenTour = localStorage.getItem('cw-tour') === '1'; } catch {}
  if (!seenTour) setTimeout(tourShow, 1600);
})();
