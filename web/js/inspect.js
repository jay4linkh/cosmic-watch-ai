// Inspect mode: dive into one object. A 3D model built from what the AI measured,
// next to the real telescope pixels it came from.
//
// The models are illustrative reconstructions: shape type, size, tilt and angle come
// from the detection; the internal structure (arms, disk, jets) is generated.
import * as THREE from 'three';
import { loadTexture, glowPointsMaterial, pointGeometry, glowSprite, label, disposeTree, fmt } from './fx.js';

// Deterministic random numbers so the same object always gets the same model.
function rng(seed) {
  let s = (seed * 2654435761) >>> 0 || 1;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) % 1e6) / 1e6; };
}
const gauss = (r) => { const u = r() || 1e-6, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

function points(n, scale = 300) {
  const g = pointGeometry(n);
  const p = new THREE.Points(g, glowPointsMaterial({ scale }));
  p.frustumCulled = false;
  return p;
}

// ---------- galaxy models ----------

function spiralGalaxy(seed, { arms = 2, wind = 2.6, tint = [0.55, 0.75, 1], bar = false } = {}) {
  const r = rng(seed), N = 14000, pts = points(N, 260);
  const barN = bar ? N * 0.12 : 0;
  const P = pts.geometry.attributes.position.array, C = pts.geometry.attributes.aColor.array, S = pts.geometry.attributes.aSize.array;
  const base = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    let x, y, z, col;
    if (i < barN) {                              // central bar: arms start at its ends
      x = (r() * 2 - 1) * 3.2; z = gauss(r) * 0.45; y = gauss(r) * 0.15;
      col = [1, 0.84, 0.62];
    } else if (i < N * 0.22) {                  // bulge
      const rr = Math.abs(gauss(r)) * 1.3;
      const th = r() * Math.PI * 2, ph = Math.acos(2 * r() - 1);
      x = rr * Math.sin(ph) * Math.cos(th); z = rr * Math.sin(ph) * Math.sin(th); y = rr * Math.cos(ph) * 0.6;
      col = [1, 0.86, 0.6];
    } else {                                    // arms
      const k = (r() * arms) | 0;
      const rad = 1 + Math.pow(r(), 0.8) * 9;
      const th = (k / arms) * Math.PI * 2 + Math.log(rad) * wind + gauss(r) * 0.28;
      x = Math.cos(th) * rad + gauss(r) * 0.25; z = Math.sin(th) * rad + gauss(r) * 0.25; y = gauss(r) * 0.18;
      const t = rad / 10;
      col = [1 - t * (1 - tint[0]), 0.9 - t * (0.9 - tint[1]), 0.75 + t * (tint[2] - 0.75)];
      if (r() < 0.035) col = [1, 0.45, 0.7];   // star-forming knots
    }
    base.set([x, y, z], i * 3); P.set([x, y, z], i * 3); C.set(col, i * 3);
    S[i] = 0.25 + r() * 0.55 + (i < N * 0.22 ? 0.35 : 0);
  }
  pts.userData.base = base;
  return pts;
}

// Two galaxies colliding: tilted disks plus tidal tails and a bridge of stripped stars.
function mergerModel(seed) {
  const g = new THREE.Group();
  const a = spiralGalaxy(seed, { arms: 2, wind: 2.2 }), b = spiralGalaxy(seed + 99, { arms: 2, wind: 3, tint: [0.75, 0.6, 1] });
  a.scale.setScalar(0.62); b.scale.setScalar(0.5);
  a.position.set(-4.2, 0, 0); b.position.set(4.6, 0.8, -1);
  a.rotation.set(0.3, 0, 0.5); b.rotation.set(1.1, 0.4, -0.3);
  const r = rng(seed + 5), N = 5000, tail = points(N, 260);
  const P = tail.geometry.attributes.position.array, C = tail.geometry.attributes.aColor.array, S = tail.geometry.attributes.aSize.array;
  for (let i = 0; i < N; i++) {
    const t = r(), side = i % 3;   // 0: bridge, 1-2: long tidal tails
    let x, y, z;
    if (side === 0) { x = -4.2 + t * 8.8; y = Math.sin(t * Math.PI) * 1.6; z = -0.5 * t; }
    else { const s = side === 1 ? -1 : 1; const ang = t * 2.4; x = s * (4.5 + Math.cos(ang) * t * 7); y = Math.sin(ang) * t * 5 * s; z = t * 2 * s; }
    P.set([x + gauss(r) * 0.35, y + gauss(r) * 0.25, z + gauss(r) * 0.35], i * 3);
    const c = [0.75 + r() * 0.25, 0.7 + r() * 0.2, 1];
    C.set(c, i * 3); S[i] = 0.2 + r() * 0.45;
  }
  g.add(a, b, tail, glowSprite(0xfff0c0, 5, 0.45));
  g.userData.spinners = [a, b];
  return g;
}

function ellipticalGalaxy(seed) {
  const r = rng(seed), N = 11000, pts = points(N, 260);
  const P = pts.geometry.attributes.position.array, C = pts.geometry.attributes.aColor.array, S = pts.geometry.attributes.aSize.array;
  const base = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const rr = Math.pow(r(), 1.8) * 8;
    const th = r() * Math.PI * 2, ph = Math.acos(2 * r() - 1);
    const x = rr * Math.sin(ph) * Math.cos(th) * 1.25, y = rr * Math.cos(ph) * 0.8, z = rr * Math.sin(ph) * Math.sin(th);
    base.set([x, y, z], i * 3); P.set([x, y, z], i * 3);
    const t = rr / 8;
    C.set([1, 0.82 - t * 0.2, 0.55 - t * 0.2], i * 3);
    S[i] = 0.25 + r() * 0.5 + (1 - t) * 0.5;
  }
  pts.userData.base = base;
  return pts;
}

function starModel() {
  const g = new THREE.Group();
  g.add(new THREE.Mesh(new THREE.SphereGeometry(1.6, 48, 32), new THREE.MeshBasicMaterial({ color: 0xfff1c9 })));
  g.add(glowSprite(0xffd27a, 12, 0.8), glowSprite(0xffffff, 5, 0.9));
  const spikeMat = new THREE.MeshBasicMaterial({ color: 0xfff4d6, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false });
  for (let k = 0; k < 4; k++) {                 // JWST-style diffraction spikes
    const s = new THREE.Mesh(new THREE.PlaneGeometry(0.12, 26), spikeMat);
    s.rotation.z = (k * Math.PI) / 3 + Math.PI / 6;
    g.add(s);
  }
  g.userData.billboard = true;
  return g;
}

// ---------- black-hole models ----------

const BH_COLOR = { TDE: 0xff4d8d, AGN: 0x8b5cf6, QSO: 0x38bdf8, Blazar: 0xfbbf24 };

function accretion(seed, { inner = 2.6, outer = 9, n = 12000, hot = 0xfff4d6, mid = 0xffa040, cool = 0xc2185b } = {}) {
  const r = rng(seed), pts = points(n, 300);
  const C = pts.geometry.attributes.aColor.array, S = pts.geometry.attributes.aSize.array;
  const rad = new Float32Array(n), ang = new Float32Array(n), yo = new Float32Array(n);
  const h = new THREE.Color(hot), m = new THREE.Color(mid), c = new THREE.Color(cool);
  for (let i = 0; i < n; i++) {
    rad[i] = inner + Math.pow(r(), 1.5) * (outer - inner); ang[i] = r() * Math.PI * 2; yo[i] = gauss(r) * 0.08 * rad[i] / 4;
    const t = (rad[i] - inner) / (outer - inner);
    const col = t < 0.35 ? h.clone().lerp(m, t / 0.35) : m.clone().lerp(c, (t - 0.35) / 0.65);
    C.set([col.r, col.g, col.b], i * 3);
    S[i] = 0.3 + r() * 0.5 + (1 - t) * 0.6;
  }
  pts.userData = { rad, ang, yo, n };
  return pts;
}

function spinDisk(pts, dt, spread = 1) {
  const { rad, ang, yo, n } = pts.userData;
  const P = pts.geometry.attributes.position.array;
  for (let i = 0; i < n; i++) {
    ang[i] += dt * 6 / Math.pow(rad[i], 1.5);
    const rr = rad[i] * spread;
    P[i * 3] = Math.cos(ang[i]) * rr; P[i * 3 + 1] = yo[i] * spread; P[i * 3 + 2] = Math.sin(ang[i]) * rr;
  }
  pts.geometry.attributes.position.needsUpdate = true;
}

function jets(seed, { length = 18, color = 0x9cc8ff, n = 3000, axis = new THREE.Vector3(0, 1, 0) } = {}) {
  const r = rng(seed + 7), pts = points(n, 300);
  const C = pts.geometry.attributes.aColor.array, S = pts.geometry.attributes.aSize.array;
  const life = new Float32Array(n), side = new Float32Array(n), off = new Float32Array(n * 2);
  const col = new THREE.Color(color);
  for (let i = 0; i < n; i++) {
    life[i] = r(); side[i] = i % 2 ? 1 : -1; off[i * 2] = gauss(r); off[i * 2 + 1] = gauss(r);
    C.set([col.r, col.g, col.b], i * 3); S[i] = 0.3 + r() * 0.6;
  }
  pts.userData = { life, side, off, n, length, axis: axis.clone().normalize() };
  return pts;
}

function flowJets(pts, dt, spread = 1) {
  const { life, side, off, n, length, axis } = pts.userData;
  const P = pts.geometry.attributes.position.array, A = pts.geometry.attributes.aAlpha.array;
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis), v = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    life[i] = (life[i] + dt * 0.35) % 1;
    const d = 2.4 + life[i] * length * spread, w = 0.15 + life[i] * 1.4;
    v.set(off[i * 2] * w, d * side[i], off[i * 2 + 1] * w).applyQuaternion(q);
    P.set([v.x, v.y, v.z], i * 3);
    A[i] = 1 - life[i];
  }
  pts.geometry.attributes.position.needsUpdate = pts.geometry.attributes.aAlpha.needsUpdate = true;
}

function tidalStream(seed) {
  const n = 3500, r = rng(seed + 3), pts = points(n, 280);
  const C = pts.geometry.attributes.aColor.array, S = pts.geometry.attributes.aSize.array;
  const life = new Float32Array(n), jit = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    life[i] = r(); jit.set([gauss(r) * 0.25, gauss(r) * 0.12, gauss(r) * 0.25], i * 3);
    const c = new THREE.Color(0xffd0a0).lerp(new THREE.Color(0xff4d8d), r() * 0.6);
    C.set([c.r, c.g, c.b], i * 3); S[i] = 0.3 + r() * 0.6;
  }
  pts.userData = { life, jit, n };
  return pts;
}

function bhModel(cls, seed) {
  const g = new THREE.Group();
  const horizon = new THREE.Mesh(new THREE.SphereGeometry(2.2, 64, 48), new THREE.MeshBasicMaterial({ color: 0x000000 }));
  const ring = new THREE.Mesh(new THREE.TorusGeometry(2.45, 0.07, 16, 128), new THREE.MeshBasicMaterial({ color: 0xffe2b0 }));
  ring.userData.billboard = true;
  const disk = accretion(seed, cls === 'QSO' ? { outer: 11, n: 16000 } : {});
  const tilt = new THREE.Group();
  tilt.rotation.x = cls === 'Blazar' ? 1.25 : 0.32;
  tilt.add(disk);
  g.add(horizon, ring, tilt, glowSprite(BH_COLOR[cls] || 0xffa040, cls === 'QSO' ? 30 : 18, cls === 'QSO' ? 0.35 : 0.2));
  const parts = { disk, tilt };
  if (cls === 'AGN' || cls === 'QSO' || cls === 'Blazar') {
    // a blazar's jet points (almost) straight at us
    const axis = cls === 'Blazar' ? new THREE.Vector3(0.12, 0.25, 1) : new THREE.Vector3(0, 1, 0);
    parts.jets = jets(seed, { length: cls === 'QSO' ? 30 : cls === 'Blazar' ? 22 : 16, color: cls === 'Blazar' ? 0xffe08a : 0x9cc8ff, axis });
    g.add(parts.jets);
  }
  if (cls === 'TDE') {
    parts.star = new THREE.Mesh(new THREE.SphereGeometry(0.75, 32, 24), new THREE.MeshBasicMaterial({ color: 0xffe6b0 }));
    parts.starGlow = glowSprite(0xffc070, 5, 0.7);
    parts.stream = tidalStream(seed);
    g.add(parts.star, parts.starGlow, parts.stream);
  }
  g.userData.parts = parts;
  g.userData.cls = cls;
  return g;
}

function animateBH(g, dt, t, e) {
  const { disk, jets: j, star, starGlow, stream } = g.userData.parts;
  spinDisk(disk, dt, 1 + e * 0.35);
  if (j) flowJets(j, dt, 1 + e * 0.5);
  if (star) {
    // star on an eccentric orbit; its torn-off gas spirals into the disk
    const a = t * 0.35, rr = 13 - Math.cos(a) * 3;
    const sp = new THREE.Vector3(Math.cos(a) * rr, Math.sin(a * 0.7) * 1.2, Math.sin(a) * rr * 0.75);
    star.position.copy(sp); starGlow.position.copy(sp);
    const { life, jit, n } = stream.userData;
    const P = stream.geometry.attributes.position.array, A = stream.geometry.attributes.aAlpha.array;
    for (let i = 0; i < n; i++) {
      life[i] = (life[i] + dt * 0.18) % 1;
      const L = life[i], ang = a - L * 4.2, rad = rr * (1 - L) + 2.8 * L;
      P[i * 3] = Math.cos(ang) * rad + jit[i * 3] * (1 + L * 3);
      P[i * 3 + 1] = Math.sin(a * 0.7) * 1.2 * (1 - L) + jit[i * 3 + 1];
      P[i * 3 + 2] = Math.sin(ang) * rad * (0.75 + 0.25 * L) + jit[i * 3 + 2] * (1 + L * 3);
      A[i] = Math.sin(Math.PI * L);
    }
    stream.geometry.attributes.position.needsUpdate = stream.geometry.attributes.aAlpha.needsUpdate = true;
  }
}

// ---------- gravitational waves: a compact binary merging ----------

function compactBody(type, r) {
  const g = new THREE.Group();
  if (type === 'bh') {
    g.add(new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), new THREE.MeshBasicMaterial({ color: 0x000000 })));
    const ring = new THREE.Mesh(new THREE.TorusGeometry(r * 1.12, r * 0.05, 12, 96), new THREE.MeshBasicMaterial({ color: 0xffd9a0 }));
    ring.userData.billboard = true;
    g.add(ring, glowSprite(0xff9a50, r * 6, 0.35));
  } else {
    g.add(new THREE.Mesh(new THREE.SphereGeometry(r * 0.75, 48, 32), new THREE.MeshBasicMaterial({ color: 0xdff4ff })));
    g.add(glowSprite(0x7dd3fc, r * 7, 0.8), glowSprite(0xffffff, r * 2.5, 0.9));
  }
  return g;
}

function rippleMaterial(color) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, wireframe: true,
    uniforms: { uTime: { value: 0 }, uAmp: { value: 0.3 }, uK: { value: 1.2 }, uColor: { value: new THREE.Color(color) } },
    vertexShader: /* glsl */`
      uniform float uTime, uAmp, uK; varying float vR; varying float vH;
      void main() {
        vec3 p = position; float r = length(p.xy); vR = r;
        float wave = uAmp * sin(uK * r - uTime) / (1.0 + 0.18 * r);
        float well = -5.0 / (1.0 + 0.06 * r * r);            // spacetime curved by the masses
        p.z = wave + well; vH = wave;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: /* glsl */`
      uniform vec3 uColor; varying float vR; varying float vH;
      void main() { float a = clamp(1.0 - vR / 34.0, 0.0, 1.0) * (0.25 + abs(vH) * 0.5); gl_FragColor = vec4(uColor * (0.6 + abs(vH)), a); }`,
  });
}

function gwModel(cls, seed) {
  const g = new THREE.Group();
  const types = cls === 'BNS' ? ['ns', 'ns'] : cls === 'NSBH' ? ['bh', 'ns'] : ['bh', 'bh'];
  const r = rng(seed);
  const m1 = 1.0, m2 = 0.55 + r() * 0.4;
  const a = compactBody(types[0], 1.7), b = compactBody(types[1], 1.7 * (types[1] === 'ns' ? 0.8 : m2));
  const merged = compactBody(types.includes('bh') ? 'bh' : 'ns', 2.1);
  const flash = glowSprite(0xffffff, 30, 0);
  const grid = new THREE.Mesh(new THREE.PlaneGeometry(70, 70, 160, 160), rippleMaterial(BH_COLOR[cls] ? 0xa78bfa : 0x8b5cf6));
  grid.rotation.x = -Math.PI / 2;
  grid.position.y = -3;
  g.add(a, b, merged, flash, grid);
  g.userData.gw = { a, b, merged, flash, grid, m1, m2, period: 9 + r() * 2 };
  return g;
}

// inspiral (orbit shrinks, speeds up: the "chirp"), merger flash, ringdown
function animateGW(g, dt, t, e) {
  const u = g.userData.gw;
  const p = (t % u.period) / u.period, P = 0.78;
  const amp = u.grid.material.uniforms.uAmp;
  u.grid.material.uniforms.uTime.value += dt * (3 + 14 * Math.pow(Math.min(p / P, 1), 3));
  if (p < P) {
    const x = p / P, d = 8 * Math.pow(1 - x, 0.25) + 0.6;
    const ang = -8 * Math.pow(1 - x, 0.625) * 6;
    const tot = u.m1 + u.m2;
    u.a.position.set(Math.cos(ang) * d * u.m2 / tot, 0, Math.sin(ang) * d * u.m2 / tot);
    u.b.position.set(-Math.cos(ang) * d * u.m1 / tot, 0, -Math.sin(ang) * d * u.m1 / tot);
    u.a.visible = u.b.visible = true;
    u.merged.visible = false;
    amp.value = (0.25 + 2.2 * Math.pow(x, 3)) * (1 + e);
    u.flash.material.opacity = 0;
  } else {
    const y = (p - P) / (1 - P);
    u.a.visible = u.b.visible = false;
    u.merged.visible = true;
    amp.value = 2.4 * Math.exp(-y * 5) * (1 + e);                 // ringdown
    u.flash.material.opacity = Math.max(0, 0.9 - y * 3);
  }
}

// ---------- inspector ----------

export function createInspector(ctx) {
  const group = new THREE.Group();
  let target = null, model = null, card = null, tags = [];
  let token = null;
  let cropStep = 0, cropLoading = false;
  let fullness = 0;                                  // 0 = card beside the model, 1 = real image fills the screen
  const tilt = new THREE.Euler(0, 0, 0, 'YXZ');      // turn the photo in your hands while in full view
  const tmpV = new THREE.Vector3(), tmpQ = new THREE.Quaternion(), side = new THREE.Vector3();
  const easeIO = (x) => x * x * (3 - 2 * x);

  // Zooming in fetches a tighter, sharper crop of the real image (galaxies only).
  async function refreshCrop(step) {
    const c = target?.crop;
    if (!c || !card || cropLoading) return;
    cropLoading = true;
    const my = token;
    const r = Math.max(0.0025, c.r * Math.pow(0.72, step));
    try {
      const tex = await loadTexture(`/api/images/${c.id}/crop?u=${c.u}&v=${c.v}&r=${r.toFixed(4)}&size=768`);
      if (my !== token || !card) { tex.dispose(); return; }
      const m = card.userData.mesh.material;
      m.map?.dispose();
      m.map = tex;
      m.needsUpdate = true;
      cropStep = step;
    } catch { /* keep the previous crop */ } finally { cropLoading = false; }
  }

  function buildModel(tg) {
    const seed = tg.seed || 1;
    if (tg.kind === 'bh') return bhModel(tg.cls, seed);
    if (tg.kind === 'gw') return gwModel(tg.cls, seed);
    if (tg.shape === 'star') return starModel();
    if (tg.shape === 'merger') return mergerModel(seed);
    const wrap = new THREE.Group();
    const inner = tg.shape === 'elliptical' ? ellipticalGalaxy(seed)
      : spiralGalaxy(seed, { arms: tg.barred ? 2 : 2 + (seed % 2), wind: 2.2 + (seed % 5) * 0.25, bar: !!tg.barred });
    wrap.add(inner);
    wrap.add(glowSprite(tg.shape === 'elliptical' ? 0xffc070 : 0xfff0c0, 7, 0.5));
    // tilt from the measured ellipticity, rotate by the measured position angle
    const e = tg.shape === 'edge-on' ? 0.92 : (tg.ellip || 0);
    const tiltAngle = Math.acos(Math.max(0.12, 1 - e));
    wrap.rotation.set(Math.PI / 2 - tiltAngle, 0, -(tg.theta || 0));
    wrap.userData.inner = inner;
    return wrap;
  }

  async function open(tg) {
    target = tg;
    const my = (token = Symbol());
    clear();
    model = buildModel(tg);
    group.add(model);
    const mTag = label(`<span class="itag-k">3D MODEL</span>${tg.modelNote}`, 'itag');
    mTag.position.set(0, -11.5, 0);
    group.add(mTag);
    tags.push(mTag);
    ctx.onInfoChange();
    if (tg.kind === 'galaxy' && tg.detId != null && !tg.cnn) {
      fetch(`/api/detections/${tg.detId}/cnn`).then((r) => r.json()).then((c) => {
        if (my === token && c.available) { tg.cnn = c; ctx.onInfoChange(); }
      }).catch(() => {});
    }
    try {
      const tex = await loadTexture(tg.imageUrl);
      if (my !== token) { tex.dispose(); return; }
      const aspect = tex.image.height / tex.image.width;
      const w = tg.kind === 'galaxy' ? 10 : 15, h = w * aspect;
      card = new THREE.Group();
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, color: 0xb0b0b0, side: THREE.DoubleSide }));
      const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(w * 1.03, h * 1.03)),
        new THREE.LineBasicMaterial({ color: tg.color || 0x67e8f9 }));
      card.add(mesh, frame);
      card.userData.mesh = mesh;
      card.userData.size = [w, h];
      cropStep = 0;
      tilt.set(0, 0, 0);
      const cTag = label(`<span class="itag-k" style="color:#4ade80">REAL DATA</span>${tg.imageNote}`, 'itag');
      cTag.position.set(0, -h / 2 - 1.6, 0);
      card.add(cTag);
      tags.push(cTag);
      card.position.set(tg.kind === 'galaxy' ? 12.5 : 15, 1, -1);
      card.userData.side = card.position.clone();
      group.add(card);
    } catch { /* image is optional; the model still shows */ }
  }

  function clear() {
    [model, card].forEach((o) => { if (o) { group.remove(o); disposeTree(o); } });
    tags.forEach((t) => { group.remove(t); t.element.remove(); });
    model = card = null; tags = [];
  }

  return {
    group,
    get target() { return target; },
    open,
    close() { token = null; clear(); target = null; },
    next(step = 1) {
      if (!target?.siblings?.length) return;
      const s = target.siblings;
      const i = (s.findIndex((x) => x.key === target.key) + step + s.length) % s.length;
      return open({ ...s[i], siblings: s });
    },
    select(i) { const s = target?.siblings; if (s?.[i]) return open({ ...s[i], siblings: s }); },
    get fullness() { return fullness; },
    // In full view, pinch/drag turns the photo itself instead of orbiting.
    tiltBy(dx, dy) { tilt.y = Math.max(-1.1, Math.min(1.1, tilt.y + dx * 3)); tilt.x = Math.max(-1.1, Math.min(1.1, tilt.x + dy * 3)); },
    // Your classification reshapes the model (the real image stays).
    reshape(shape) {
      if (!target || target.kind !== 'galaxy') return;
      target.shape = shape;
      target.modelNote = shape === 'star' ? 'foreground star · your classification' : `${shape} galaxy · your classification`;
      if (model) { group.remove(model); disposeTree(model); }
      model = buildModel(target);
      group.add(model);
      const mt = tags[0];
      if (mt) mt.element.innerHTML = `<span class="itag-k">3D MODEL</span>${target.modelNote}`;
    },
    update(dt, t, e) {
      if (!model) return;
      if (model.userData.parts) animateBH(model, dt, t, e);
      if (model.userData.gw) animateGW(model, dt, t, e);
      if (model.userData.spinners) model.userData.spinners.forEach((s, k) => { s.rotation.y += dt * (k ? -0.16 : 0.12); });
      if (model.userData.inner) {
        model.userData.inner.rotation.y += dt * 0.12;
        const base = model.userData.inner.userData.base, P = model.userData.inner.geometry.attributes.position.array;
        const k = 1 + e * 0.35;
        for (let i = 0; i < base.length; i++) P[i] = base[i] * k;
        model.userData.inner.geometry.attributes.position.needsUpdate = true;
      }
      model.traverse((o) => { if (o.userData.billboard) o.quaternion.copy(ctx.camera.quaternion); });
      if (model.userData.billboard) model.quaternion.copy(ctx.camera.quaternion);
      // Zoom past ~45 % and the real image flies to the centre and fills the screen.
      fullness = Math.min(1, Math.max(0, (e - 0.45) / 0.5));
      const f = easeIO(fullness);
      if (card) {
        const cam = ctx.camera, D = 10;
        const fwd = tmpV.set(0, 0, -1).applyQuaternion(cam.quaternion);
        const center = cam.position.clone().addScaledVector(fwd, D);
        side.copy(card.userData.side);
        card.position.lerpVectors(side, center, f);
        const [w, h] = card.userData.size;
        const fillH = 2 * D * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2), fillW = fillH * cam.aspect;
        // galaxies cover the screen; 3-panel alert stamps fit inside it
        const full = target.kind === 'galaxy' ? Math.max(fillW / w, fillH / h) * 1.02 : Math.min(fillW / w, fillH / h) * 0.96;
        card.scale.setScalar((1 + Math.min(e, 0.45) * 0.6) * (1 - f) + full * f);
        tmpQ.setFromEuler(tilt);
        const facing = cam.quaternion.clone().multiply(tmpQ);
        card.quaternion.slerp(facing, 0.18);
        if (fullness < 0.95) tilt.x *= 0.9, tilt.y *= 0.9;   // tilt only sticks in full view
        const step = Math.round(e * 6);
        if (step !== cropStep) refreshCrop(step);
      }
      model.visible = fullness < 0.92;
      model.scale.setScalar(1 - f * 0.7);
      tags.forEach((t, k) => { if (k === 0) t.element.style.opacity = 1 - f; });
    },
    items() {
      return (target?.siblings || []).map((s) => ({ label: s.short, sub: s.badge }));
    },
    get itemIndex() { return target?.siblings ? target.siblings.findIndex((x) => x.key === target.key) : -1; },
    info() {
      if (!target) return { kicker: '', title: '', sub: '', stats: [] };
      return { kicker: target.kicker, title: target.title, sub: target.sub, stats: target.stats, links: target.links };
    },
  };
}

export { BH_COLOR, fmt };
