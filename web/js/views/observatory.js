// The observatory map: telescopes -> AI detectors -> discoveries, with live data pulses.
import * as THREE from 'three';
import { glowPointsMaterial, pointGeometry, glowSprite, label, fmt, timeAgo, lerp } from '../fx.js';

const NODES = [
  // id, name, kind, color, position, view to open, live-stat function
  { id: 'rubin', name: 'Rubin · LSST', kind: 'telescope', color: 0x67e8f9, pos: [-38, 15, -2], view: 'bh',
    stat: (s) => `${fmt(s.metrics.rubin_objects)} objects · newest ${timeAgo(s.metrics.newest_rubin)}` },
  { id: 'ztf', name: 'ZTF', kind: 'telescope', color: 0x5eead4, pos: [-33, 5, 7], view: 'bh',
    stat: (s) => `${fmt(s.metrics.ztf_objects)} candidates` },
  { id: 'jwst', name: 'JWST', kind: 'telescope', color: 0xfcd34d, pos: [-39, -3, -5], view: 'deep',
    stat: (s) => `${fmt(s.metrics.jwst_images)} new images` },
  { id: 'hst', name: 'Hubble', kind: 'telescope', color: 0xd8b4fe, pos: [-32, -11, 5], view: 'deep',
    stat: (s) => `${fmt(s.metrics.hst_images)} new images` },
  { id: 'sdo', name: 'SDO · Sun', kind: 'telescope', color: 0xfdba74, pos: [-37, -21, -1], view: 'sun',
    stat: (s) => `live every 15 min` },
  { id: 'alertai', name: 'Alert Classifier', kind: 'ai', color: 0xa78bfa, pos: [-6, 10, 2], view: 'bh',
    stat: (s) => `cut-outs · difference S/N` },
  { id: 'galaxyai', name: 'Galaxy Detector', kind: 'ai', color: 0x60a5fa, pos: [-4, -6, -3], view: 'deep',
    stat: (s) => `${fmt(s.metrics.galaxies)} galaxies found` },
  { id: 'sunai', name: 'Solar Detector', kind: 'ai', color: 0xfb923c, pos: [-7, -20, 3], view: 'sun',
    stat: (s) => `${fmt(s.metrics.active_regions)} active regions` },
  { id: 'bh', name: 'Black Hole Watch', kind: 'result', color: 0xf472b6, pos: [26, 11, -2], view: 'bh', big: true,
    stat: (s) => `${fmt(s.metrics.bh_candidates)} candidates · ${fmt(s.metrics.tde)} TDE` },
  { id: 'deep', name: 'Deep Field Atlas', kind: 'result', color: 0x38bdf8, pos: [28, -6, 3], view: 'deep',
    stat: (s) => `${fmt(s.metrics.deep_images)} images · ${fmt(s.metrics.stars)} stars` },
  { id: 'weather', name: 'Space Weather', kind: 'result', color: 0xfacc15, pos: [25, -21, -3], view: 'sun',
    stat: (s) => `the Sun right now` },
];

const EDGES = [
  ['rubin', 'alertai'], ['ztf', 'alertai'], ['alertai', 'bh'],
  ['jwst', 'galaxyai'], ['hst', 'galaxyai'], ['galaxyai', 'deep'],
  ['sdo', 'sunai'], ['sunai', 'weather'],
];

// Which edges light up when an event arrives from a source.
const SOURCE_EDGES = {
  lsst: ['rubin>alertai', 'alertai>bh'], ztf: ['ztf>alertai', 'alertai>bh'],
  alerts: ['rubin>alertai', 'ztf>alertai', 'alertai>bh'],
  jwst: ['jwst>galaxyai', 'galaxyai>deep'], hst: ['hst>galaxyai', 'galaxyai>deep'],
  mast: ['jwst>galaxyai', 'hst>galaxyai', 'galaxyai>deep'],
  sdo: ['sdo>sunai', 'sunai>weather'],
};

const MAX_PULSES = 400;

function planet(node) {
  const g = new THREE.Group();
  const r = node.big ? 3.6 : node.kind === 'ai' ? 2.6 : 2.2;
  const color = new THREE.Color(node.color);

  const core = new THREE.Mesh(
    new THREE.SphereGeometry(r * 0.78, 48, 32),
    new THREE.MeshStandardMaterial({ color: color.clone().multiplyScalar(0.35), emissive: color, emissiveIntensity: 0.55, roughness: 0.4, metalness: 0.1 }),
  );
  g.add(core);
  if (node.id === 'bh') {          // a black core with a hot ring
    core.material.color.set(0x000000);
    core.material.emissive.set(0x220011);
  }

  const shell = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(r, 1)),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }),
  );
  g.add(shell);

  const ringR = r * (node.kind === 'ai' ? 1.75 : 1.5);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(ringR, ringR + 0.06, 128),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7, side: THREE.DoubleSide }),
  );
  ring.rotation.x = Math.PI / 2 - 0.35;
  g.add(ring);

  const halo = glowSprite(color, r * 6, node.big ? 0.4 : 0.22);
  g.add(halo);

  // small moons orbiting telescopes and AI nodes
  const moons = [];
  const nMoons = node.kind === 'telescope' ? 2 : node.kind === 'ai' ? 3 : 0;
  for (let i = 0; i < nMoons; i++) {
    const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.28), new THREE.MeshBasicMaterial({ color, wireframe: true }));
    m.userData = { a: Math.random() * Math.PI * 2, r: ringR + 0.8 + i * 0.7, s: 0.6 + Math.random() * 0.6 };
    g.add(m);
    moons.push(m);
  }

  g.position.set(...node.pos);
  core.userData.node = node;
  return { group: g, core, shell, ring, halo, moons, r };
}

export function createObservatory(ctx) {
  const group = new THREE.Group();
  const planets = {};
  const hoverables = [];
  let summary = null;

  for (const n of NODES) {
    const p = planet(n);
    const tag = label(`<div class="lbl-name">${n.name}</div><div class="lbl-stat" data-stat="${n.id}">…</div>`,
      `lbl lbl-${n.kind}`);
    tag.position.set(0, p.r + 2.4, 0);
    p.group.add(tag);
    p.tag = tag;
    planets[n.id] = p;
    group.add(p.group);
    hoverables.push(p.core);
  }

  // Edges as arcing curves.
  const edges = {};
  for (const [a, b] of EDGES) {
    const A = new THREE.Vector3(...NODES.find((n) => n.id === a).pos);
    const B = new THREE.Vector3(...NODES.find((n) => n.id === b).pos);
    const mid = A.clone().lerp(B, 0.5).add(new THREE.Vector3(0, 4 + Math.random() * 3, (Math.random() - 0.5) * 8));
    const curve = new THREE.QuadraticBezierCurve3(A, mid, B);
    const pts = curve.getPoints(80);
    const colA = new THREE.Color(NODES.find((n) => n.id === a).color), colB = new THREE.Color(NODES.find((n) => n.id === b).color);
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    const cols = new Float32Array(pts.length * 3);
    pts.forEach((_, i) => { const c = colA.clone().lerp(colB, i / (pts.length - 1)); cols.set([c.r, c.g, c.b], i * 3); });
    geom.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.32 }));
    group.add(line);
    edges[`${a}>${b}`] = { curve, colA, colB, line, heat: 0 };
  }

  // Pulses: a pool of glowing points that travel along edges.
  const pg = pointGeometry(MAX_PULSES);
  const pulsePts = new THREE.Points(pg, glowPointsMaterial({ scale: 520 }));
  pulsePts.frustumCulled = false;
  group.add(pulsePts);
  const pulses = [];   // {edge, t, speed, size}

  function spawn(edgeKey, size = 1.6, speed = 0.18) {
    if (pulses.length >= MAX_PULSES) pulses.shift();
    pulses.push({ edge: edgeKey, t: 0, speed: speed * (0.8 + Math.random() * 0.4), size });
    edges[edgeKey].heat = 1;
  }

  function burst(source, n = 6) {
    for (const key of SOURCE_EDGES[source] || []) {
      for (let i = 0; i < n; i++) setTimeout(() => spawn(key, 2.6, 0.3), i * 140 + (key.startsWith('alertai') || key.startsWith('galaxyai') || key.startsWith('sunai') ? 900 : 0));
    }
  }

  let ambientClock = 0;
  const tmp = new THREE.Vector3();

  return {
    name: 'observatory',
    title: 'Observatory',
    group,
    hoverables,
    home: { pos: [2, 6, 100], target: [-3, -3, 0] },
    autoRotate: true,
    explodeLabel: 'Orbit spread',

    refresh(s) {
      summary = s;
      for (const n of NODES) {
        const el = planets[n.id].tag.element.querySelector('[data-stat]');
        if (el) el.textContent = n.stat(s);
      }
    },
    pulse: burst,

    setExplode(e) {
      for (const n of NODES) {
        const p = planets[n.id];
        const base = new THREE.Vector3(...n.pos);
        p.group.position.copy(base.multiplyScalar(1 + e * 0.35));
      }
    },

    update(dt, t, e) {
      for (const n of NODES) {
        const p = planets[n.id];
        p.shell.rotation.y += dt * 0.25;
        p.shell.rotation.x += dt * 0.08;
        p.ring.rotation.z += dt * 0.15;
        p.core.material.emissiveIntensity = 0.45 + Math.sin(t * 1.6 + n.pos[0]) * 0.12;
        for (const m of p.moons) {
          m.userData.a += dt * m.userData.s;
          m.position.set(Math.cos(m.userData.a) * m.userData.r, Math.sin(m.userData.a * 0.7) * 0.6, Math.sin(m.userData.a) * m.userData.r);
          m.rotation.y += dt;
        }
      }
      // Edges follow planets when spread apart.
      for (const [key, ed] of Object.entries(edges)) {
        const [a, b] = key.split('>');
        ed.curve.v0.copy(planets[a].group.position);
        ed.curve.v2.copy(planets[b].group.position);
        const pos = ed.line.geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) { ed.curve.getPoint(i / (pos.count - 1), tmp); pos.setXYZ(i, tmp.x, tmp.y, tmp.z); }
        pos.needsUpdate = true;
        ed.heat = Math.max(0, ed.heat - dt * 0.6);
        ed.line.material.opacity = 0.28 + ed.heat * 0.5;
      }
      // Ambient flow so the map always feels alive.
      ambientClock += dt;
      if (ambientClock > 0.35) {
        ambientClock = 0;
        const keys = Object.keys(edges);
        spawn(keys[(Math.random() * keys.length) | 0], 1.2, 0.14);
      }
      const P = pg.attributes.position.array, C = pg.attributes.aColor.array, S = pg.attributes.aSize.array, A = pg.attributes.aAlpha.array;
      for (let i = pulses.length - 1; i >= 0; i--) {
        pulses[i].t += dt * pulses[i].speed;
        if (pulses[i].t >= 1) pulses.splice(i, 1);
      }
      for (let i = 0; i < MAX_PULSES; i++) {
        const p = pulses[i];
        if (!p) { S[i] = 0; continue; }
        const ed = edges[p.edge];
        ed.curve.getPoint(p.t, tmp);
        P.set([tmp.x, tmp.y, tmp.z], i * 3);
        const c = ed.colA.clone().lerp(ed.colB, p.t);
        C.set([c.r, c.g, c.b], i * 3);
        S[i] = p.size;
        A[i] = Math.sin(Math.PI * p.t);
      }
      pg.attributes.position.needsUpdate = pg.attributes.aColor.needsUpdate = true;
      pg.attributes.aSize.needsUpdate = pg.attributes.aAlpha.needsUpdate = true;
    },

    hover(obj) {
      const n = obj.userData.node;
      if (!n) return null;
      return `<b>${n.name}</b><br><span>${summary ? n.stat(summary) : ''}</span><br><em>click / point to open</em>`;
    },
    pick(obj) {
      const n = obj.userData.node;
      if (n) ctx.goTo(n.view);
    },
    items: () => [],
    info() {
      const m = summary?.metrics || {};
      return {
        kicker: 'LIVE PIPELINE',
        title: 'Cosmic Watch Observatory',
        sub: 'Five telescopes stream data into three AI detectors. Every pulse is real data flowing.',
        stats: [['Galaxies found', fmt(m.galaxies)], ['Black-hole candidates', fmt(m.bh_candidates)],
                ['Rubin objects', fmt(m.rubin_objects)], ['Active regions on the Sun', fmt(m.active_regions)]],
      };
    },
  };
}
