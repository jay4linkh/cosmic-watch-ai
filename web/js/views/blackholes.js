// Black Hole Watch: live TDE / AGN / QSO / Blazar candidates orbiting an accretion disk.
import * as THREE from 'three';
import { api, loadTexture, glowPointsMaterial, pointGeometry, glowSprite, label, fmt, timeAgo, disposeTree } from '../fx.js';

const CLASS_COLOR = { TDE: 0xff4d8d, AGN: 0x8b5cf6, QSO: 0x38bdf8, Blazar: 0xfbbf24 };
const CLASS_INFO = {
  TDE: 'Tidal disruption event: a star torn apart by a supermassive black hole',
  AGN: 'Active galactic nucleus: a supermassive black hole feeding',
  QSO: 'Quasar: an extremely luminous, distant feeding black hole',
  Blazar: 'Blazar: a black-hole jet pointed straight at Earth',
};
const FILTERS = ['All', 'TDE', 'AGN', 'QSO', 'Blazar'];
const MAX_CARDS = 16;

function accretionDisk() {
  const N = 16000;
  const g = pointGeometry(N);
  const P = g.attributes.position.array, C = g.attributes.aColor.array, S = g.attributes.aSize.array;
  const radius = new Float32Array(N), angle = new Float32Array(N), yoff = new Float32Array(N);
  const hot = new THREE.Color(0xfff4d6), warm = new THREE.Color(0xffa040), cool = new THREE.Color(0xc2185b);
  for (let i = 0; i < N; i++) {
    const r = 5.8 + Math.pow(Math.random(), 1.4) * 15;
    radius[i] = r; angle[i] = Math.random() * Math.PI * 2; yoff[i] = (Math.random() - 0.5) * 0.35 * (r / 10);
    const t = (r - 5) / 15;
    const c = t < 0.35 ? hot.clone().lerp(warm, t / 0.35) : warm.clone().lerp(cool, (t - 0.35) / 0.65);
    C.set([c.r, c.g, c.b], i * 3);
    S[i] = 0.35 + Math.random() * 0.6 + (1 - t) * 0.5;
  }
  const pts = new THREE.Points(g, glowPointsMaterial({ scale: 300 }));
  pts.frustumCulled = false;
  pts.userData = { radius, angle, yoff, N };
  return pts;
}

export function createBlackHoles(ctx) {
  const group = new THREE.Group();
  const tilt = new THREE.Group();
  tilt.rotation.x = 0.42;
  tilt.rotation.z = -0.12;
  group.add(tilt);

  const horizon = new THREE.Mesh(new THREE.SphereGeometry(4.6, 64, 48), new THREE.MeshBasicMaterial({ color: 0x000000 }));
  group.add(horizon);
  const photonRing = new THREE.Mesh(new THREE.TorusGeometry(5.0, 0.1, 16, 160),
    new THREE.MeshBasicMaterial({ color: 0xffd9a0 }));
  group.add(photonRing);
  const lens = glowSprite(0xff9a50, 26, 0.14);
  group.add(lens);
  const disk = accretionDisk();
  tilt.add(disk);

  let all = [];
  let filter = 0;
  let cards = [];
  const cardsGroup = new THREE.Group();
  group.add(cardsGroup);
  const tmpQ = new THREE.Quaternion();

  async function build() {
    cardsGroup.children.slice().forEach((c) => { cardsGroup.remove(c); disposeTree(c); });
    cards = [];
    const f = FILTERS[filter];
    const sel = all.filter((a) => f === 'All' || a.cls === f).slice(0, MAX_CARDS);
    const texes = await Promise.all(sel.map((a) => loadTexture(a.stamp_url).catch(() => null)));
    sel.forEach((a, i) => {
      if (!texes[i]) return;
      const img = texes[i].image, aspect = img.height / img.width;
      const w = 6.4, h = w * aspect;
      const card = new THREE.Group();
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: texes[i], color: 0x8f8f8f, transparent: true, opacity: 0.92, side: THREE.DoubleSide }));
      mesh.userData.alert = a;
      const border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(w * 1.04, h * 1.08)),
        new THREE.LineBasicMaterial({ color: CLASS_COLOR[a.cls] || 0xffffff }));
      const glow = glowSprite(CLASS_COLOR[a.cls] || 0xffffff, w * 1.5, 0.1);
      glow.position.z = -0.2;
      const tag = label(`<b style="color:#${(CLASS_COLOR[a.cls] || 0xffffff).toString(16).padStart(6, '0')}">${a.cls}</b> ${a.oid}`, 'card-tag');
      tag.position.y = -h / 2 - 0.9;
      card.add(glow, mesh, border, tag);
      card.userData = { phase: (i / sel.length) * Math.PI * 2, lane: i % 2, speed: 0.04 + (i % 4) * 0.005, mesh, tag, hover: 0 };
      cardsGroup.add(card);
      cards.push(card);
    });
    ctx.onInfoChange();
  }

  let hovered = null;

  const MODEL_NOTE = {
    TDE: 'a star shredded into a stream, spiralling in',
    AGN: 'feeding black hole: hot disk + jets',
    QSO: 'quasar: blazing disk + long jets',
    Blazar: 'jet pointed straight at Earth',
  };
  function alertTarget(a) {
    return {
      key: `${a.survey}:${a.oid}`,
      kind: 'bh', cls: a.cls, seed: Number(String(a.oid).replace(/\D/g, '').slice(-7)) || 1,
      color: CLASS_COLOR[a.cls] || 0xffffff,
      short: `${a.cls} ${String(a.oid).slice(-6)}`, badge: a.probability.toFixed(2),
      kicker: `${a.cls} · ${a.survey === 'lsst' ? 'RUBIN' : 'ZTF'} · INSPECT`,
      title: String(a.oid),
      sub: `${CLASS_INFO[a.cls] || ''}. Right: the real ${a.survey === 'lsst' ? 'Rubin' : 'ZTF'} cut-outs (science · template · difference). Left: a 3D model of this kind of event. Is it real? Classify it yourself.`,
      stats: [['Probability', a.probability.toFixed(2)], ['Last seen', timeAgo(a.last_seen)], ['Detections', fmt(a.n_det)],
              ['AI diff S/N', a.diff_snr != null ? String(Math.round(a.diff_snr)) : '—'],
              ['RA', a.ra.toFixed(4)], ['Dec', `${a.dec >= 0 ? '+' : ''}${a.dec.toFixed(4)}`]],
      imageUrl: a.stamp_url,
      imageNote: `${a.survey === 'lsst' ? 'Rubin' : 'ZTF'} alert cut-outs`,
      modelNote: `${MODEL_NOTE[a.cls] || 'black hole'} · illustrative`,
    };
  }

  return {
    name: 'bh',
    title: 'Black Hole Watch',
    group,
    get hoverables() { return cards.map((c) => c.userData.mesh); },
    home: { pos: [0, 10, 64], target: [0, 0, 0] },
    autoRotate: false,
    explodeLabel: 'Orbit spread',
    async load() {
      all = await api('/api/alerts?bh=true&limit=120');
      await build();
    },
    async enter() { if (!all.length) await this.load(); },
    async refresh() {},
    update(dt, t, e) {
      const { radius, angle, yoff, N } = disk.userData;
      const P = disk.geometry.attributes.position.array;
      for (let i = 0; i < N; i++) {
        angle[i] += dt * 2.2 / Math.pow(radius[i], 1.5) * 3;       // Keplerian: inner orbits faster
        const r = radius[i] * (1 + e * 0.25);
        P[i * 3] = Math.cos(angle[i]) * r; P[i * 3 + 1] = yoff[i] * (1 + e * 3); P[i * 3 + 2] = Math.sin(angle[i]) * r;
      }
      disk.geometry.attributes.position.needsUpdate = true;
      photonRing.quaternion.copy(ctx.camera.quaternion);
      const R = 30 + e * 18;
      cards.forEach((c) => {
        const u = c.userData;
        u.phase += dt * u.speed * (1 - u.hover * 0.9);
        const lane = u.lane ? 1 : -1;
        const r = R + lane * 3;
        c.position.set(Math.cos(u.phase) * r, lane * (6 + e * 10) + Math.sin(u.phase * 2) * 1.2, Math.sin(u.phase) * r * 0.5);
        c.quaternion.slerp(ctx.camera.quaternion, 0.2);
        const target = c === hovered ? 1 : 0;
        u.hover += (target - u.hover) * Math.min(1, dt * 8);
        c.scale.setScalar(1 + u.hover * 0.45);
        u.tag.element.style.opacity = Math.max(u.hover, (e - 0.35) / 0.4);
        u.mesh.material.color.setScalar(0.56 + u.hover * 0.3);
      });
    },
    hover(obj) {
      const a = obj.userData.alert;
      hovered = cards.find((c) => c.userData.mesh === obj) || null;
      if (!a) return null;
      return `<b>${a.cls} candidate</b> · p = ${a.probability.toFixed(2)}<br><span>${a.survey.toUpperCase()} ${a.oid}<br>last seen ${timeAgo(a.last_seen)}${a.diff_snr != null ? ` · AI diff S/N ${Math.round(a.diff_snr)}` : ''}</span><br><em>click / point to open</em>`;
    },
    unhover() { hovered = null; },
    pick(obj) {
      const a = obj.userData.alert;
      if (!a) return;
      // Dive in: 3D model of this kind of black-hole event next to the real cut-outs.
      const shown = cards.map((c) => c.userData.mesh.userData.alert);
      const siblings = shown.map(alertTarget);
      ctx.inspect({ ...alertTarget(a), siblings });
    },
    async openOid(survey, oid) {
      if (!all.length) await this.load();
      let a = all.find((x) => x.survey === survey && String(x.oid) === String(oid));
      if (!a) {
        const more = await api(`/api/alerts?survey=${survey}&limit=500`);
        a = more.find((x) => String(x.oid) === String(oid));
      }
      if (!a) return false;
      const sibs = [a, ...all.filter((x) => x !== a)].slice(0, 20).map(alertTarget);
      ctx.inspect({ ...sibs[0], siblings: sibs });
      return true;
    },
    items: () => FILTERS.map((f) => ({ label: f, sub: f === 'All' ? `${all.length}` : `${all.filter((a) => a.cls === f).length}` })),
    get itemIndex() { return filter; },
    selectItem(i) { filter = i; return build(); },
    next() { filter = (filter + 1) % FILTERS.length; return build(); },
    info() {
      const counts = Object.fromEntries(FILTERS.slice(1).map((f) => [f, all.filter((a) => a.cls === f).length]));
      return {
        kicker: 'RUBIN + ZTF · LIVE',
        title: 'Black Hole Watch',
        sub: 'No telescope sees a black hole directly. We catch what they do: shred stars, flare, beam jets. Each card is a real cut-out (science · template · difference).',
        stats: Object.entries(counts).map(([k, v]) => [k, fmt(v)]),
        legend: Object.entries(CLASS_COLOR).map(([k, c]) => [`${k}: ${CLASS_INFO[k].split(':')[0]}`, `#${c.toString(16).padStart(6, '0')}`]),
      };
    },
  };
}
