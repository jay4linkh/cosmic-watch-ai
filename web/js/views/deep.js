// Deep field: a newly public JWST/Hubble image that pulls apart into a 3D cloud of AI detections.
import * as THREE from 'three';
import { api, loadTexture, glowPointsMaterial, pointGeometry, label, fmt, timeAgo, disposeTree, lerp } from '../fx.js';

const WIDTH = 44;      // world units across the image
const LABEL_COLORS = {
  star: [1.0, 0.85, 0.35],
  galaxy: [0.35, 0.8, 1.0],
  'elongated galaxy': [0.95, 0.4, 1.0],
  'large galaxy': [0.4, 1.0, 0.55],
};
const N_CALLOUTS = 9;

export function createDeep(ctx) {
  const group = new THREE.Group();
  let list = [];
  let index = 0;
  let current = null;       // {plane, points, callouts, base: Float32Array, depth, meta}
  let loading = null;

  async function loadList() {
    list = await api('/api/images?source=jwst,hst&limit=40');
    // open on the richest field
    if (list.length) index = list.reduce((bi, im, i) => (im.stats.n_galaxies || 0) > (list[bi].stats.n_galaxies || 0) ? i : bi, 0);
  }

  async function show(i) {
    if (!list.length) return;
    index = (i + list.length) % list.length;
    const meta = list[index];
    const token = (loading = Symbol());
    const [tex, det] = await Promise.all([loadTexture(meta.image_url), api(`/api/images/${meta.id}/detections?limit=6000`)]);
    if (token !== loading) { tex.dispose(); return; }

    if (current) { group.remove(current.root); disposeTree(current.root); }
    const root = new THREE.Group();
    const aspect = det.height / det.width;
    const W = WIDTH, H = WIDTH * aspect;

    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(W, H),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 1, depthWrite: false }),
    );
    root.add(plane);
    const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(W, H)),
      new THREE.LineBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.35 }));
    root.add(frame);

    // Detections -> particles. Galaxies get pushed back in depth (fainter = farther),
    // stars come forward, so opening the image reveals the 3D structure the AI found.
    const rows = det.rows;
    const n = rows.length;
    const g = pointGeometry(n);
    const base = new Float32Array(n * 3), depth = new Float32Array(n);
    const col = g.attributes.aColor.array, size = g.attributes.aSize.array;
    const scores = rows.map((r) => Math.log10(Math.max(r[6], 1)));
    const smin = Math.min(...scores), smax = Math.max(...scores);
    rows.forEach((r, i) => {
      const lbl = det.labels[r[5]];
      const x = (r[0] - 0.5) * W, y = (0.5 - r[1]) * H;
      base.set([x, y, 0], i * 3);
      const bright = (scores[i] - smin) / Math.max(smax - smin, 1e-6);
      depth[i] = lbl === 'star' ? 2 + bright * 7 + Math.random() * 2 : -(5 + (1 - bright) * 30 + Math.random() * 4);
      const c = LABEL_COLORS[lbl] || [1, 1, 1];
      col.set(c, i * 3);
      size[i] = Math.min(4.2, 0.55 + r[2] * W * 3.2 + bright * 1.2);
    });
    const points = new THREE.Points(g, glowPointsMaterial({ scale: 260 }));
    points.frustumCulled = false;
    root.add(points);

    // Callouts with leader lines on the biggest galaxies.
    const callouts = [];
    const ranked = rows.map((r, i) => [i, r]).filter(([, r]) => det.labels[r[5]].includes('galaxy'))
      .sort((a, b) => b[1][6] - a[1][6]).slice(0, N_CALLOUTS);
    ranked.forEach(([i, r], k) => {
      const lbl = det.labels[r[5]];
      const side = r[0] < 0.5 ? -1 : 1;
      const tag = label(`<span class="callout-k">${String(k + 1).padStart(2, '0')}</span>${lbl}`, 'callout');
      tag.element.style.setProperty('--c', `rgb(${(LABEL_COLORS[lbl] || [1, 1, 1]).map((v) => v * 255 | 0).join(',')})`);
      const lineGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
      const line = new THREE.Line(lineGeom, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
      root.add(tag, line);
      callouts.push({ i, tag, line, side, offset: new THREE.Vector3(side * (W * 0.62 - Math.abs(base[i * 3]) * 0.25), (k - N_CALLOUTS / 2) * 2.6, 0) });
    });

    group.add(root);
    current = { root, plane, frame, points, base, depth, callouts, meta, det, W, H };
    ctx.onInfoChange();
  }

  let lastE = -1;
  function applyExplode(e) {
    if (!current) return;
    const { points, base, depth, callouts, plane, frame } = current;
    const P = points.geometry.attributes.position.array;
    const spread = 1 + e * 0.18;
    for (let i = 0; i < depth.length; i++) {
      P[i * 3] = base[i * 3] * spread;
      P[i * 3 + 1] = base[i * 3 + 1] * spread;
      P[i * 3 + 2] = depth[i] * e;
    }
    points.geometry.attributes.position.needsUpdate = true;
    plane.material.opacity = 1 - e * 0.82;
    plane.position.z = -e * 38;
    frame.position.z = plane.position.z;
    frame.material.opacity = 0.35 + e * 0.3;
    const show = Math.min(1, Math.max(0, (e - 0.3) / 0.3));
    for (const c of callouts) {
      const p = new THREE.Vector3(P[c.i * 3], P[c.i * 3 + 1], P[c.i * 3 + 2]);
      const anchor = new THREE.Vector3(c.side * current.W * 0.56, c.offset.y, p.z);
      c.tag.position.copy(anchor);
      c.line.geometry.setFromPoints([p, anchor]);
      c.line.material.opacity = show * 0.55;
      c.tag.element.style.opacity = show;
    }
  }

  // Everything Inspect mode needs to show one detection: model shape + the real pixels.
  function objectTarget(i, rank) {
    const { det, meta } = current;
    const r = det.rows[i], lbl = det.labels[r[5]];
    const ellip = 1 - r[3] / Math.max(r[2], 1e-6);
    const shape = lbl === 'star' ? 'star' : (lbl === 'galaxy' && ellip < 0.2) ? 'elliptical' : 'spiral';
    const sizePx = r[2] * det.width;
    const c = LABEL_COLORS[lbl] || [1, 1, 1];
    return {
      key: `${meta.id}:${i}`,
      kind: 'galaxy', shape, ellip, theta: r[4], seed: i + 1, aiLabel: lbl,
      crop: { id: meta.id, u: r[0].toFixed(4), v: r[1].toFixed(4), r: Math.min(0.08, Math.max(0.012, r[2] * 5)) },
      color: new THREE.Color(...c).getHex(),
      short: `#${rank} ${lbl}`, badge: fmt(Math.round(r[6])),
      kicker: `${meta.source.toUpperCase()} · ${meta.target} · INSPECT`,
      title: `${lbl[0].toUpperCase()}${lbl.slice(1)} #${rank}`,
      sub: `Found by the AI in this ${meta.source.toUpperCase()} image. Right: the real pixels at full resolution. Left: a 3D model built from its measured size, shape and angle. Zoom in as deep as you like, then classify it yourself.`,
      stats: [['Brightness (flux)', fmt(Math.round(r[6]))], ['Size', `${sizePx.toFixed(1)} px`],
              ['Ellipticity', ellip.toFixed(2)], ['Angle', `${Math.round((r[4] * 180) / Math.PI)}°`],
              ['Position', `${(r[0] * 100).toFixed(1)}%, ${(r[1] * 100).toFixed(1)}%`], ['Model', shape]],
      imageUrl: `/api/images/${meta.id}/crop?u=${r[0].toFixed(4)}&v=${r[1].toFixed(4)}&r=${Math.min(0.08, Math.max(0.012, r[2] * 5)).toFixed(4)}`,
      imageNote: `${meta.source.toUpperCase()} ${meta.instrument} · full-resolution close-up`,
      modelNote: shape === 'star' ? 'foreground star (illustrative)' : `${shape} galaxy · illustrative, from measured shape`,
    };
  }

  return {
    name: 'deep',
    title: 'Deep Field',
    group,
    get hoverables() { return current ? [current.points] : []; },
    home: { pos: [0, 0, 70], target: [0, 0, -10] },
    autoRotate: false,
    explodeLabel: 'Exploded view',
    async load() { await loadList(); await show(index); },
    async enter() { if (!list.length) await this.load(); },
    update(dt, t, e) {
      if (Math.abs(e - lastE) > 1e-4 || current?._fresh !== false) { applyExplode(e); lastE = e; if (current) current._fresh = false; }
      if (current) {
        current.points.material.uniforms.uTime.value = t;
        current.points.material.uniforms.uScale.value = ctx.handMode ? 380 : 260;   // bigger, brighter over the camera
      }
    },
    hover(obj, hit) {
      if (!current || hit?.index == null) return null;
      const r = current.det.rows[hit.index];
      return `<b>${current.det.labels[r[5]]}</b><br><span>flux ${fmt(Math.round(r[6]))} · size ${(r[2] * current.det.width).toFixed(1)} px</span>`;
    },
    pick(obj, hit) {
      if (!current || hit?.index == null) return;
      // Inspect the clicked source; snap cycles through the field's biggest galaxies.
      const rows = current.det.rows, labels = current.det.labels;
      const order = rows.map((r, i) => i).filter((i) => labels[rows[i][5]].includes('galaxy'))
        .sort((a, b) => rows[b][6] - rows[a][6]).slice(0, 12);
      if (!order.includes(hit.index)) order.unshift(hit.index);
      const siblings = order.map((i, k) => objectTarget(i, k + 1));
      ctx.inspect({ ...siblings[order.indexOf(hit.index)], siblings });
    },
    items: () => list.map((im) => ({ label: `${im.source.toUpperCase()} · ${im.target}`, sub: `${im.stats.n_galaxies || 0} galaxies` })),
    get itemIndex() { return index; },
    selectItem(i) { return show(i); },
    next() { return show(index + 1); },
    info() {
      if (!current) return { kicker: 'DEEP FIELD', title: 'Loading…', sub: '', stats: [] };
      const m = current.meta, s = m.stats;
      return {
        kicker: `${m.source.toUpperCase()} · ${m.instrument}`,
        title: m.target,
        sub: `Public since ${m.obs_time?.slice(0, 16).replace('T', ' ')} UTC (${timeAgo(m.obs_time)}). Filter ${m.filter}. Open your hand to pull it apart.`,
        stats: [['Sources detected', fmt(s.n_sources)], ['Galaxies', fmt(s.n_galaxies)], ['Stars', fmt(s.n_stars)],
                ...Object.entries(s.counts || {}).filter(([k]) => k !== 'star' && k !== 'galaxy').map(([k, v]) => [k, fmt(v)])],
        legend: Object.entries(LABEL_COLORS).map(([k, c]) => [k, `rgb(${c.map((v) => v * 255 | 0).join(',')})`]),
      };
    },
  };
}
