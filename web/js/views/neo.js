// Near Earth: the ISS live, asteroids flying past this week (NASA/JPL), fireballs that hit the atmosphere,
// and the Sentry list of objects with a small chance of impact.
import * as THREE from 'three';
import { api, loadTexture, glowSprite, label, fmt, timeAgo, disposeTree, esc } from '../fx.js';

const RE = 5;                                         // Earth radius in scene units
const LD = (ld) => RE * 1.5 + 18 * Math.sqrt(ld);     // distances compressed: square-root scale
const FILTERS = ['Next 60 days', 'Last 7 days', 'All'];

function latLon(lat, lon, r) {
  const phi = THREE.MathUtils.degToRad(90 - lat), theta = THREE.MathUtils.degToRad(lon + 180);
  return new THREE.Vector3(-r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
}
const hashDir = (s) => {
  let h = 2166136261;
  for (const c of String(s)) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0;
  const a = (h % 3600) / 3600 * Math.PI * 2, b = ((h >>> 12) % 1000) / 1000 * 0.9 - 0.45;
  return new THREE.Vector3(Math.cos(a) * Math.cos(b), Math.sin(b), Math.sin(a) * Math.cos(b));
};
const jpl = (s) => Date.parse(s.replace(/(\d{4})-(\w{3})-(\d{2}) (\d{2}):(\d{2})/, '$2 $3 $1 $4:$5 UTC'));

export function createNeo(ctx) {
  const group = new THREE.Group();
  const earth = new THREE.Group();
  group.add(earth);
  earth.rotation.z = THREE.MathUtils.degToRad(23.4);
  const globe = new THREE.Mesh(new THREE.SphereGeometry(RE, 96, 64), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  loadTexture('/textures/2k_earth_daymap.jpg').then((t) => { globe.material.map = t; globe.material.needsUpdate = true; }).catch(() => {});
  const spin = new THREE.Group();                          // turns with the real time of day
  spin.add(globe);
  earth.add(spin);
  group.add(glowSprite(0x60a5fa, RE * 3.2, 0.25));
  const moonOrbit = new THREE.Mesh(new THREE.TorusGeometry(LD(1), 0.04, 8, 200), new THREE.MeshBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.5 }));
  moonOrbit.rotation.x = Math.PI / 2;
  const moon = new THREE.Mesh(new THREE.SphereGeometry(RE * 0.27, 48, 32), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  loadTexture('/textures/2k_moon.jpg').then((t) => { moon.material.map = t; moon.material.needsUpdate = true; }).catch(() => {});
  const moonTag = label('Moon · 1 lunar distance (384,400 km)', 'callout');
  group.add(moonOrbit, moon, moonTag);

  const iss = new THREE.Group();
  iss.add(glowSprite(0xffffff, 1.6, 1), glowSprite(0x67e8f9, 3.2, 0.5));
  const issTag = label('ISS', 'callout');
  iss.add(issTag);
  issTag.position.y = 1.2;
  spin.add(iss);
  let issData = null, issTimer = null;

  let data = null, filter = 0, objs = [], hovered = null;
  const objGroup = new THREE.Group();
  group.add(objGroup);
  const fireGroup = new THREE.Group();
  spin.add(fireGroup);

  function placeISS() {
    if (!issData) return;
    iss.position.copy(latLon(issData.lat, issData.lon, RE * 1.18));      // altitude exaggerated ~4x
    issTag.element.innerHTML = `<b>ISS</b> · ${Math.round(issData.v_kmh).toLocaleString()} km/h · ${Math.round(issData.alt_km)} km up`;
  }
  async function pollISS() { try { issData = await api('/api/iss'); placeISS(); } catch { /* keep the last fix */ } }

  function build() {
    objGroup.children.slice().forEach((c) => { objGroup.remove(c); disposeTree(c); });
    fireGroup.children.slice().forEach((c) => { fireGroup.remove(c); disposeTree(c); });
    objs = [];
    if (!data) return;
    const now = Date.now();
    const list = data.approaches.filter((a) => filter === 2 || (filter === 0 ? jpl(a.time) >= now : jpl(a.time) < now));
    list.forEach((a) => {
      const past = jpl(a.time) < now;
      const color = a.dist_ld <= 1 ? 0xf472b6 : past ? 0x94a3b8 : 0xfbbf24;
      const dir = hashDir(a.des), pos = dir.clone().multiplyScalar(LD(a.dist_ld));
      const o = new THREE.Group();
      const size = 0.35 + Math.min(1.4, Math.log10(1 + (a.size_m?.[1] || 10)) * 0.45);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(size * 0.45, 16, 12), new THREE.MeshBasicMaterial({ color }));
      dot.userData.neo = a;
      o.add(dot, glowSprite(color, size * 4, 0.55));
      // a short straight path through the closest point (direction illustrative)
      const along = new THREE.Vector3(0, 1, 0).cross(dir).normalize().multiplyScalar(6);
      const path = new THREE.Line(new THREE.BufferGeometry().setFromPoints([along.clone().negate(), along]),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 }));
      o.add(path);
      const tag = label(`${esc(a.name.replace(/[()]/g, ''))} · ${a.dist_ld.toFixed(2)} LD`, 'card-tag');
      tag.position.y = -1.4;
      o.add(tag);
      o.position.copy(pos);
      o.userData = { dot, tag, a, hover: 0 };
      objGroup.add(o);
      objs.push(o);
    });
    (data.fireballs || []).slice(0, 15).forEach((f) => {
      if (f.lat == null || f.lon == null) return;
      const m = glowSprite(0xff7849, 0.6 + Math.min(2.5, Math.log10(1 + (f.energy_kt || 0.01) * 100)), 0.9);
      m.position.copy(latLon(f.lat, f.lon, RE * 1.01));
      fireGroup.add(m);
    });
    ctx.onInfoChange();
  }

  function detail(a) {
    const when = jpl(a.time), past = when < Date.now();
    const size = a.size_m ? (a.size_m[0] === a.size_m[1] ? `${a.size_m[0]} m (measured)` : `${a.size_m[0]}–${a.size_m[1]} m (from brightness)`) : 'unknown';
    const compare = a.size_m ? (a.size_m[1] < 20 ? 'about a bus or a house: it would burn up or explode high in the atmosphere'
      : a.size_m[1] < 60 ? 'about a building: the size of the 1908 Tunguska impactor' : 'a stadium or larger: a city-scale threat if it ever hit') : '';
    ctx.showDetail(`<div class="kicker">NEAR-EARTH ASTEROID · NASA/JPL</div><h3>${esc(a.name)}</h3>
      <p>${past ? 'Passed' : 'Passes'} <b>${a.dist_ld.toFixed(2)} lunar distances</b> (${fmt(a.dist_km)} km) from Earth on <b>${a.time} UTC</b>${past ? '' : ` (in ${Math.max(0, Math.round((when - Date.now()) / 864e5))} days)`}.</p>
      <div class="stats"><div class="stat"><b>${size}</b><span>size</span></div><div class="stat"><b>${a.v_kms} km/s</b><span>speed vs Earth</span></div>
      <div class="stat"><b>${a.h ?? '—'}</b><span>abs. magnitude H</span></div><div class="stat"><b>${a.dist_ld <= 1 ? 'inside' : 'outside'}</b><span>the Moon's orbit</span></div></div>
      ${compare ? `<p>${compare}.</p>` : ''}<p class="feed-dim small">No impact risk on this pass: JPL computes the orbit precisely. Direction in the 3D view is illustrative; distance is to scale (square-root compressed).</p>`);
  }

  return {
    name: 'neo',
    title: 'Near Earth',
    group,
    get hoverables() { return objs.map((o) => o.userData.dot); },
    home: { pos: [0, 30, 95], target: [0, 0, 0] },
    autoRotate: false,
    explodeLabel: 'Spread the flybys',
    async load() {
      data = await api('/api/neo');
      build();
      await pollISS();
    },
    async enter() {
      if (!data) await this.load();
      clearInterval(issTimer);
      issTimer = setInterval(() => { if (ctx.currentView?.() === 'neo') pollISS(); }, 5000);
    },
    async refreshPanel() { try { data = await api('/api/neo'); } catch { /* keep */ } },
    update(dt, t, e) {
      const d = new Date();
      spin.rotation.y = ((d.getUTCHours() * 3600 + d.getUTCMinutes() * 60 + d.getUTCSeconds()) / 86400) * Math.PI * 2 + Math.PI;
      const ang = t * 0.03;
      moon.position.set(Math.cos(ang) * LD(1), 0, Math.sin(ang) * LD(1));
      moonTag.position.copy(moon.position).add(new THREE.Vector3(0, 2.4, 0));
      objs.forEach((o) => {
        const u = o.userData;
        o.scale.setScalar(1 + e * 0.6 + u.hover * 0.6);
        u.hover += ((o === hovered ? 1 : 0) - u.hover) * Math.min(1, dt * 8);
        u.tag.element.style.opacity = Math.max(u.hover, 0.35 + e * 0.65);
      });
    },
    hover(obj) {
      const a = obj.userData.neo;
      hovered = objs.find((o) => o.userData.dot === obj) || null;
      if (!a) return null;
      return `<b>${esc(a.name)}</b><br><span>${a.dist_ld.toFixed(2)} lunar distances · ${a.time} UTC<br>${a.size_m ? `${a.size_m[0]}–${a.size_m[1]} m` : ''} · ${a.v_kms} km/s</span><br><em>click / point for details</em>`;
    },
    unhover() { hovered = null; },
    pick(obj) { const a = obj.userData.neo; if (a) detail(a); },
    focus(des) {
      const a = data?.approaches.find((x) => x.des === des);
      if (a) detail(a);
      return !!a;
    },
    panel() {
      if (!data) return '<div class="feed-k">NEAR EARTH</div><p class="feed-dim">Loading…</p>';
      const now = Date.now();
      const next = data.approaches.filter((a) => jpl(a.time) >= now).sort((a, b) => jpl(a.time) - jpl(b.time)).slice(0, 6);
      const close = data.approaches.filter((a) => a.dist_ld <= 1);
      return `
        <div class="feed-k">NEAR EARTH · NASA/JPL <span class="feed-time">${timeAgo(data.fetched_at)}</span></div>
        <div class="sw-row">
          <div class="sw-big"><b>${data.approaches.length}</b><span>flybys within 10 LD</span></div>
          <div class="sw-big ${close.length ? 'warn' : ''}"><b>${close.length}</b><span>inside the Moon's orbit</span></div>
          <div class="sw-big"><b>${issData ? Math.round(issData.alt_km) : '—'}</b><span>km · ISS altitude</span></div>
        </div>
        <div class="feed-sub">Next close approaches</div>
        <div class="feed-list">${next.map((a) => `<button class="feed-item" data-act="neo" data-des="${esc(a.des)}">
          <span class="chip" style="--c:${a.dist_ld <= 1 ? '#f472b6' : '#fbbf24'}">${a.dist_ld.toFixed(2)} LD</span>
          <b>${esc(a.name.replace(/[()]/g, ''))}</b><small>${a.time} · ${a.size_m ? `${a.size_m[0]}–${a.size_m[1]} m` : ''}</small></button>`).join('')}</div>
        <div class="feed-sub">Sentry: highest impact risk (next ~100 years)</div>
        <table class="feed-table"><tr><th>Object</th><th>Years</th><th>Chance</th><th>Size</th></tr>
          ${data.sentry.top.slice(0, 6).map((s) => `<tr><td>${esc(s.name.replace(/[()]/g, ''))}</td><td>${esc(s.years)}</td><td>1 in ${fmt(Math.round(1 / s.p_impact))}</td><td>${s.size_m ? `${s.size_m} m` : '—'}</td></tr>`).join('')}</table>
        <p class="feed-dim small">${fmt(data.sentry.count)} objects on the list; all are rated Torino 0 (no hazard) unless shown otherwise.</p>
        <div class="feed-sub">Fireballs: meteors that exploded in the atmosphere (orange on the globe)</div>
        <div class="feed-line feed-dim">${(data.fireballs || []).slice(0, 4).map((f) => `${f.time.slice(0, 10)} · ${f.energy_kt ?? '?'} kt${f.lat != null ? ` · ${Math.abs(f.lat).toFixed(0)}°${f.lat >= 0 ? 'N' : 'S'} ${Math.abs(f.lon).toFixed(0)}°${f.lon >= 0 ? 'E' : 'W'}` : ''}`).join('<br>')}</div>
        <p class="feed-dim small">Distances in 3D are square-root scaled so 0.3 and 10 lunar distances fit on one screen; directions are illustrative.</p>`;
    },
    onPanel(act, el) { if (act === 'neo') this.focus(el.dataset.des); },
    items: () => FILTERS.map((f, i) => ({ label: f, sub: data ? `${data.approaches.filter((a) => i === 2 || (i === 0 ? jpl(a.time) >= Date.now() : jpl(a.time) < Date.now())).length}` : '' })),
    get itemIndex() { return filter; },
    selectItem(i) { filter = i; build(); },
    next() { filter = (filter + 1) % FILTERS.length; build(); },
    info() {
      return {
        kicker: 'NASA/JPL · ISS · LIVE',
        title: 'Near Earth',
        sub: 'Earth in real time: the ISS where it is right now, every asteroid flying within 10 lunar distances this week and next, and fireballs that lit up the sky. The ring is the Moon\'s orbit.',
        stats: data ? [['Flybys', fmt(data.approaches.length)], ['Inside Moon orbit', fmt(data.approaches.filter((a) => a.dist_ld <= 1).length)],
                       ['Closest', `${Math.min(...data.approaches.map((a) => a.dist_ld)).toFixed(2)} LD`], ['Sentry list', fmt(data.sentry.count)]] : [],
        legend: [['Inside the Moon\'s orbit', '#f472b6'], ['Upcoming flyby', '#fbbf24'], ['Past week', '#94a3b8'], ['Fireball', '#ff7849'], ['ISS', '#67e8f9']],
      };
    },
  };
}
