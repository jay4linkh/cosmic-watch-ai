// The Sun: the latest SDO image wrapped onto a sphere, with AI-detected regions rising off the surface.
import * as THREE from 'three';
import { api, loadTexture, glowPointsMaterial, pointGeometry, glowSprite, label, fmt, timeAgo, disposeTree, sparkline, esc } from '../fx.js';

const R = 14;
const CHANNEL = {
  '0171': { name: 'AIA 171 Å', desc: 'Quiet corona and coronal loops (~600 000 °C)', glow: 0xffc94d },
  '0193': { name: 'AIA 193 Å', desc: 'Hot corona and dark coronal holes (~1.2 million °C)', glow: 0xe8b07a },
  '0304': { name: 'AIA 304 Å', desc: 'Chromosphere and prominences (~50 000 °C)', glow: 0xff5a36 },
  '0131': { name: 'AIA 131 Å', desc: 'Flaring plasma (~10 million °C): flares light up here first', glow: 0x2dd4bf },
  '0094': { name: 'AIA 94 Å', desc: 'Hot flare loops (~6 million °C)', glow: 0x4ade80 },
  HMIIC: { name: 'HMI visible light', desc: 'The photosphere: where sunspots show', glow: 0xffa040 },
};
const ORDER = ['0171', '0193', '0304', '0131', '0094', 'HMIIC'];
const MAX_TAGS = 7;      // label only the biggest regions; the rest get rings

// Front hemisphere = the SDO disk, projected orthographically; the back gets a dimmed mirror.
function sunMaterial(tex, center, radius, glow) {
  return new THREE.ShaderMaterial({
    uniforms: { map: { value: tex }, uCenter: { value: center }, uRadius: { value: radius }, uGlow: { value: new THREE.Color(glow) } },
    vertexShader: /* glsl */`
      varying vec3 vN; varying vec3 vViewN; varying vec3 vViewPos;
      void main() {
        vN = normalize(position);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vViewN = normalize(normalMatrix * normal);
        vViewPos = mv.xyz;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D map; uniform vec2 uCenter; uniform vec2 uRadius; uniform vec3 uGlow;
      varying vec3 vN; varying vec3 vViewN; varying vec3 vViewPos;
      void main() {
        vec2 uv = uCenter + vec2(vN.x, vN.y) * uRadius * 0.985;
        vec3 c = texture2D(map, uv).rgb;
        if (vN.z < 0.0) c *= 0.45;              // far side: dim stand-in
        float fres = pow(1.0 - abs(dot(normalize(-vViewPos), vViewN)), 2.5);
        gl_FragColor = vec4(c * 0.78 + uGlow * fres * 0.45, 1.0);
      }`,
  });
}

function coronaMaterial(color) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.BackSide, blending: THREE.AdditiveBlending,
    uniforms: { uColor: { value: new THREE.Color(color) }, uStrength: { value: 1 } },
    vertexShader: /* glsl */`
      varying vec3 vN; varying vec3 vV;
      void main() { vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: /* glsl */`
      uniform vec3 uColor; uniform float uStrength; varying vec3 vN; varying vec3 vV;
      void main() { float i = pow(max(0.72 - dot(vN, vV), 0.0), 3.0); gl_FragColor = vec4(uColor, clamp(i,0.0,1.0) * uStrength * 0.55); }`,
  });
}

// ---------- space weather (NOAA SWPC + SOHO LASCO) ----------
const KP_TEXT = (kp) => (kp == null ? '—' : kp < 4 ? 'quiet' : kp < 5 ? 'active' : `G${Math.min(5, Math.floor(kp) - 4)} storm`);
const AURORA = (kp) => (kp == null ? '' : kp < 4 ? 'Aurora only near the poles.' : kp < 5 ? 'Aurora possible at high latitudes (Norway, Alaska, Iceland).'
  : kp < 7 ? 'Aurora may reach Scotland, Canada, northern US and New Zealand.' : 'Aurora possible far from the poles: look north (or south)!');
const FLARE_LINES = [[1e-7, 'B'], [1e-6, 'C'], [1e-5, 'M'], [1e-4, 'X']];

function spaceWeatherPanel(sw) {
  if (!sw) return '<div class="feed-k">SPACE WEATHER</div><p class="feed-dim">Waiting for NOAA…</p>';
  const x = sw.xrays || {}, flares = sw.flares || [], kp = sw.kp?.now, w = sw.wind?.now || {}, reg = sw.regions || {};
  const count = {};
  flares.forEach((f) => { count[f.class[0]] = (count[f.class[0]] || 0) + 1; });
  const big = [...flares].sort((a, b) => (b.flux || 0) - (a.flux || 0)).slice(0, 3);
  const pr = reg.probabilities || {};
  const lasco = sw.lasco || {};
  const now = x.now || {};
  const level = (now.class || '?')[0];
  return `
    <div class="feed-k">SPACE WEATHER · LIVE <span class="feed-time">${timeAgo(sw.fetched_at)}</span></div>
    <div class="sw-row">
      <div class="sw-big flare-${level}"><b>${now.class || '—'}</b><span>X-rays now</span></div>
      <div class="sw-big"><b>${kp?.kp != null ? kp.kp.toFixed(1) : '—'}</b><span>Kp · ${KP_TEXT(kp?.kp)}</span></div>
      <div class="sw-big ${w.bz != null && w.bz < -5 ? 'warn' : ''}"><b>${w.speed != null ? Math.round(w.speed) : '—'}</b><span>km/s wind</span></div>
    </div>
    <div class="feed-sub">GOES X-ray flux, 24 h (flare classes A·B·C·M·X)</div>
    ${sparkline(x.series, { log: true, min: 1e-8, max: Math.max(1e-5, ...(x.series || []).map((p) => p[1] || 0)), color: '#fbbf24', lines: FLARE_LINES })}
    <div class="feed-line">24 h peak <b>${x.peak_24h?.class || '—'}</b> · flares this week:
      ${['X', 'M', 'C', 'B'].map((k) => `<span class="flare-chip flare-${k}">${count[k] || 0} ${k}</span>`).join('')}</div>
    ${big.length ? `<div class="feed-line feed-dim">Biggest: ${big.map((f) => `${f.class} (${f.max?.slice(5, 16).replace('T', ' ')})`).join(' · ')}</div>` : ''}
    <div class="feed-sub">Solar wind at L1 · speed (km/s) · magnetic field Bz ${w.bz != null ? `<b class="${w.bz < 0 ? 'neg' : ''}">${w.bz.toFixed(1)} nT</b>` : ''}</div>
    ${sparkline(sw.wind?.speed_series, { color: '#67e8f9', h: 42 })}
    ${sparkline(sw.wind?.bz_series, { color: '#f472b6', h: 34, zero: 0 })}
    <div class="feed-line feed-dim">Density ${w.density != null ? w.density.toFixed(1) : '—'} p/cm³ · ${w.bz != null && w.bz < 0 ? 'Bz south: the wind can couple to Earth\'s field (aurora-friendly).' : 'Bz north: Earth\'s field is mostly shielded.'} ${AURORA(kp?.kp)}</div>
    <div class="feed-sub">Flare odds, next 24 h (NOAA)</div>
    <div class="sw-odds"><span>C <b>${pr.c_class_1_day ?? '—'}%</b></span><span>M <b>${pr.m_class_1_day ?? '—'}%</b></span><span>X <b>${pr.x_class_1_day ?? '—'}%</b></span></div>
    ${(reg.regions || []).length ? `<table class="feed-table"><tr><th>Region</th><th>Where</th><th>Spots</th><th>M %</th><th>X %</th></tr>
      ${reg.regions.slice(0, 6).map((r) => `<tr><td>AR ${r.region}</td><td>${r.location}</td><td>${r.spots} · ${esc(r.spot_class)}</td><td>${r.p_m ?? '—'}</td><td>${r.p_x ?? '—'}</td></tr>`).join('')}</table>` : ''}
    <div class="feed-sub">SOHO LASCO coronagraphs: the corona out to 30 Sun-widths. Click for the movie.</div>
    <div class="lasco-row">${['c2', 'c3'].map((cam) => {
      const l = lasco[cam];
      if (!l) return '';
      const c = l.cme || {};
      const verdict = c.area_frac >= 0.02 ? `<b class="warn">possible CME, PA ${c.angle_deg}°</b>` : 'no new bright front';
      return `<button class="lasco-btn" data-act="lasco" data-cam="${cam}"><img src="${l.url}" alt="LASCO ${cam}"><span>${cam.toUpperCase()} · ${timeAgo(l.time)}<br>${verdict}</span></button>`;
    }).join('')}</div>
    <p class="feed-dim small">Our CV compares consecutive coronagraph frames (running difference) to catch CMEs.</p>`;
}

export function createSun(ctx) {
  const group = new THREE.Group();
  let channels = [];
  let sw = null;
  let index = 0;
  let cur = null;
  const tmp = new THREE.Vector3();

  // A slowly expanding shell of solar-wind particles.
  const WIND = 2500;
  const wg = pointGeometry(WIND);
  const wDir = new Float32Array(WIND * 3), wPhase = new Float32Array(WIND);
  for (let i = 0; i < WIND; i++) {
    const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u);
    wDir.set([s * Math.cos(th), u, s * Math.sin(th)], i * 3);
    wPhase[i] = Math.random();
    wg.attributes.aSize.array[i] = 0.3 + Math.random() * 0.7;
  }
  const wind = new THREE.Points(wg, glowPointsMaterial({ scale: 260, opacity: 0.45 }));
  wind.frustumCulled = false;
  group.add(wind);

  async function show(i) {
    if (!channels.length) return;
    beacon = null;
    index = (i + channels.length) % channels.length;
    const ch = channels[index];
    const tex = await loadTexture(ch.image_url);
    if (cur) { group.remove(cur.root); disposeTree(cur.root); }
    const root = new THREE.Group();
    const [w, h] = ch.stats.size || [1024, 1024];
    const [cx, cy] = ch.stats.disk_center_px || [w / 2, h / 2];
    const rpx = ch.stats.disk_radius_px || w * 0.4;
    const info = CHANNEL[ch.filter] || { glow: 0xffaa55 };

    const sphere = new THREE.Mesh(new THREE.SphereGeometry(R, 128, 96),
      sunMaterial(tex, new THREE.Vector2(cx / w, 1 - cy / h), new THREE.Vector2(rpx / w, rpx / h), info.glow));
    root.add(sphere);
    const corona = new THREE.Mesh(new THREE.SphereGeometry(R * 1.35, 64, 48), coronaMaterial(info.glow));
    root.add(corona);
    root.add(glowSprite(info.glow, R * 4.6, 0.12));

    // Detected regions -> markers on the sphere + loops that rise with the explode value.
    const regions = [];
    ch.regions.forEach((d, k) => {
      const dx = (d.x - cx) / rpx, dy = -(d.y - cy) / rpx;
      const rr = dx * dx + dy * dy;
      if (rr >= 0.98) return;
      const n = new THREE.Vector3(dx, dy, Math.sqrt(1 - rr));
      const color = d.label === 'sunspot' ? 0x7dd3fc : 0x86efac;
      const size = Math.max(0.6, Math.sqrt((d.a || 4) * (d.b || 4)) / rpx * R * 1.1);
      const ring = new THREE.Mesh(new THREE.RingGeometry(size, size + 0.12, 48),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
      ring.position.copy(n).multiplyScalar(R * 1.002);
      ring.lookAt(n.clone().multiplyScalar(R * 2));
      root.add(ring);
      // magnetic loop: an arc whose height scales with explode
      const t = new THREE.Vector3().crossVectors(n, new THREE.Vector3(0, 1, 0.3)).normalize();
      const loopGeom = new THREE.BufferGeometry().setFromPoints(new Array(40).fill(0).map(() => new THREE.Vector3()));
      const loop = new THREE.Line(loopGeom, new THREE.LineBasicMaterial({ color: info.glow, transparent: true, opacity: 0 }));
      root.add(loop);
      const tag = label(`<span class="callout-k">${String(k + 1).padStart(2, '0')}</span>${d.label === 'sunspot' ? 'Sunspot' : 'Active region'}`, 'callout');
      tag.element.style.setProperty('--c', d.label === 'sunspot' ? '#7dd3fc' : '#86efac');
      root.add(tag);
      regions.push({ n, t, size, ring, loop, tag, d, k });
    });

    group.add(root);
    cur = { root, sphere, corona, regions, ch, info };
    lastE = -1;
    ctx.onInfoChange();
  }

  // A pulsing beacon where the agent spotted something.
  let beacon = null;
  function placeBeacon(ch, x, y) {
    if (beacon) { cur.root.remove(beacon); beacon = null; }
    const [w, h] = ch.stats.size || [1024, 1024];
    const [cx, cy] = ch.stats.disk_center_px || [w / 2, h / 2];
    const rpx = ch.stats.disk_radius_px || w * 0.4;
    const dx = (x - cx) / rpx, dy = -(y - cy) / rpx, rr = Math.min(0.97, dx * dx + dy * dy);
    const n = new THREE.Vector3(dx, dy, Math.sqrt(1 - rr)).normalize();
    beacon = new THREE.Group();
    const ring = new THREE.Mesh(new THREE.RingGeometry(1.4, 1.62, 64), new THREE.MeshBasicMaterial({ color: 0xff3b6b, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
    beacon.add(ring, glowSprite(0xff3b6b, 6, 0.6));
    const tag = label('<span class="callout-k">!</span>agent: look here', 'callout');
    tag.element.style.setProperty('--c', '#ff3b6b');
    tag.position.set(0, 3, 0);
    beacon.add(tag);
    beacon.position.copy(n).multiplyScalar(R * 1.01);
    beacon.lookAt(n.clone().multiplyScalar(R * 3));
    beacon.userData.n = n;
    cur.root.add(beacon);
    return n;
  }

  let lastE = -1;
  function applyExplode(e) {
    if (!cur) return;
    cur.corona.scale.setScalar(1 + e * 0.35);
    cur.corona.material.uniforms.uStrength.value = 0.7 + e * 0.7;
    const show = Math.min(1, Math.max(0, (e - 0.2) / 0.35));
    for (const r of cur.regions) {
      const pts = [];
      const span = Math.max(r.size * 1.6, 1.4), hgt = 0.5 + e * (3 + r.size * 2.2);
      for (let i = 0; i < 40; i++) {
        const s = i / 39, ang = (s - 0.5) * span / R;
        const p = r.n.clone().applyAxisAngle(r.t.clone().cross(r.n).normalize(), ang).multiplyScalar(R + Math.sin(Math.PI * s) * hgt);
        pts.push(p);
      }
      r.loop.geometry.setFromPoints(pts);
      r.loop.material.opacity = 0.25 + e * 0.6;
      r.tag.position.copy(r.n).multiplyScalar(R + hgt + 2.2);
      r.tag.element.style.opacity = r.k < MAX_TAGS ? show * (r.n.z > 0.15 ? 1 : 0.25) : 0;
    }
  }

  return {
    name: 'sun',
    title: 'The Sun',
    group,
    get hoverables() { return cur ? cur.regions.map((r) => r.ring) : []; },
    home: { pos: [0, 0, 52], target: [0, 0, 0] },
    autoRotate: false,
    explodeLabel: 'Magnetic loops',
    async load() {
      const [all] = await Promise.all([api('/api/sun'), this.refreshPanel()]);
      channels = ORDER.map((f) => all.find((c) => c.filter === f)).filter(Boolean);
      await show(0);
    },
    async refreshPanel() { try { sw = await api('/api/spacewx'); } catch { /* keep the last one */ } },
    panel() { return spaceWeatherPanel(sw); },
    onPanel(act, el) {
      if (act === 'lasco') this.showLasco(el.dataset.cam);
    },
    showLasco(cam) {
      const l = sw?.lasco?.[cam];
      if (!l) return;
      ctx.movie(`SOHO LASCO ${cam.toUpperCase()}`, l.frame_urls,
        `${cam === 'c2' ? 'C2: 2 to 6 solar radii' : 'C3: 3.7 to 30 solar radii'}. The disc blocks the Sun (white circle = its size) so the faint corona shows. Coronal mass ejections appear as expanding bright clouds.`);
    },
    async enter() { if (!channels.length) await this.load(); },
    update(dt, t, e) {
      if (Math.abs(e - lastE) > 1e-4) { applyExplode(e); lastE = e; }
      if (cur) cur.root.rotation.y = beacon ? 0 : Math.sin(t * 0.08) * 0.35;  // gentle wobble shows off the limb
      if (beacon) { const k = 1 + 0.25 * Math.sin(t * 5); beacon.children[0].scale.setScalar(k); }
      const P = wg.attributes.position.array, C = wg.attributes.aColor.array, A = wg.attributes.aAlpha.array;
      const glow = new THREE.Color(cur?.info.glow || 0xffaa55);
      for (let i = 0; i < WIND; i++) {
        wPhase[i] = (wPhase[i] + dt * (0.03 + e * 0.12)) % 1;
        const d = R * (1.05 + wPhase[i] * (1.2 + e * 2.2));
        P.set([wDir[i * 3] * d, wDir[i * 3 + 1] * d, wDir[i * 3 + 2] * d], i * 3);
        C.set([glow.r, glow.g, glow.b], i * 3);
        A[i] = (1 - wPhase[i]) * (0.25 + e * 0.75);
      }
      wg.attributes.position.needsUpdate = wg.attributes.aColor.needsUpdate = wg.attributes.aAlpha.needsUpdate = true;
    },
    hover(obj) {
      const r = cur?.regions.find((x) => x.ring === obj);
      if (!r) return null;
      return `<b>${r.d.label === 'sunspot' ? 'Sunspot' : 'Active region'} #${r.k + 1}</b><br><span>area ${fmt(Math.round(4 * r.d.a * r.d.b))} px² · brightness ${r.d.score?.toFixed(2)}</span>`;
    },
    pick() {},
    async focusOn(channel, x, y) {
      if (!channels.length) await this.load();
      const all = await api('/api/sun');
      channels = ORDER.map((f) => all.find((c) => c.filter === f)).filter(Boolean);
      const i = channels.findIndex((c) => c.filter === channel);
      if (i < 0) return false;
      await show(i);
      if (x == null) return true;
      const n = placeBeacon(channels[i], x, y);
      ctx.flyTo({ pos: n.clone().multiplyScalar(R * 2.6).toArray(), target: n.clone().multiplyScalar(R * 0.5).toArray() }, 1600);
      return true;
    },
    items: () => channels.map((c) => ({ label: (CHANNEL[c.filter] || {}).name || c.filter, sub: timeAgo(c.obs_time) })),
    get itemIndex() { return index; },
    selectItem(i) { return show(i); },
    next() { return show(index + 1); },
    info() {
      if (!cur) return { kicker: 'SDO', title: 'Loading…', sub: '', stats: [] };
      const s = cur.ch.stats, c = CHANNEL[cur.ch.filter] || {};
      const stats = 'n_sunspots' in s
        ? [['Sunspots', fmt(s.n_sunspots)], ['Spot area', `${fmt(s.spot_area_px)} px`]]
        : [['Active regions', fmt(s.n_active_regions)], ['Activity index', `${s.activity_index}% of disk`]];
      return {
        kicker: `NASA SDO · ${c.name || cur.ch.filter}`,
        title: 'The Sun, right now',
        sub: `${c.desc || ''}. Taken ${cur.ch.obs_time?.slice(11, 16)} UTC (${timeAgo(cur.ch.obs_time)}). Open your hand to raise the magnetic loops.`,
        stats,
      };
    },
  };
}
