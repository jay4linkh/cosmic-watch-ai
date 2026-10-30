// Mars, today: the newest raw images from Perseverance (Jezero crater) and Curiosity (Gale crater),
// floating above each rover's real landing site, classified by our Mars CNN.
import * as THREE from 'three';
import { api, loadTexture, glowSprite, label, fmt, timeAgo, disposeTree, esc } from '../fx.js';

const R = 14;
const ROVER_COLOR = { perseverance: 0x67e8f9, curiosity: 0xfbbf24 };
const SCENE_COLOR = {
  'Drill hole': '#f472b6', 'Layers': '#a78bfa', 'Light-toned veins': '#e879f9', 'Float': '#c084fc', 'Sun': '#fde047',
  'Night sky': '#818cf8', 'Distant landscape': '#67e8f9', 'Close-up rock': '#fdba74', 'Nearby surface': '#fb923c',
  'Sand': '#fcd34d', 'Wheel tracks': '#94a3b8', 'Artifact': '#64748b',
};
const FILTERS = ['All', 'Perseverance', 'Curiosity'];
const PER_ROVER = 6;

function latLon(lat, lon, r) {
  const phi = THREE.MathUtils.degToRad(90 - lat), theta = THREE.MathUtils.degToRad(lon + 180);
  return new THREE.Vector3(-r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
}

function atmosphere() {
  return new THREE.Mesh(new THREE.SphereGeometry(R * 1.06, 64, 48), new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.BackSide, blending: THREE.AdditiveBlending,
    vertexShader: 'varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix*vec4(position,1.0); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }',
    fragmentShader: 'varying vec3 vN; varying vec3 vV; void main(){ float i = pow(max(0.62 - dot(vN, vV), 0.0), 2.4); gl_FragColor = vec4(1.0, 0.55, 0.35, i * 0.9); }',
  }));
}

export function createMars(ctx) {
  const group = new THREE.Group();
  const planet = new THREE.Group();
  group.add(planet);
  const globe = new THREE.Mesh(new THREE.SphereGeometry(R, 96, 64), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  loadTexture('/textures/2k_mars.jpg').then((t) => { globe.material.map = t; globe.material.needsUpdate = true; }).catch(() => {});
  planet.add(globe);
  group.add(atmosphere(), glowSprite(0xff8a5c, R * 3.4, 0.12));

  let images = [], filter = 0, cards = [], hovered = null;
  const sites = {};
  const cardsGroup = new THREE.Group();                     // in view space (not on the turning planet)
  group.add(cardsGroup);

  for (const [key, lat, lon, name] of [['perseverance', 18.44, 77.45, 'Perseverance · Jezero crater'], ['curiosity', -4.59, 137.44, 'Curiosity · Gale crater']]) {
    const n = latLon(lat, lon, 1).normalize();
    const pin = new THREE.Group();
    pin.add(glowSprite(ROVER_COLOR[key], 2.4, 0.95));
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.7, 40), new THREE.MeshBasicMaterial({ color: ROVER_COLOR[key], side: THREE.DoubleSide, transparent: true }));
    pin.add(ring);
    pin.position.copy(n).multiplyScalar(R * 1.004);
    pin.lookAt(n.clone().multiplyScalar(R * 2));
    const tag = label(`<b style="color:#${ROVER_COLOR[key].toString(16)}">●</b> ${name}`, 'callout');
    tag.position.copy(n).multiplyScalar(R * 1.12);
    planet.add(pin, tag);
    sites[key] = { n, pin, ring };
  }

  // turn Mars so both rover sites face the camera
  const mid = sites.perseverance.n.clone().add(sites.curiosity.n).normalize();
  const faceYaw = Math.atan2(-mid.x, mid.z);
  planet.rotation.y = faceYaw;
  planet.rotation.x = Math.asin(mid.y) * 0.8;

  function scene(s) { return s?.scene ? `${s.scene} ${Math.round((s.scene_conf || 0) * 100)}%` : 'not classified'; }

  async function build() {
    cardsGroup.children.slice().forEach((c) => { cardsGroup.remove(c); disposeTree(c); });
    cards = [];
    const f = FILTERS[filter].toLowerCase();
    // up to PER_ROVER photos each; skip blank calibration frames ("Artifact") in the 3D view (they stay in the panel)
    const sel = ['perseverance', 'curiosity'].flatMap((r) => images.filter((im) => im.stats.rover === r && (f === 'all' || f === r)
      && im.stats.scene !== 'Artifact').slice(0, PER_ROVER));
    const texes = await Promise.all(sel.map((im) => loadTexture(im.image_url.replace('/thumb/2048/', '/thumb/512/')).catch(() => null)));
    const perRover = {};
    sel.forEach((im, i) => {
      if (!texes[i]) return;
      const rover = im.stats.rover, k = (perRover[rover] = (perRover[rover] || 0) + 1) - 1;
      const img = texes[i].image, aspect = img.height / img.width;
      const w = 6.2, h = w * aspect;
      const card = new THREE.Group();
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: texes[i], color: 0xa0a0a0, side: THREE.DoubleSide }));
      mesh.userData.image = im;
      const col = SCENE_COLOR[im.stats.scene] || '#e6edf7';
      const border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(w * 1.04, h * 1.06)), new THREE.LineBasicMaterial({ color: new THREE.Color(col) }));
      const tag = label(`<b style="color:#${ROVER_COLOR[rover].toString(16)}">●</b> <b style="color:${col}">${esc(im.stats.scene || '?')}</b> · ${esc(im.stats.camera_name || '')}`, 'card-tag');
      tag.position.y = -h / 2 - 0.6;
      card.add(mesh, border, tag);
      // two arcs over the planet: Perseverance upper-left, Curiosity upper-right
      const n = sel.filter((x) => x.stats.rover === rover).length;
      const t = n > 1 ? k / (n - 1) : 0.5;
      const ang = rover === 'perseverance' ? THREE.MathUtils.degToRad(158 - t * 58) : THREE.MathUtils.degToRad(22 + t * 58);
      const rad = R + 9 + (k % 2) * 3.5;
      card.userData = { base: new THREE.Vector3(Math.cos(ang) * rad * 0.95, Math.sin(ang) * rad * 0.8 + 3, 4 + (k % 2) * 2),
                        mesh, tag, hover: 0, phase: Math.random() * 6 };
      card.position.copy(card.userData.base);
      cardsGroup.add(card);
      cards.push(card);
    });
    ctx.onInfoChange();
  }

  function newest(rover) { return images.filter((im) => im.stats.rover === rover).reduce((m, im) => Math.max(m, im.stats.sol || 0), 0); }

  return {
    name: 'mars',
    title: 'Mars',
    group,
    get hoverables() { return cards.map((c) => c.userData.mesh); },
    home: { pos: [0, 6, 66], target: [0, 5, 0] },
    autoRotate: false,
    explodeLabel: 'Lift the photos',
    async load() { images = await api('/api/mars?limit=80'); await build(); },
    async enter() { if (!images.length) await this.load(); },
    async refreshPanel() {},
    update(dt, t, e) {
      planet.rotation.y = faceYaw + Math.sin(t * 0.05) * 0.18;     // gentle sway, rovers stay in view
      Object.values(sites).forEach((s, i) => s.ring.scale.setScalar(1 + 0.3 * Math.sin(t * 3 + i)));
      cards.forEach((c) => {
        const u = c.userData;
        c.position.copy(u.base).multiplyScalar(1 + e * 0.3);
        c.position.y += Math.sin(t * 0.8 + u.phase) * 0.25;
        c.quaternion.copy(ctx.camera.quaternion);
        u.hover += ((c === hovered ? 1 : 0) - u.hover) * Math.min(1, dt * 8);
        c.scale.setScalar(1 + u.hover * 0.6);
        u.mesh.material.color.setScalar(0.62 + u.hover * 0.35);
        u.tag.element.style.opacity = Math.max(u.hover, 0.25 + e * 0.75);
      });
    },
    hover(obj) {
      const im = obj.userData.image;
      hovered = cards.find((c) => c.userData.mesh === obj) || null;
      if (!im) return null;
      const s = im.stats;
      return `<b>${esc(im.title)}</b><br><span>${esc(scene(s))}${s.sky_fraction ? ` · sky ${Math.round(s.sky_fraction * 100)}%` : ''}<br>${timeAgo(im.obs_time)}</span><br><em>click / point to open</em>`;
    },
    unhover() { hovered = null; },
    pick(obj) { const im = obj.userData.image; if (im) this.openImage(im.id); },
    async openImage(id) {
      if (!images.length) await this.load();
      const im = images.find((x) => x.id === Number(id));
      if (!im) return false;
      const [w, h] = im.stats.size || [1, 1];
      ctx.openDeepZoom({ kind: 'image', imageId: im.id, u: 0.5, v: 0.5, r: 0.5, rh: 0.5 * h / w, mars: im,
                         kicker: `${im.stats.rover_name?.toUpperCase()} · SOL ${im.stats.sol} · DEEP ZOOM`, title: im.title });
      return true;
    },
    panel() {
      const by = (r) => images.filter((im) => im.stats.rover === r).slice(0, 6);
      const finds = images.filter((im) => ['Drill hole', 'Layers', 'Light-toned veins', 'Float', 'Sun', 'Night sky', 'DRT spot'].includes(im.stats.scene));
      return `
        <div class="feed-k">MARS ROVERS · RAW IMAGES, DAILY</div>
        <p class="feed-dim small">NASA publishes each sol's raw images within hours of downlink. Our Mars CNN (trained on 6,820 labelled Curiosity images on this laptop's GPU: 80% accuracy on unseen terrain) names what each shows.</p>
        ${['perseverance', 'curiosity'].map((r) => `
          <div class="feed-sub"><b style="color:#${ROVER_COLOR[r].toString(16)}">●</b> ${r === 'perseverance' ? 'Perseverance · Jezero crater' : 'Curiosity · Gale crater'} · sol ${newest(r) || '—'}</div>
          <div class="feed-list">${by(r).map((im) => `<button class="feed-item" data-act="img" data-id="${im.id}">
            <span class="chip" style="--c:${SCENE_COLOR[im.stats.scene] || '#94a3b8'}">${esc(im.stats.scene || '?')}</span>
            <b>${esc(im.stats.camera_name)}</b><small>${Math.round((im.stats.scene_conf || 0) * 100)}% · ${timeAgo(im.obs_time)}</small></button>`).join('') || '<p class="feed-dim">No images yet.</p>'}</div>`).join('')}
        ${finds.length ? `<p class="feed-line">Science highlights today: ${finds.map((im) => `<b>${esc(im.stats.scene)}</b>`).join(', ')}</p>` : ''}
        <p class="feed-dim small">Mars texture: Solar System Scope (CC BY 4.0), from NASA data.</p>`;
    },
    onPanel(act, el) { if (act === 'img') this.openImage(el.dataset.id); },
    items: () => FILTERS.map((f) => ({ label: f, sub: f === 'All' ? `${images.length}` : `${images.filter((im) => im.stats.rover === f.toLowerCase()).length}` })),
    get itemIndex() { return filter; },
    selectItem(i) { filter = i; return build(); },
    next() { filter = (filter + 1) % FILTERS.length; return build(); },
    info() {
      const sols = ['perseverance', 'curiosity'].map((r) => newest(r));
      return {
        kicker: 'NASA MARS ROVERS · LIVE',
        title: 'Mars, today',
        sub: 'The newest raw photos from two rovers, placed above their real sites: Perseverance in Jezero crater (an ancient lake) and Curiosity on Mount Sharp in Gale crater. Click a photo to zoom into it and see what our Mars CNN thinks it shows.',
        stats: [['Photos', fmt(images.length)], ['Perseverance sol', fmt(sols[0] || null)], ['Curiosity sol', fmt(sols[1] || null)],
                ['Classified', fmt(images.filter((im) => im.stats.scene).length)]],
        legend: Object.entries(SCENE_COLOR).slice(0, 8).map(([k, c]) => [k, c]),
      };
    },
  };
}
