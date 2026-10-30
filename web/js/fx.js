// Shared visual building blocks: glowing point clouds, starfield, labels, textures.
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

export const loader = new THREE.TextureLoader();

export function loadTexture(url) {
  return new Promise((resolve, reject) => {
    loader.load(url, (t) => { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; resolve(t); }, undefined, reject);
  });
}

// Soft round glowing points with per-point size, colour and alpha.
export function glowPointsMaterial({ scale = 300, opacity = 1, blending = THREE.AdditiveBlending } = {}) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending,
    uniforms: {
      uScale: { value: scale },
      uOpacity: { value: opacity },
      uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
      uTime: { value: 0 },
    },
    vertexShader: /* glsl */`
      attribute float aSize;
      attribute vec3 aColor;
      attribute float aAlpha;
      varying vec3 vColor;
      varying float vAlpha;
      uniform float uScale, uPixelRatio;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = aSize * uScale * uPixelRatio / max(-mv.z, 0.1);
        vColor = aColor;
        vAlpha = aAlpha;
      }`,
    fragmentShader: /* glsl */`
      varying vec3 vColor;
      varying float vAlpha;
      uniform float uOpacity;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard;
        float halo = smoothstep(0.5, 0.0, d);
        float core = smoothstep(0.18, 0.0, d);
        gl_FragColor = vec4(vColor * (0.55 + core * 1.6), (halo * 0.75 + core * 0.25) * vAlpha * uOpacity);
      }`,
  });
}

export function pointGeometry(n) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(n), 1));
  g.setAttribute('aAlpha', new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
  return g;
}

export function makeStarfield(n = 7000, radius = 600) {
  const g = pointGeometry(n);
  const pos = g.attributes.position.array, col = g.attributes.aColor.array, size = g.attributes.aSize.array;
  const tints = [[0.7, 0.8, 1], [1, 0.95, 0.85], [1, 0.8, 0.7], [0.8, 0.9, 1]];
  for (let i = 0; i < n; i++) {
    const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, r = radius * (0.6 + Math.random() * 0.4);
    const s = Math.sqrt(1 - u * u);
    pos.set([r * s * Math.cos(th), r * u, r * s * Math.sin(th)], i * 3);
    const t = tints[(Math.random() * tints.length) | 0], b = 0.35 + Math.random() ** 3 * 0.9;
    col.set([t[0] * b, t[1] * b, t[2] * b], i * 3);
    size[i] = 0.6 + Math.random() ** 4 * 3.5;
  }
  const m = glowPointsMaterial({ scale: 520, opacity: 0.9 });
  const pts = new THREE.Points(g, m);
  pts.frustumCulled = false;
  return pts;
}

// Large, faint coloured clouds behind everything.
export function makeNebula() {
  const group = new THREE.Group();
  const tex = radialTexture();
  const blobs = [
    [-260, 120, -420, 520, 0x3b2a8f, 0.35], [300, -80, -460, 600, 0x0e4a6e, 0.32],
    [60, 220, -520, 480, 0x6b1d5c, 0.22], [-120, -240, -380, 420, 0x12355e, 0.3],
  ];
  for (const [x, y, z, s, c, o] of blobs) {
    const m = new THREE.SpriteMaterial({ map: tex, color: c, transparent: true, opacity: o, depthWrite: false, blending: THREE.AdditiveBlending });
    const sp = new THREE.Sprite(m);
    sp.position.set(x, y, z);
    sp.scale.set(s, s, 1);
    group.add(sp);
  }
  return group;
}

let _radial;
export function radialTexture() {
  if (_radial) return _radial;
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.25, 'rgba(255,255,255,0.45)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 256, 256);
  _radial = new THREE.CanvasTexture(c);
  return _radial;
}

export function glowSprite(color, size, opacity = 0.8) {
  const m = new THREE.SpriteMaterial({ map: radialTexture(), color, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending });
  const s = new THREE.Sprite(m);
  s.scale.set(size, size, 1);
  return s;
}

export function label(html, cls = 'lbl') {
  const el = document.createElement('div');
  el.className = cls;
  el.innerHTML = html;
  const o = new CSS2DObject(el);
  o.center.set(0.5, 0.5);
  return o;
}

export function disposeTree(obj) {
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => { if (m.map && m.map !== _radial) m.map.dispose(); m.dispose(); });
    }
    if (o.isCSS2DObject && o.element) o.element.remove();
  });
}

export const lerp = (a, b, t) => a + (b - a) * t;
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const hex = (c) => new THREE.Color(c);

export function timeAgo(iso) {
  if (!iso) return '—';
  const s = (Date.now() - Date.parse(iso)) / 1000;
  if (s < 60) return `${Math.max(0, s | 0)} s ago`;
  if (s < 3600) return `${(s / 60) | 0} min ago`;
  if (s < 86400) return `${(s / 3600) | 0} h ago`;
  return `${(s / 86400) | 0} d ago`;
}

export function fmt(n) {
  return n == null ? '—' : Number(n).toLocaleString('en-US');
}

export async function api(path) {
  const r = await fetch(path, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
