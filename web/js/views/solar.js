// Solar System: the Sun, planets and Moon where they really are, for any date from 1800 to 2050.
// Positions: JPL's Keplerian elements for the major planets (Standish), checked against astropy to ~0.2 deg;
// the Moon from a low-precision series (~1.5 deg). Bodies are lit by the Sun only, so phases and seasons are real.
// Sizes are exaggerated and (by default) distances compressed, otherwise nothing but the Sun would be visible.
import * as THREE from 'three';
import { loadTexture, glowSprite, label, fmt, esc } from '../fx.js';

const D2R = Math.PI / 180, AU_KM = 149597870.7, C_KMS = 299792.458;
const MIN_MS = Date.UTC(1800, 0, 1), MAX_MS = Date.UTC(2050, 0, 1);
// a, e, I, L, long. of perihelion, long. of node: value at J2000 and change per century (AU, deg)
const EL = {
  mercury: [0.38709927, 0.00000037, 0.20563593, 0.00001906, 7.00497902, -0.00594749, 252.25032350, 149472.67411175, 77.45779628, 0.16047689, 48.33076593, -0.12534081],
  venus: [0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890, 181.97909950, 58517.81538729, 131.60246718, 0.00268329, 76.67984255, -0.27769418],
  earth: [1.00000261, 0.00000562, 0.01671123, -0.00004392, -0.00001531, -0.01294668, 100.46457166, 35999.37244981, 102.93768193, 0.32327364, 0, 0],
  mars: [1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131, -4.55343205, 19140.30268499, -23.94362959, 0.44441088, 49.55953891, -0.29257343],
  jupiter: [5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714, 34.39644051, 3034.74612775, 14.72847983, 0.21252668, 100.47390909, 0.20469106],
  saturn: [9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187, 0.00193609, 49.95424423, 1222.49362201, 92.59887831, -0.41897216, 113.66242448, -0.28867794],
  uranus: [19.18916464, -0.00196176, 0.04725744, -0.00004397, 0.77263783, -0.00242939, 313.23810451, 428.48202785, 170.95427630, 0.40805281, 74.01692503, 0.04240589],
  neptune: [30.06992276, 0.00026291, 0.00859048, 0.00005105, 1.77004347, 0.00035372, -55.12002969, 218.45945325, 44.96476227, -0.32241464, 131.78422574, -0.00508664],
};
// radius km, sidereal rotation h, axial tilt deg, year in days, surface gravity m/s2, mass in Earths
const BODIES = [
  { id: 'sun', name: 'Sun', R: 695700, rot: 609.12, tilt: 7.25, g: 274, mass: 332946, color: '#ffd27a', kind: 'G2V star', live: ['sun', 'Open the live Sun (SDO, flares, CMEs)'] },
  { id: 'mercury', name: 'Mercury', R: 2439.7, rot: 1407.6, tilt: 0.03, year: 87.969, g: 3.7, mass: 0.0553, color: '#b5a79a', kind: 'Rocky planet', tex: 'moon', tint: [0.86, 0.78, 0.7] },
  { id: 'venus', name: 'Venus', R: 6051.8, rot: -5832.5, tilt: 2.64, year: 224.701, g: 8.87, mass: 0.815, color: '#e8c97a', kind: 'Rocky planet', tex: 'venus', atm: [1.0, 0.85, 0.55, 0.9] },
  { id: 'earth', name: 'Earth', R: 6371, rot: 23.9345, tilt: 23.44, year: 365.256, g: 9.81, mass: 1, color: '#6bb6ff', kind: 'Rocky planet', tex: 'earth', atm: [0.35, 0.6, 1.0, 1.1], clouds: true, live: ['neo', 'Open Near Earth (ISS, asteroid flybys)'] },
  { id: 'mars', name: 'Mars', R: 3389.5, rot: 24.6229, tilt: 25.19, year: 686.98, g: 3.71, mass: 0.107, color: '#ff7b54', kind: 'Rocky planet', tex: 'mars', atm: [1.0, 0.6, 0.4, 0.35], live: ['mars', 'Open Mars (newest rover photos)'] },
  { id: 'jupiter', name: 'Jupiter', R: 69911, rot: 9.925, tilt: 3.13, year: 4332.59, g: 24.79, mass: 317.8, color: '#e0b48a', kind: 'Gas giant', tex: 'jupiter', atm: [1.0, 0.85, 0.7, 0.3] },
  { id: 'saturn', name: 'Saturn', R: 58232, rot: 10.656, tilt: 26.73, year: 10759.2, g: 10.44, mass: 95.2, color: '#ecd9a0', kind: 'Gas giant', tex: 'saturn', atm: [1.0, 0.9, 0.7, 0.3], rings: true },
  { id: 'uranus', name: 'Uranus', R: 25362, rot: 17.24, tilt: 97.77, year: 30688.5, g: 8.87, mass: 14.5, color: '#9be8ee', kind: 'Ice giant', tex: 'uranus', atm: [0.6, 0.95, 1.0, 0.5] },
  { id: 'neptune', name: 'Neptune', R: 24622, rot: 16.11, tilt: 28.32, year: 60182, g: 11.15, mass: 17.1, color: '#6f8dff', kind: 'Ice giant', tex: 'neptune', atm: [0.4, 0.55, 1.0, 0.6] },
  { id: 'moon', name: 'Moon', R: 1737.4, rot: 655.72, tilt: 0, year: 27.3217, g: 1.62, mass: 0.0123, color: '#cfd6e0', kind: "Earth's moon", tex: 'moon' },
];
const SPEEDS = [[-31557600, '1 year / s, backwards'], [-2629800, '1 month / s, backwards'], [-86400, '1 day / s, backwards'], [-3600, '1 hour / s, backwards'],
  [1, 'real time'], [60, '1 minute / s'], [3600, '1 hour / s'], [86400, '1 day / s'], [604800, '1 week / s'], [2629800, '1 month / s'], [31557600, '1 year / s']];
const REAL = 4;
const SCALES = ['Compressed distances', 'True distances', 'Journey through the galaxy'];

// heliocentric ecliptic J2000 position (AU) from the elements
function helio(el, jd, out) {
  const T = (jd - 2451545.0) / 36525;
  const a = el[0] + el[1] * T, e = el[2] + el[3] * T, I = (el[4] + el[5] * T) * D2R, L = el[6] + el[7] * T, w = el[8] + el[9] * T, O = el[10] + el[11] * T;
  const M = ((((L - w + 180) % 360) + 360) % 360 - 180) * D2R;
  let E = M + e * Math.sin(M);
  for (let i = 0; i < 8; i++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  return orbitPoint(a, e, I, (w - O) * D2R, O * D2R, E, out);
}
function orbitPoint(a, e, I, om, O, E, out) {
  const x = a * (Math.cos(E) - e), y = a * Math.sqrt(1 - e * e) * Math.sin(E);
  const co = Math.cos(om), so = Math.sin(om), cO = Math.cos(O), sO = Math.sin(O), cI = Math.cos(I), sI = Math.sin(I);
  // ecliptic (x, y, z) -> scene (x, z, -y): the ecliptic is the horizontal plane, north is up
  return out.set((co * cO - so * sO * cI) * x + (-so * cO - co * sO * cI) * y, so * sI * x + co * sI * y, -((co * sO + so * cO * cI) * x + (-so * sO + co * cO * cI) * y));
}
function moonGeo(jd, out) {          // geocentric ecliptic direction and distance (km)
  const d = jd - 2451545.0, Mm = (134.963 + 13.064993 * d) * D2R, F = (93.272 + 13.229350 * d) * D2R;
  const lam = (218.316 + 13.176396 * d) * D2R + 6.289 * D2R * Math.sin(Mm), beta = 5.128 * D2R * Math.sin(F);
  out.set(Math.cos(beta) * Math.cos(lam), Math.sin(beta), -Math.cos(beta) * Math.sin(lam));
  return 385001 - 20905 * Math.cos(Mm);
}

// ---------- procedural textures (no image exists for these: clearly not photographs) ----------
function vnoise(px) {                 // tileable-in-x value noise
  const h = (x, y) => { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); };
  return (x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi, u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const w = (i) => ((i % px) + px) % px;
    return (h(w(xi), yi) * (1 - u) + h(w(xi + 1), yi) * u) * (1 - v) + (h(w(xi), yi + 1) * (1 - u) + h(w(xi + 1), yi + 1) * u) * v;
  };
}
function fbm(n, x, y, oct = 4) { let s = 0, a = 0.5; for (let i = 0; i < oct; i++) { s += a * n(x, y); a *= 0.5; } return s; }
const hex = (c) => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
function bandTexture({ palette, warp = 0.03, streak = 0.12, spot = null, swirl = 0 }) {
  const W = 1024, H = 512, cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const g = cv.getContext('2d'), img = g.createImageData(W, H), cols = palette.map(hex);
  const n1 = vnoise(8), n2 = vnoise(16), n3 = vnoise(64), n4 = vnoise(32);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = x / W, v = y / H;
      const t = fbm((a, b) => n1(a, b), u * 8, v * 6) * 0.6 + n2(u * 16, v * 14) * 0.4;
      let vv = v + (t - 0.5) * warp + swirl * (n4(u * 32 + v * 9, v * 10) - 0.5);
      vv = Math.min(0.9999, Math.max(0, vv)) * (cols.length - 1);
      const i = Math.floor(vv), f = vv - i, s = f * f * (3 - 2 * f), a = cols[i], b = cols[i + 1];
      const k = 1 + (n3(u * 64, v * 90) - 0.5) * streak + (n2(u * 16, v * 40) - 0.5) * streak;
      let r = (a[0] + (b[0] - a[0]) * s) * k, gg = (a[1] + (b[1] - a[1]) * s) * k, bl = (a[2] + (b[2] - a[2]) * s) * k;
      if (spot) {
        const dx = Math.min(Math.abs(u - spot.u), 1 - Math.abs(u - spot.u)) / spot.w, dy = (v - spot.v) / spot.h, d = dx * dx + dy * dy;
        if (d < 1) { const m = (1 - d) ** 0.7, c = hex(spot.color); r += (c[0] - r) * m; gg += (c[1] - gg) * m; bl += (c[2] - bl) * m; }
      }
      const o = (y * W + x) * 4;
      img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = bl; img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  const tx = new THREE.CanvasTexture(cv);
  tx.colorSpace = THREE.SRGBColorSpace; tx.wrapS = THREE.RepeatWrapping; tx.anisotropy = 4;
  return tx;
}
const PROC = {
  jupiter: () => bandTexture({ palette: ['#8f7558', '#c9b08c', '#e9dcc2', '#b98a5e', '#efe4cf', '#a8744a', '#e6d6b8', '#f1e9da', '#b07b50', '#e4d2b1', '#9c6b46', '#dcc8a6', '#c2a47c', '#8d7155'], warp: 0.035, streak: 0.16, spot: { u: 0.3, v: 0.62, w: 0.045, h: 0.035, color: '#b5532f' } }),
  saturn: () => bandTexture({ palette: ['#b9a276', '#d8c394', '#e6d5a8', '#d2b983', '#eadbb5', '#d6bd88', '#e3cf9f', '#cbb07c', '#dcc797', '#b49c70'], warp: 0.012, streak: 0.05 }),
  uranus: () => bandTexture({ palette: ['#9fd6dc', '#a9dfe4', '#b3e6ea', '#a9dfe4', '#a3dade'], warp: 0.01, streak: 0.02 }),
  neptune: () => bandTexture({ palette: ['#2a47a8', '#3458c8', '#3f66d8', '#3257c6', '#4670e0', '#2f52c0', '#2743a0'], warp: 0.03, streak: 0.07, spot: { u: 0.62, v: 0.6, w: 0.035, h: 0.03, color: '#1c2f78' } }),
  venus: () => bandTexture({ palette: ['#cdb37a', '#e6d3a0', '#d9bf80', '#efe0b4', '#dcc48a', '#e8d6a6', '#cbb07a'], warp: 0.12, streak: 0.05, swirl: 0.1 }),
};

// ---------- materials ----------
const PLANET_VS = /* glsl */`
  varying vec2 vUv; varying vec3 vN; varying vec3 vW;
  void main() { vUv = uv; vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
function planetMaterial(b) {
  return new THREE.ShaderMaterial({
    uniforms: { map: { value: null }, uSun: { value: new THREE.Vector3(1, 0, 0) }, uTint: { value: new THREE.Vector3(...(b.tint || [1, 1, 1])) },
                uAtm: { value: new THREE.Vector4(...(b.atm || [0, 0, 0, 0])) }, uBase: { value: new THREE.Color(b.color) }, uHas: { value: 0 },
                uClouds: { value: b.clouds ? 1 : 0 }, uTime: { value: 0 } },
    vertexShader: PLANET_VS,
    fragmentShader: /* glsl */`
      uniform sampler2D map; uniform vec3 uSun, uTint, uBase; uniform vec4 uAtm; uniform float uHas, uClouds, uTime;
      varying vec2 vUv; varying vec3 vN; varying vec3 vW;
      float h(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float n(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(h(i), h(i + vec2(1, 0)), f.x), mix(h(i + vec2(0, 1)), h(i + vec2(1, 1)), f.x), f.y); }
      float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * n(p); p *= 2.03; a *= 0.5; } return s; }
      void main() {
        vec3 N = normalize(vN), V = normalize(cameraPosition - vW);
        vec3 alb = mix(uBase, texture2D(map, vUv).rgb, uHas) * uTint;
        float ndl = dot(N, uSun), day = smoothstep(-0.06, 0.28, ndl);
        if (uClouds > 0.5) {                                   // procedural cloud deck (not today's weather)
          vec2 q = vec2(vUv.x * 12.0 + uTime * 0.004, vUv.y * 6.0);
          float c = fbm(q + fbm(q * 1.7) * 1.2);
          c = smoothstep(0.5, 0.78, c) * (0.55 + 0.45 * smoothstep(0.0, 0.25, abs(vUv.y - 0.5)));
          alb = mix(alb, vec3(0.96), c * 0.85);
          float spec = pow(max(dot(reflect(-uSun, N), V), 0.0), 40.0) * (1.0 - c) * step(alb.r + alb.g, alb.b * 1.9);
          alb += vec3(1.0, 0.95, 0.85) * spec * 0.9;
        }
        vec3 col = alb * (day * 0.8 + 0.004);          // kept under the bloom threshold so surfaces stay detailed
        float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        col += uAtm.rgb * rim * uAtm.a * smoothstep(-0.25, 0.35, ndl);
        col = mix(col, col * vec3(1.0, 0.55, 0.35), uAtm.a * 0.35 * smoothstep(0.3, 0.0, abs(ndl)) * day);   // sunset band at the terminator
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
}
function haloMaterial(atm) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uSun: { value: new THREE.Vector3(1, 0, 0) }, uAtm: { value: new THREE.Vector4(...atm) } },
    vertexShader: PLANET_VS,
    fragmentShader: /* glsl */`
      uniform vec3 uSun; uniform vec4 uAtm; varying vec3 vN; varying vec3 vW;
      void main() { vec3 N = normalize(vN); float x = max(dot(N, normalize(cameraPosition - vW)), 0.0);
        float a = pow(1.0 - x, 2.2) * smoothstep(0.0, 0.3, x) * smoothstep(-0.35, 0.3, dot(N, uSun)) * uAtm.a * 1.6;
        gl_FragColor = vec4(uAtm.rgb * a, a); }`,
  });
}
const NOISE3 = /* glsl */`
  float h3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
  float n3(vec3 p) { vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(h3(i), h3(i + vec3(1, 0, 0)), f.x), mix(h3(i + vec3(0, 1, 0)), h3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(h3(i + vec3(0, 0, 1)), h3(i + vec3(1, 0, 1)), f.x), mix(h3(i + vec3(0, 1, 1)), h3(i + vec3(1, 1, 1)), f.x), f.y), f.z); }
  float fbm3(vec3 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * n3(p); p *= 2.02; a *= 0.5; } return s; }`;
function sunMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 } },
    vertexShader: /* glsl */`varying vec3 vP; varying vec3 vN; varying vec3 vW;
      void main() { vP = position; vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: /* glsl */`uniform float uTime; varying vec3 vP; varying vec3 vN; varying vec3 vW; ${NOISE3}
      void main() {
        float big = fbm3(vP * 2.5 + vec3(0.0, uTime * 0.03, 0.0)), gran = fbm3(vP * 16.0 + vec3(uTime * 0.12));
        float mu = max(dot(normalize(vN), normalize(cameraPosition - vW)), 0.0);
        vec3 c = mix(vec3(1.0, 0.36, 0.05), vec3(1.0, 0.82, 0.42), smoothstep(0.3, 0.72, big * 0.55 + gran * 0.45));
        c *= (0.45 + 0.55 * pow(mu, 0.45)) * 2.4;                       // limb darkening; > 1 feeds the bloom
        gl_FragColor = vec4(c, 1.0);
      }`,
  });
}
function ringMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    uniforms: { uSun: { value: new THREE.Vector3(1, 0, 0) }, uCenter: { value: new THREE.Vector3() }, uR: { value: 1 } },
    vertexShader: /* glsl */`varying vec3 vL; varying vec3 vW; void main() { vL = position; vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: /* glsl */`uniform vec3 uSun, uCenter; uniform float uR; varying vec3 vL; varying vec3 vW;
      float h(float x) { return fract(sin(x * 91.7) * 43758.5); }
      void main() {
        float r = length(vL.xy);                                         // in Saturn radii: C 1.24-1.53, B 1.53-1.95, Cassini gap, A 2.03-2.27
        float a = smoothstep(1.24, 1.28, r) * 0.25 + smoothstep(1.52, 1.55, r) * 0.65 - smoothstep(1.94, 1.96, r) * 0.82 + smoothstep(2.02, 2.04, r) * 0.6;
        a *= 1.0 - smoothstep(2.25, 2.27, r);
        a *= 0.75 + 0.25 * h(floor(r * 160.0)) ;                         // fine ringlets
        a *= 1.0 - 0.6 * smoothstep(0.008, 0.0, abs(r - 2.212));          // Encke gap
        vec3 p = vW - uCenter; float b = dot(p, uSun);
        float lit = (b < 0.0 && dot(p, p) - b * b < uR * uR) ? 0.06 : 1.0;   // the planet's shadow on the rings
        gl_FragColor = vec4(mix(vec3(0.62, 0.52, 0.4), vec3(0.95, 0.88, 0.72), smoothstep(1.3, 1.9, r)) * lit * 0.6, clamp(a, 0.0, 1.0));
      }`,
  });
}
function milkyWay() {                // the galaxy's band in its real orientation; brightness and dust are procedural
  const pole = new THREE.Vector3(), centre = new THREE.Vector3();
  const ecl = (lon, lat, o) => o.set(Math.cos(lat * D2R) * Math.cos(lon * D2R), Math.sin(lat * D2R), -Math.cos(lat * D2R) * Math.sin(lon * D2R));
  ecl(180.02, 29.81, pole); ecl(266.84, -5.54, centre);
  const m = new THREE.Mesh(new THREE.SphereGeometry(2400, 48, 32), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, transparent: true, blending: THREE.AdditiveBlending,
    uniforms: { uPole: { value: pole }, uCentre: { value: centre }, uFade: { value: 1 } },
    vertexShader: /* glsl */`varying vec3 vD; void main() { vD = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */`uniform vec3 uPole, uCentre; uniform float uFade; varying vec3 vD; ${NOISE3}
      void main() {
        vec3 d = normalize(vD); float lat = dot(d, uPole), toC = dot(d, uCentre) * 0.5 + 0.5;
        float width = mix(0.10, 0.24, toC * toC);
        float band = exp(-lat * lat / (width * width));
        float cloud = fbm3(d * 5.0) * 0.7 + fbm3(d * 13.0) * 0.3;
        float dust = smoothstep(0.42, 0.7, fbm3(d * 9.0 + 4.0)) * exp(-lat * lat / (width * width * 0.25));
        float stars = pow(n3(d * 520.0), 46.0) * 14.0 + pow(n3(d * 260.0 + 9.0), 60.0) * 30.0;
        float b = band * (0.18 + 0.9 * toC * toC) * (0.45 + cloud) * (1.0 - 0.8 * dust);
        vec3 col = mix(vec3(0.55, 0.62, 0.85), vec3(1.0, 0.86, 0.66), toC) * b * 0.42 + vec3(0.9, 0.92, 1.0) * stars * (0.05 + band * 0.9);
        gl_FragColor = vec4(col * uFade, 1.0);
      }`,
  }));
  m.renderOrder = -10;
  return m;
}

// ---------- the Milky Way from outside ----------
// A model galaxy (spiral arms, bar and bulge are drawn, not observed: nobody has a photo of our galaxy from outside),
// placed so that its plane, its centre and the direction the Sun travels are the real ones as seen from the solar system.
const GAL_R0 = 12000;                                   // scene units for the Sun's 26,000 light-years from the centre
const eclDir = (lon, lat) => new THREE.Vector3(Math.cos(lat * D2R) * Math.cos(lon * D2R), Math.sin(lat * D2R), -Math.cos(lat * D2R) * Math.sin(lon * D2R));
const GAL_POLE = eclDir(180.02, 29.81);
const GAL_X = eclDir(266.84, -5.54).addScaledVector(GAL_POLE, -eclDir(266.84, -5.54).dot(GAL_POLE)).normalize();     // toward the galactic centre
const GAL_Y = new THREE.Vector3().crossVectors(GAL_POLE, GAL_X).normalize();                                            // the way the Sun is moving
// The face of the galaxy: painted by a shader (spiral arms, dust lanes, bar, star-forming knots), not a photograph.
function galaxyDisk() {
  const size = GAL_R0 * 4.8;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
    uniforms: { uFade: { value: 0 } },
    vertexShader: /* glsl */`varying vec2 vP; void main() { vP = position.xy / ${GAL_R0.toFixed(1)}; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */`uniform float uFade; varying vec2 vP;
      float h(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float n2(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(h(i), h(i + vec2(1, 0)), f.x), mix(h(i + vec2(0, 1)), h(i + vec2(1, 1)), f.x), f.y); }
      float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 6; i++) { s += a * n2(p); p = p * 2.03 + 1.7; a *= 0.5; } return s; }
      void main() {
        vec2 p = vP;
        float r = length(p);
        if (r > 2.35) discard;
        float th = atan(p.y, p.x);
        vec2 q = p * 3.2;
        float t1 = fbm(q + 7.3), t2 = fbm(q * 2.4 - 3.1), t3 = fbm(q * 6.0 + 2.0);
        // logarithmic spiral, pitch 12.5 deg; the turbulence makes the arms ragged instead of ruled lines
        float ph = th - log(max(r, 0.02) / 0.22) / 0.2217 + (t1 - 0.5) * 1.9 + (t2 - 0.5) * 0.6;
        float arm2 = pow(0.5 + 0.5 * cos(2.0 * ph), 2.0);
        float arm4 = pow(0.5 + 0.5 * cos(4.0 * ph + 1.1), 3.0);
        float arms = mix(arm2, max(arm2, arm4 * 0.75), smoothstep(0.45, 1.2, r)) * smoothstep(0.14, 0.34, r);
        float disk = exp(-r / 0.85) * (1.0 - smoothstep(1.7, 2.3, r));
        // dust lanes hug the inner edge of each arm and break into filaments
        float lane = pow(0.5 + 0.5 * cos(2.0 * (ph + 0.62)), 3.0) + 0.6 * pow(0.5 + 0.5 * cos(4.0 * (ph + 0.5) + 1.1), 4.0) * smoothstep(0.5, 1.2, r);
        float dust = clamp(lane * (0.35 + 1.3 * t3) * smoothstep(0.16, 0.3, r) * (1.0 - smoothstep(1.35, 2.0, r)) + (t3 - 0.55) * 0.9 * arms, 0.0, 1.0);
        // bar (turned ~25 deg from our line of sight) and bulge
        vec2 b = vec2(p.x * 0.9004 + p.y * 0.4350, -p.x * 0.4350 + p.y * 0.9004);
        float bar = exp(-(b.x * b.x / 0.085 + b.y * b.y / 0.013));
        float bulge = exp(-r * r / 0.016);
        vec3 warm = vec3(1.0, 0.52, 0.2), cream = vec3(1.0, 0.74, 0.5), rose = vec3(1.0, 0.5, 0.46), blue = vec3(0.22, 0.42, 1.0);
        vec3 armCol = mix(mix(cream, rose, smoothstep(0.22, 0.7, r)), blue, smoothstep(0.75, 1.55, r));
        float clump = 0.45 + 1.15 * fbm(q * 4.2 + 1.7);
        vec3 col = armCol * arms * disk * clump * 1.5;
        col += mix(warm, blue * 0.7, smoothstep(0.35, 1.3, r)) * disk * (0.5 + 0.5 * t2) * 0.22;     // light between the arms
        col *= mix(vec3(1.0), vec3(0.2, 0.09, 0.05), dust * 0.95);                                     // dust dims and reddens
        // pink star-forming regions strung along the arms, a few brilliant blue clusters further out
        float k1 = n2(p * 70.0), k2 = n2(p * 150.0 + 11.0);
        float knots = smoothstep(0.8, 0.96, k1) * smoothstep(0.35, 0.75, arms) * smoothstep(0.3, 0.5, r) * (1.0 - smoothstep(1.5, 2.0, r));
        col += vec3(1.0, 0.16, 0.3) * knots * (0.35 + disk * 1.6);
        col += vec3(0.6, 0.76, 1.0) * pow(k2, 26.0) * arms * 5.0 * smoothstep(0.55, 1.0, r) * (1.0 - smoothstep(1.7, 2.2, r));
        col *= 0.75 + 0.5 * n2(p * 380.0);                                                             // star grain
        col += warm * bar * 0.9 * (0.6 + 0.8 * t3) + vec3(1.0, 0.86, 0.6) * bulge * 1.5;
        col += blue * 0.05 * exp(-r / 1.1) * (1.0 - smoothstep(1.9, 2.35, r));                         // faint outer halo
        gl_FragColor = vec4(col * uFade, 1.0);
      }`,
  }));
  m.frustumCulled = false;
  m.renderOrder = -5;
  return m;
}

function galaxyStage() {
  const g = new THREE.Group(), centre = GAL_X.clone().multiplyScalar(GAL_R0);
  const at = (u, v, w) => centre.clone().addScaledVector(GAL_X, u).addScaledVector(GAL_Y, v).addScaledVector(GAL_POLE, w);
  const N = 14000, pos = new Float32Array(N * 3), col = new Float32Array(N * 3), rnd = (a = 1) => (Math.random() * 2 - 1) * a;
  const gauss = () => (Math.random() + Math.random() + Math.random() + Math.random() - 2) / 2;
  const pitch = Math.tan(12.5 * D2R), c = new THREE.Color();
  for (let i = 0; i < N; i++) {
    let u, v, w;
    const k = Math.random();
    if (k < 0.16) {                                       // bulge and bar
      const r = Math.abs(gauss()) * 0.22 * GAL_R0, a = Math.random() * Math.PI * 2;
      u = Math.cos(a) * r * 1.9; v = Math.sin(a) * r * 0.7; w = gauss() * 0.06 * GAL_R0;
      const b = 0.45;                                    // the bar is turned about 25 degrees from our line of sight
      [u, v] = [u * Math.cos(b) - v * Math.sin(b), u * Math.sin(b) + v * Math.cos(b)];
      c.setHSL(0.09 + rnd(0.02), 0.75, 0.55 + rnd(0.15));
    } else {
      const arm = k < 0.84, r = (0.22 + Math.pow(Math.random(), 0.75) * 1.85) * GAL_R0;
      let a = Math.random() * Math.PI * 2;
      if (arm) a = Math.floor(Math.random() * 4) * Math.PI / 2 + Math.log(r / (0.22 * GAL_R0)) / pitch + gauss() * 0.42;
      u = Math.cos(a) * r + gauss() * 260; v = Math.sin(a) * r + gauss() * 260; w = gauss() * 0.018 * GAL_R0;
      if (arm && Math.random() < 0.06) c.setHSL(0.94, 0.8, 0.62);                       // star-forming regions glow pink
      else c.setHSL(arm ? 0.6 + rnd(0.04) : 0.1, arm ? 0.55 : 0.35, (arm ? 0.6 : 0.4) + rnd(0.15));
    }
    const q = at(u, v, w), fall = k < 0.16 ? 0.25 : 0.22;             // the points only add depth and sparkle; the disk carries the look
    pos[i * 3] = q.x; pos[i * 3 + 1] = q.y; pos[i * 3 + 2] = q.z;
    col[i * 3] = c.r * fall; col[i * 3 + 1] = c.g * fall; col[i * 3 + 2] = c.b * fall;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const stars = new THREE.Points(geo, new THREE.PointsMaterial({ size: 1.3, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
  stars.frustumCulled = false;
  const bulge = glowSprite(0xffd9a0, GAL_R0 * 1.25, 0), haze = glowSprite(0x8fa8ff, GAL_R0 * 5.2, 0);
  bulge.position.copy(centre); haze.position.copy(centre);
  // the Sun's path: nearly a circle, drifting in and out by a few percent and bobbing through the disk
  const pts = [];
  for (let i = 0; i <= 720; i++) {
    const a = Math.PI + (i / 720) * Math.PI * 2, r = GAL_R0 * (1 + 0.035 * Math.sin(1.35 * (a - Math.PI)));
    pts.push(at(Math.cos(a) * r, Math.sin(a) * r, 0.012 * GAL_R0 * Math.sin(2.7 * (a - Math.PI))));
  }
  const orbit = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xffe066, transparent: true, opacity: 0, depthWrite: false }));
  orbit.frustumCulled = false;
  const here = glowSprite(0xffd24a, 1, 0);
  here.material.sizeAttenuation = false;
  here.scale.setScalar(0.035);
  const tags = [[label('<span class="ss-tag" style="--c:#ffe066">Sun · you are here</span>', 'lbl'), new THREE.Vector3(0, GAL_R0 * 0.06, 0)],
                [label('<span class="ss-tag" style="--c:#ffd9a0">Milky Way · model</span>', 'lbl'), centre.clone().addScaledVector(GAL_POLE, GAL_R0 * 0.28)],
                [label('<span class="ss-tag" style="--c:#ffe066">one lap ≈ 230 million years</span>', 'lbl'), at(0, GAL_R0 * 1.06, 0)]];
  tags.forEach(([t, q]) => { t.position.copy(q); g.add(t); });
  const disk = galaxyDisk();
  disk.position.copy(centre);
  disk.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(GAL_X, GAL_Y, GAL_POLE));
  g.add(haze, disk, stars, bulge, orbit, here);
  return { group: g, centre, setFade(f) {
    disk.material.uniforms.uFade.value = f;
    stars.material.opacity = f; bulge.material.opacity = 0.12 * f; haze.material.opacity = 0.16 * f; orbit.material.opacity = 0.95 * f; here.material.opacity = f;
    g.visible = f > 0.003;
    tags.forEach(([t]) => { t.element.style.opacity = f > 0.5 ? 1 : 0; });
  } };
}
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const HELIX_DRIFT = 14;            // scene units per year along the Sun's path (see the panel note: far slower than real)
const TRAIL = 900;

export function createSolar(ctx) {
  const group = new THREE.Group();
  const backdrop = milkyWay(), galaxy = galaxyStage();
  group.add(backdrop, galaxy.group);
  let helix = false, helixT0 = 0, intro = null, introSeen = false;
  const drift = new THREE.Vector3();
  let simMs = Date.now(), speed = REAL, paused = false, scale = 0, focus = null, trans = 0, wantDist = 20, jump = false;
  let lastUi = 0, enlarge = 0;
  const last = new THREE.Vector3(), tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3(), earthAu = new THREE.Vector3();
  const mapR = (au) => (scale === 0 ? 26 * Math.sqrt(au) : 9 * au);
  const sizeK = () => (scale === 0 ? 1 : 0.3);
  const sphere = new THREE.SphereGeometry(1, 96, 64);
  const texCache = {};
  const tex = (k) => texCache[k] || (texCache[k] = PROC[k] ? Promise.resolve().then(PROC[k]) : loadTexture(`/textures/2k_${k === 'earth' ? 'earth_daymap' : k}.jpg`));

  const bodies = BODIES.map((b) => {
    const o = { ...b, au: new THREE.Vector3(), obj: new THREE.Group(), tiltG: new THREE.Group(), spin: new THREE.Group(), r: 1, rSun: 0, rEarth: 0 };
    const mat = b.id === 'sun' ? sunMaterial() : planetMaterial(b);
    o.mesh = new THREE.Mesh(sphere, mat);
    o.mesh.userData.body = o;
    o.spin.add(o.mesh);
    o.tiltG.add(o.spin);
    o.tiltG.rotation.x = -b.tilt * D2R;
    o.obj.add(o.tiltG);
    if (b.tex) tex(b.tex).then((t) => { mat.uniforms.map.value = t; mat.uniforms.uHas.value = 1; }).catch(() => {});
    if (b.atm) { o.halo = new THREE.Mesh(sphere, haloMaterial(b.atm)); o.halo.scale.setScalar(1.07); o.obj.add(o.halo); }
    if (b.rings) { o.ring = new THREE.Mesh(new THREE.RingGeometry(1.24, 2.27, 256, 1), ringMaterial()); o.ring.rotation.x = Math.PI / 2; o.tiltG.add(o.ring); }
    if (b.id === 'sun') {
      o.obj.add(glowSprite(0xffc46b, 5.2, 0.55), glowSprite(0xff8a3c, 11, 0.22), glowSprite(0xfff1d6, 2.9, 0.5));
    } else {
      o.dot = glowSprite(new THREE.Color(b.color).getHex(), 1, 0.9);      // keeps far planets visible as points of light
      o.dot.material.sizeAttenuation = false;
      o.dot.scale.setScalar(0.012);
      o.obj.add(o.dot);
    }
    o.tag = label(`<span class="ss-tag" style="--c:${b.color}">${b.name}</span>`, 'lbl');
    o.obj.add(o.tag);
    group.add(o.obj);
    return o;
  });
  const by = Object.fromEntries(bodies.map((b) => [b.id, b]));
  const trails = new THREE.Group();
  group.add(trails);
  bodies.filter((b) => b.id !== 'moon').forEach((b) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
    geo.setDrawRange(0, 0);
    b.trail = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: b.id === 'sun' ? 0xffd24a : b.color, transparent: true, opacity: b.id === 'sun' ? 1 : 0.85, depthWrite: false }));
    b.trail.frustumCulled = false;
    b.trailN = 0;
    trails.add(b.trail);
  });
  function clearTrails() { bodies.forEach((b) => { if (b.trail) { b.trailN = 0; b.trail.geometry.setDrawRange(0, 0); } }); }
  function growTrails() {
    bodies.forEach((b) => {
      if (!b.trail) return;
      const a = b.trail.geometry.attributes.position.array, p = b.obj.position, n = b.trailN;
      if (n) {
        const d = Math.hypot(a[n * 3 - 3] - p.x, a[n * 3 - 2] - p.y, a[n * 3 - 1] - p.z);
        if (d < 0.5) return;
        if (d > 400) { b.trailN = 0; }                     // time jumped: start a fresh line
      }
      if (b.trailN >= TRAIL) { a.copyWithin(0, 3); b.trailN = TRAIL - 1; }
      a.set([p.x, p.y, p.z], b.trailN * 3);
      b.trailN++;
      b.trail.geometry.setDrawRange(0, b.trailN);
      b.trail.geometry.attributes.position.needsUpdate = true;
    });
  }

  // orbits
  const orbits = new THREE.Group();
  group.add(orbits);
  function buildOrbits() {
    orbits.clear();
    const T = (simMs / 864e5 + 2440587.5 - 2451545.0) / 36525;
    bodies.forEach((b) => {
      const el = EL[b.id];
      if (!el) return;
      const pts = [], a = el[0] + el[1] * T, e = el[2] + el[3] * T, I = (el[4] + el[5] * T) * D2R, w = el[8] + el[9] * T, O = el[10] + el[11] * T;
      for (let i = 0; i <= 360; i++) {
        orbitPoint(a, e, I, (w - O) * D2R, O * D2R, (i / 360) * Math.PI * 2, tmp);
        pts.push(tmp.clone().multiplyScalar(mapR(tmp.length()) / tmp.length()));
      }
      orbits.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: b.color, transparent: true, opacity: 0.32, depthWrite: false })));
    });
  }
  // main asteroid belt: illustrative swarm on circular Keplerian orbits (not individual real asteroids)
  const N_BELT = 2600, belt = new Float32Array(N_BELT * 4);
  for (let i = 0; i < N_BELT; i++) { belt[i * 4] = 2.15 + Math.pow(Math.random(), 0.8) * 1.15; belt[i * 4 + 1] = Math.random() * Math.PI * 2; belt[i * 4 + 2] = (Math.random() - 0.5) * 0.28; belt[i * 4 + 3] = Math.random(); }
  const beltGeo = new THREE.BufferGeometry();
  beltGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N_BELT * 3), 3));
  const beltPts = new THREE.Points(beltGeo, new THREE.PointsMaterial({ color: 0xa89c8c, size: 1.4, sizeAttenuation: false, transparent: true, opacity: 0.55, depthWrite: false }));
  beltPts.frustumCulled = false;
  group.add(beltPts);

  // ---------- time dock ----------
  const dock = document.createElement('div');
  dock.id = 'timedock';
  dock.className = 'glass';
  dock.hidden = true;
  dock.innerHTML = `<button data-t="rev" title="Slower / backwards">◀◀</button><button data-t="play" class="td-play" title="Pause / play"></button><button data-t="fwd" title="Faster">▶▶</button>
    <div class="td-date"><b></b><span></span></div><div class="td-speed"></div><button data-t="now" class="td-now" title="Back to this moment"><i></i>NOW</button>`;
  document.body.appendChild(dock);
  dock.addEventListener('click', (e) => {
    const t = e.target.closest('button')?.dataset.t;
    if (t === 'rev') speed = Math.max(0, speed - 1);
    else if (t === 'fwd') speed = Math.min(SPEEDS.length - 1, speed + 1);
    else if (t === 'play') paused = !paused;
    else if (t === 'now') { simMs = Date.now(); speed = REAL; paused = false; jump = true; }
    if (t) { if (t !== 'play') paused = false; paintDock(); }
  });
  const isNow = () => !paused && speed === REAL && Math.abs(simMs - Date.now()) < 5000;
  function paintDock() {
    const d = new Date(simMs);
    dock.querySelector('.td-date b').textContent = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' });
    dock.querySelector('.td-date span').textContent = `${d.toISOString().slice(11, 19)} UTC`;
    dock.querySelector('.td-speed').textContent = paused ? 'paused' : SPEEDS[speed][1];
    dock.querySelector('.td-play').textContent = paused ? '▶' : '❚❚';
    dock.querySelector('.td-now').classList.toggle('live', isNow());
  }

  // ---------- positions ----------
  function place(dt) {
    const jd = simMs / 864e5 + 2440587.5, d = jd - 2451545.0, k = sizeK() * (1 + enlarge * 1.5);
    drift.copy(GAL_Y).multiplyScalar(helix ? HELIX_DRIFT * (simMs - helixT0) / 31557600000 : 0);
    helio(EL.earth, jd, earthAu);
    bodies.forEach((b) => {
      if (b.id === 'sun') b.au.set(0, 0, 0);
      else if (b.id !== 'moon') helio(EL[b.id], jd, b.au);
      b.r = b.id === 'sun' ? (scale === 0 ? 5 : 1.4) : 0.9 * Math.pow(b.R / 6371, 0.45) * k;
    });
    const moon = by.moon, earth = by.earth, km = moonGeo(jd, tmp);
    moon.au.copy(earthAu).addScaledVector(tmp, km / AU_KM);
    moon.distKm = km;
    bodies.forEach((b) => {
      b.rSun = b.au.length(); b.rEarth = b.au.distanceTo(earthAu);
      if (b.id === 'moon') b.obj.position.copy(earth.obj.position).addScaledVector(tmp, earth.r * 3.6 + moon.r);   // distance exaggerated with the sizes
      else if (b.rSun > 0) b.obj.position.copy(b.au).multiplyScalar(mapR(b.rSun) / b.rSun).add(drift);
      else b.obj.position.copy(drift);
      b.mesh.scale.setScalar(b.r);
      b.halo?.scale.setScalar(b.r * 1.07);
      b.ring?.scale.setScalar(b.r);
      b.tag.position.set(0, b.r * (b.ring ? 1.5 : 1.25) + 0.5, 0);
      const u = b.mesh.material.uniforms;
      if (b.id !== 'sun') {
        tmp2.copy(b.obj.position).sub(drift).normalize().negate();
        u.uSun.value.copy(tmp2); u.uTime.value += dt;
        b.halo?.material.uniforms.uSun.value.copy(tmp2);
        if (b.ring) { const ru = b.ring.material.uniforms; ru.uSun.value.copy(tmp2); ru.uCenter.value.copy(b.obj.position); ru.uR.value = b.r; }
        b.dot.material.opacity = THREE.MathUtils.clamp(ctx.camera.position.distanceTo(b.obj.position) / (b.r * 60) - 0.3, 0, 0.9);
      } else u.uTime.value += dt;
      // rotation: Earth by sidereal time (so the right side faces the Sun), the Moon keeps one face to Earth
      if (b.id === 'earth') b.spin.rotation.y = ((280.46061837 + 360.98564736629 * d) % 360) * D2R;
      else if (b.id === 'moon') b.spin.rotation.y = Math.atan2(tmp.z, -tmp.x) + Math.PI;
      else b.spin.rotation.y = ((d * 24) / b.rot % 1) * Math.PI * 2;
    });
    const p = beltGeo.attributes.position.array, yr = d / 365.25;
    for (let i = 0; i < N_BELT; i++) {
      const a = belt[i * 4], ang = belt[i * 4 + 1] + (yr / Math.pow(a, 1.5)) * Math.PI * 2, r = mapR(a);
      p[i * 3] = Math.cos(ang) * r; p[i * 3 + 1] = belt[i * 4 + 2] * r * 0.25; p[i * 3 + 2] = -Math.sin(ang) * r;
    }
    beltGeo.attributes.position.needsUpdate = true;
  }

  // ---------- facts ----------
  const au = (x) => (x < 0.01 ? `${fmt(Math.round(x * AU_KM))} km` : `${x.toFixed(3)} AU`);
  const light = (x) => { const s = x * AU_KM / C_KMS; return s < 90 ? `${s.toFixed(1)} s` : s < 5400 ? `${(s / 60).toFixed(1)} min` : `${(s / 3600).toFixed(2)} h`; };
  const dayLen = (h) => (Math.abs(h) < 48 ? `${Math.abs(h).toFixed(2)} h` : `${(Math.abs(h) / 24).toFixed(1)} days`) + (h < 0 ? ' (backwards)' : '');
  const yearLen = (d) => (d < 800 ? `${d.toFixed(1)} days` : `${(d / 365.256).toFixed(2)} years`);
  function live(b) {
    const v = EL[b.id] ? 29.7847 * Math.sqrt(Math.max(0, 2 / b.rSun - 1 / EL[b.id][0])) : null;
    return { sun: b.id === 'sun' ? '—' : au(b.rSun), earth: b.id === 'earth' ? '—' : b.id === 'moon' ? `${fmt(Math.round(b.distKm))} km` : au(b.rEarth),
             light: b.id === 'earth' ? '—' : light(b.id === 'moon' ? b.distKm / AU_KM : b.rEarth), speed: b.id === 'moon' ? '1.02 km/s (around Earth)' : v ? `${v.toFixed(2)} km/s` : '—' };
  }
  function detail(b) {
    const l = live(b), acc = b.id === 'sun' ? '' : b.id === 'moon' ? 'Position: low-precision lunar series, good to about 1.5°.' : 'Position: JPL Keplerian elements, good to about 0.2° (checked against astropy).';
    const proc = PROC[b.tex] ? 'Surface: procedural (bands and colours modelled, not a photograph).' : b.clouds ? 'Surface: real map; the clouds are procedural, not today\'s weather.' : b.id === 'mercury' ? 'Surface: stand-in cratered map (tinted Moon map).' : '';
    ctx.showDetail(`<div class="kicker" style="color:${b.color}">${esc(b.kind.toUpperCase())}</div><h3>${b.name}</h3>
      <div class="ss-props">
        <span>Distance from the Sun</span><b id="ss-sun">${l.sun}</b>
        <span>Distance from Earth</span><b id="ss-earth">${l.earth}</b>
        <span>Light takes</span><b id="ss-light">${l.light}</b>
        <span>Orbital speed now</span><b id="ss-speed">${l.speed}</b>
        <span>Radius</span><b>${fmt(Math.round(b.R))} km · ${(b.R / 6371).toFixed(b.R > 60000 ? 1 : 2)} Earths</b>
        <span>Mass</span><b>${b.mass >= 100 ? fmt(Math.round(b.mass)) : b.mass} Earths</b>
        <span>Surface gravity</span><b>${b.g} m/s² · ${(b.g / 9.81).toFixed(2)} g</b>
        <span>One rotation</span><b>${dayLen(b.rot)}</b>
        ${b.year ? `<span>One orbit</span><b>${yearLen(b.year)}</b>` : ''}
        <span>Axial tilt</span><b>${b.tilt}°</b>
      </div>
      ${b.live ? `<button class="btn ss-go" data-go="${b.live[0]}">${b.live[1]} →</button>` : ''}
      <p class="feed-dim small">${acc} ${proc} Sizes are exaggerated; ${scale === 0 ? 'distances are square-root compressed' : 'distances are to scale'}.</p>`);
    document.querySelector('#detail-body .ss-go')?.addEventListener('click', (e) => ctx.goTo(e.currentTarget.dataset.go));
  }
  function setFocus(b) {
    intro = null;
    focus = b; trans = 1.6; jump = true;
    wantDist = b.id === 'sun' ? b.r * 5 : b.r * 5.5;
    ctx.controls.minDistance = b.r * 1.3;          // zoom right down to just above the surface
    detail(b);
    ctx.onInfoChange();
  }
  function startIntro() {
    introSeen = true;
    focus = null;
    intro = { t: 0, dur: 11, from: GAL_POLE.clone().multiplyScalar(0.8).addScaledVector(GAL_X, -0.45).addScaledVector(GAL_Y, 0.25).normalize(),
              to: new THREE.Vector3(0, 62, 150).normalize(),
              said: [[0.02, 'The Milky Way. The gold dot is our Sun, 26,000 light-years from the centre.'],
                     [0.3, 'One lap around the galaxy takes the Sun about 230 million years.'],
                     [0.9, 'And this is home: the planets where they are right now.']] };
  }
  ctx.controls.addEventListener('start', () => { intro = null; });       // touch the camera and the movie hands over to you
  function tick() {
    paintDock();
    if (focus && document.querySelector('#detail.open #ss-sun')) {
      const l = live(focus);
      ['sun', 'earth', 'light', 'speed'].forEach((k) => { const el = document.getElementById(`ss-${k}`); if (el) el.textContent = l[k]; });
    }
  }

  return {
    name: 'solar',
    title: 'Solar System',
    group,
    get hoverables() { return bodies.map((b) => b.mesh); },
    home: { pos: [0, 62, 150], target: [0, 0, 0] },
    autoRotate: false,
    explodeLabel: 'Enlarge the planets',
    async enter() {
      const c = ctx.controls;
      this._lim = [c.minDistance, c.maxDistance];
      c.minDistance = 1.2; c.maxDistance = 90000;
      this._far = ctx.camera.far;
      ctx.camera.far = 400000; ctx.camera.updateProjectionMatrix();     // far enough to see the whole galaxy
      this._bloom = [ctx.bloom.threshold, ctx.bloom.strength];
      ctx.bloom.threshold = 0.9; ctx.bloom.strength = 0.75;      // only the Sun glows; lit surfaces keep their detail
      focus = null;
      buildOrbits(); place(0); paintDock();
      dock.hidden = false;
      if (!introSeen) startIntro();
    },
    leave() {
      const c = ctx.controls;
      if (this._lim) [c.minDistance, c.maxDistance] = this._lim;
      if (this._far) { ctx.camera.far = this._far; ctx.camera.updateProjectionMatrix(); }
      intro = null;
      if (this._bloom) [ctx.bloom.threshold, ctx.bloom.strength] = this._bloom;
      c.target.set(0, 0, 0);
      dock.hidden = true;
      focus = null;
    },
    update(dt, t, e) {
      enlarge = e;
      if (!paused) {
        simMs += dt * 1000 * SPEEDS[speed][0];
        if (simMs < MIN_MS || simMs > MAX_MS) { simMs = Math.min(MAX_MS, Math.max(MIN_MS, simMs)); paused = true; ctx.toast('The planet model is valid from 1800 to 2050: paused at the limit', '#fbbf24'); }
      }
      place(dt);
      const c = ctx.controls, cam = ctx.camera;
      if (helix) growTrails();
      if (intro) {                                        // the opening shot: the galaxy, then all the way in to the planets
        ctx.stopFlight?.();
        intro.t += Math.min(dt, 0.05);
        const u = Math.min(1, intro.t / intro.dur), z = smooth(0.2, 1, u);
        const dist = Math.exp(Math.log(64000) + (Math.log(162) - Math.log(64000)) * z);
        tmp.copy(galaxy.centre).multiplyScalar(0.55 * (1 - z) * (1 - z));
        tmp2.copy(intro.from).lerp(intro.to, z).normalize();
        c.target.copy(tmp);
        cam.position.copy(tmp).addScaledVector(tmp2, dist);
        intro.said.forEach((m, i) => { if (m && u >= m[0]) { ctx.toast(m[1], '#ffe066'); intro.said[i] = null; } });
        if (u >= 1) intro = null;
      }
      const far = cam.position.distanceTo(by.sun.obj.position), gf = smooth(500, 3200, far), near = 1 - smooth(1800, 6000, far);
      galaxy.setFade(gf);
      backdrop.position.copy(cam.position);              // the inside view of the galaxy travels with you and fades as you leave
      backdrop.material.uniforms.uFade.value = 1 - gf;
      orbits.visible = !helix && near > 0.02;
      orbits.children.forEach((o) => { o.material.opacity = 0.32 * near; });
      beltPts.visible = !helix && near > 0.02;
      trails.visible = helix && near > 0.02;
      bodies.forEach((b) => { b.tag.element.style.opacity = near; });
      if (focus) {
        const p = focus.obj.position;
        if (!jump) { tmp.copy(p).sub(last); cam.position.add(tmp); c.target.add(tmp); }      // ride along with the body
        jump = false;
        last.copy(p);
        if (trans > 0) {
          const k = 1 - Math.exp(-dt * 4.5);
          c.target.lerp(p, k);
          tmp.copy(cam.position).sub(c.target);
          const len = tmp.length();
          cam.position.copy(c.target).add(tmp.setLength(len + (wantDist - len) * k));
          trans -= dt;
        } else c.target.copy(p);
      }
      if (t - lastUi > 0.25) { lastUi = t; tick(); }
    },
    hover(obj) {
      const b = obj.userData.body;
      if (!b) return null;
      const l = live(b);
      return `<b>${b.name}</b><br><span>${b.id === 'sun' ? `${l.earth} from Earth` : `${l.sun} from the Sun · ${l.earth} from Earth`}<br>light takes ${l.light}</span><br><em>click / point to follow</em>`;
    },
    pick(obj) { if (obj.userData.body) setFocus(obj.userData.body); },
    next() { setFocus(bodies[(bodies.indexOf(focus) + 1) % bodies.length]); },
    panel() {
      const rows = bodies.filter((b) => b.id !== 'sun').map((b) => { const l = live(b); return `<tr class="ss-row" data-act="focus" data-body="${b.id}"><td><i style="background:${b.color}"></i>${b.name}</td><td>${l.sun}</td><td>${l.earth}</td><td>${l.light}</td></tr>`; }).join('');
      return `<div class="feed-k">SOLAR SYSTEM · ${isNow() ? 'RIGHT NOW' : new Date(simMs).toISOString().slice(0, 10)}</div>
        <table class="feed-table ss-table"><tr><th>Body</th><th>From Sun</th><th>From Earth</th><th>Light time</th></tr>${rows}</table>
        <div class="feed-list"><button class="feed-item" data-act="focus" data-body="sun"><span class="chip" style="--c:#ffd27a">star</span><b>Follow the Sun</b></button>
        <button class="feed-item" data-act="intro"><span class="chip" style="--c:#ffe066">movie</span><b>Fly in from the Milky Way</b></button>
        <button class="feed-item" data-act="free"><span class="chip" style="--c:#94a3b8">camera</span><b>Free camera · whole system</b></button></div>
        <p class="feed-dim small">Click a row or a planet to follow it. Scroll all the way out to see the Sun's place in the Milky Way (a drawn model of the galaxy, in its true orientation). In <b>Journey through the galaxy</b> the coils are real geometry but the Sun's forward speed is drawn about 90× slower than its true 230 km/s. Planet positions come from JPL's orbital elements (about 0.2°, valid 1800–2050), so you can run time forwards or backwards and see real alignments, phases and seasons. The asteroid belt is an illustrative swarm; the Milky Way is in its true direction but drawn procedurally.</p>`;
    },
    onPanel(act, el) {
      const row = el.closest?.('[data-body]') || el;
      if (act === 'focus') setFocus(by[row.dataset.body]);
      else if (act === 'intro') startIntro();
      else if (act === 'free') { focus = null; ctx.controls.target.set(0, 0, 0); ctx.flyTo(this.home, 1200); ctx.onInfoChange(); }
    },
    items: () => SCALES.map((s) => ({ label: s, sub: '' })),
    get itemIndex() { return helix ? 2 : scale; },
    selectItem(i) {
      i = ((i % 3) + 3) % 3;
      intro = null;
      helix = i === 2;
      scale = helix ? 0 : i;
      clearTrails();
      if (helix) {                                        // ride with the Sun and let time run so the coils appear
        helixT0 = simMs;
        if (speed <= REAL + 3) speed = 9;
        paused = false;
        focus = by.sun; trans = 1.6; wantDist = 230;
        ctx.toast('The planets do not circle a still Sun: it carries them through the galaxy, so each path is a coil. Forward motion is drawn about 90× slower than real.', '#ffe066');
      } else if (focus) { trans = 1.2; wantDist = focus.r * 5.5; }
      buildOrbits(); place(0); jump = true; paintDock();
    },
    info() {
      const m = by.mars, j = by.jupiter;
      return {
        kicker: 'JPL EPHEMERIS · REAL POSITIONS',
        title: focus ? focus.name : 'Solar System',
        sub: 'The planets where they actually are. Run time forwards or backwards with the bar at the bottom, click any world to ride along with it, and dive into the live feeds from the Sun, Earth and Mars.',
        stats: [['Earth → Sun', au(by.earth.rSun)], ['Earth → Mars', au(m.rEarth)], ['Light to Mars', light(m.rEarth)], ['Earth → Jupiter', au(j.rEarth)]],
        legend: bodies.filter((b) => EL[b.id]).map((b) => [b.name, b.color]),
      };
    },
  };
}
