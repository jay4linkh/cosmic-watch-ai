// Webcam hand tracking (MediaPipe, runs fully in the browser) -> gestures for the 3D scene.
//
//   open hand ......... explode (or zoom in, while inspecting a galaxy)
//   pinch + move ...... rotate the view
//   two-hand pinch .... zoom (pull hands apart / together)
//   point ............. aim the cursor; hold still to select / inspect
//   snap (thumb+middle) next item (next galaxy while inspecting)
//   fist (hold) ....... back out (leave inspection, then go home)
//
// Stability: landmarks pass through a One-Euro filter, gestures use hysteresis
// (harder to leave than to enter) and must hold for a few frames before they switch.
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const TIPS = [4, 8, 12, 16, 20];

export const SENSITIVITY = {
  relaxed: { pinch: 0.42, snap: 0.36, open: [0.95, 1.75], hold: 2 },
  standard: { pinch: 0.34, snap: 0.3, open: [1.0, 1.85], hold: 3 },
  precise: { pinch: 0.27, snap: 0.24, open: [1.05, 1.95], hold: 4 },
};
const RELEASE = 1.4;           // a pinch ends only when fingers open 40 % past the entry threshold
const DWELL_MS = 900;          // point-and-hold time to select
const DWELL_RADIUS = 42;       // px the fingertip may wander while dwelling
const ROTATE_DEADZONE = 0.0015;
const ROTATE_GAIN = 0.75;

const d2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// One-Euro filter: smooths jitter when still, stays responsive when moving fast.
class OneEuro {
  constructor(minCutoff = 1.4, beta = 0.04, dCutoff = 1.0) { Object.assign(this, { minCutoff, beta, dCutoff }); this.x = null; this.dx = 0; this.t = 0; }
  static alpha(cutoff, dt) { const r = 2 * Math.PI * cutoff * dt; return r / (r + 1); }
  filter(x, t) {
    if (this.x === null) { this.x = x; this.t = t; return x; }
    const dt = Math.max(1e-3, (t - this.t) / 1000);
    this.t = t;
    const dx = (x - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (dx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (x - this.x);
    return this.x;
  }
}

class HandFilter {
  constructor() { this.f = Array.from({ length: 21 }, () => [new OneEuro(), new OneEuro(), new OneEuro()]); }
  apply(lm, t) { return lm.map((p, i) => ({ x: this.f[i][0].filter(p.x, t), y: this.f[i][1].filter(p.y, t), z: this.f[i][2].filter(p.z || 0, t) })); }
}

export class HandController {
  constructor({ video, canvas, on }) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.on = on;   // callbacks: explode, rotate, zoom, point, pointEnd, select, snap, fist, presence, status
    this.sens = SENSITIVITY.standard;
    this.running = false;
    this.landmarker = null;
    this.filters = [new HandFilter(), new HandFilter()];
    this.latched = [{}, {}];          // hysteresis state per hand
    this.active = 'none';             // the gesture currently in charge
    this.candidate = 'none';
    this.candidateFrames = 0;
    this.prevPinch = null;
    this.prevTwo = null;
    this.rotVel = { x: 0, y: 0 };
    this.openSmooth = 0;
    this.dwell = { x: 0, y: 0, t: 0 };
    this.snapCooldown = 0;
    this.fistSince = 0;
    this.present = false;
    this.lastSeen = 0;
    this.lastVideoTime = -1;
    this.fps = 0;
    this._frames = 0;
    this._fpsT = performance.now();
  }

  setSensitivity(key) { this.sens = SENSITIVITY[key] || SENSITIVITY.standard; }

  async start() {
    if (!this.landmarker) {
      this.on.status?.('Loading hand-tracking model…');
      const files = await FilesetResolver.forVisionTasks(WASM);
      this.landmarker = await HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO', numHands: 2,
        minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.55, minTrackingConfidence: 0.55,
      });
    }
    this.on.status?.('Starting camera…');
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.running = true;
    this.on.status?.('Show your hand to the camera');
    this.loop();
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.setPresent(false);
  }

  loop = () => {
    if (!this.running) return;
    const v = this.video;
    if (v.readyState >= 2 && v.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = v.currentTime;
      const res = this.landmarker.detectForVideo(v, performance.now());
      this.handle(res);
      this._frames++;
      const now = performance.now();
      if (now - this._fpsT > 1000) { this.fps = Math.round(this._frames * 1000 / (now - this._fpsT)); this._frames = 0; this._fpsT = now; }
    }
    requestAnimationFrame(this.loop);
  };

  // Landmarks are normalised to the camera frame; the video is shown mirrored and object-fit: cover.
  toScreen(p) {
    const W = this.canvas.width, H = this.canvas.height;
    const vw = this.video.videoWidth || 1280, vh = this.video.videoHeight || 720;
    const s = Math.max(W / vw, H / vh);
    const ox = (W - vw * s) / 2, oy = (H - vh * s) / 2;
    return { x: W - (p.x * vw * s + ox), y: p.y * vh * s + oy };
  }

  setPresent(p) {
    if (p !== this.present) { this.present = p; this.on.presence?.(p); }
  }

  analyse(lm, k) {
    const L = this.latched[k];
    const palm = d2(lm[0], lm[9]) || 1e-3;
    const fingers = [8, 12, 16, 20].map((i) => d2(lm[i], lm[0]) / palm);
    const [lo, hi] = this.sens.open;
    const openness = Math.min(1, Math.max(0, (fingers.reduce((a, b) => a + b, 0) / 4 - lo) / (hi - lo)));
    const pinch = d2(lm[4], lm[8]) / palm;
    const snap = d2(lm[4], lm[12]) / palm;
    // In a fist the thumb rests on the index finger, which looks like a pinch: rule it out.
    const fist = fingers.every((f) => f < (L.fist ? 1.15 : 1.02));
    // Hysteresis: easy to start, hard to drop by accident mid-gesture.
    const pinchThr = L.pinching ? this.sens.pinch * RELEASE : this.sens.pinch;
    L.pinching = pinch < pinchThr && pinch <= snap * 1.05 && !fist;
    const indexOut = fingers[0] > (L.pointing ? 1.4 : 1.55);
    const othersIn = fingers[1] < 1.3 && fingers[2] < 1.25 && fingers[3] < 1.2;
    L.pointing = indexOut && othersIn && pinch > this.sens.pinch * 1.3;
    L.fist = fist;
    return {
      openness, fist,
      pinching: L.pinching,
      snapping: snap < this.sens.snap && snap < pinch * 0.6 && !fist,
      pointing: L.pointing,
      pinchPt: this.toScreen({ x: (lm[4].x + lm[8].x) / 2, y: (lm[4].y + lm[8].y) / 2 }),
      tip: this.toScreen(lm[8]),
    };
  }

  // A new gesture must be seen for `hold` consecutive frames before it takes over.
  settle(raw) {
    if (raw === this.active) { this.candidate = raw; this.candidateFrames = 0; return this.active; }
    if (raw === this.candidate) this.candidateFrames++;
    else { this.candidate = raw; this.candidateFrames = 1; }
    const need = raw === 'snap' ? 2 : this.sens.hold;
    if (this.candidateFrames >= need) {
      this.active = raw;
      this.prevPinch = this.prevTwo = null;
      this.rotVel = { x: 0, y: 0 };
    }
    return this.active;
  }

  handle(res) {
    const c = this.ctx, W = this.canvas.width, H = this.canvas.height;
    c.clearRect(0, 0, W, H);
    const raw = res.landmarks || [];
    const now = performance.now();
    if (!raw.length) {
      // brief dropouts (motion blur) shouldn't end the interaction
      if (now - this.lastSeen > 1200) {
        this.setPresent(false);
        this.active = this.candidate = 'none';
        this.filters = [new HandFilter(), new HandFilter()];
        this.latched = [{}, {}];
      }
      this.prevPinch = this.prevTwo = null;
      this.on.pointEnd?.();
      this.on.status?.('Show your hand to the camera', 0, '—');
      return;
    }
    this.lastSeen = now;
    this.setPresent(true);
    const hands = raw.slice(0, 2).map((lm, i) => this.filters[i].apply(lm, now));
    const a = hands.map((lm, i) => this.analyse(lm, i));
    hands.forEach((lm) => this.draw(lm));
    const g = a[0];
    this.openSmooth += (g.openness - this.openSmooth) * 0.35;

    let rawGesture = 'hand';
    if (a.length === 2 && a[0].pinching && a[1].pinching) rawGesture = 'zoom';
    else if (g.pinching) rawGesture = 'rotate';
    else if (g.snapping) rawGesture = 'snap';
    else if (g.pointing) rawGesture = 'point';
    else if (g.fist) rawGesture = 'fist';
    const gesture = this.settle(rawGesture);

    let label = 'hand';
    if (gesture !== 'point') this.on.pointEnd?.();
    if (gesture !== 'fist') this.fistSince = 0;

    switch (gesture) {
      case 'zoom': {
        label = 'two-hand zoom';
        if (a.length < 2) break;
        const dist = Math.hypot(a[0].pinchPt.x - a[1].pinchPt.x, a[0].pinchPt.y - a[1].pinchPt.y);
        if (this.prevTwo) {
          const f = this.prevTwo / dist;
          if (Math.abs(f - 1) > 0.004) this.on.zoom?.(Math.min(1.08, Math.max(0.92, f)));
        }
        this.prevTwo = dist;
        a.forEach((h) => this.ring(h.pinchPt.x, h.pinchPt.y, 30, 1, '#fbbf24'));
        break;
      }
      case 'rotate': {
        label = 'pinch · rotate';
        if (this.prevPinch) {
          let dx = (g.pinchPt.x - this.prevPinch.x) / W, dy = (g.pinchPt.y - this.prevPinch.y) / H;
          if (Math.abs(dx) < ROTATE_DEADZONE) dx = 0;
          if (Math.abs(dy) < ROTATE_DEADZONE) dy = 0;
          this.rotVel.x += (dx - this.rotVel.x) * 0.5;
          this.rotVel.y += (dy - this.rotVel.y) * 0.5;
          this.on.rotate?.(this.rotVel.x * ROTATE_GAIN, this.rotVel.y * ROTATE_GAIN);
        }
        this.prevPinch = g.pinchPt;
        this.ring(g.pinchPt.x, g.pinchPt.y, 46, 1, '#fbbf24');
        break;
      }
      case 'snap':
        label = 'snap · next';
        if (now > this.snapCooldown) { this.snapCooldown = now + 1300; this.on.snap?.(); }
        break;
      case 'point': {
        label = 'point · hold to select';
        this.on.point?.(g.tip.x, g.tip.y);
        const moved = Math.hypot(g.tip.x - this.dwell.x, g.tip.y - this.dwell.y);
        if (moved > DWELL_RADIUS) this.dwell = { x: g.tip.x, y: g.tip.y, t: now };
        const prog = Math.min(1, Math.max(0, (now - this.dwell.t) / DWELL_MS));
        this.ring(g.tip.x, g.tip.y, 28, prog, '#67e8f9');
        this.crosshair(g.tip.x, g.tip.y);
        if (prog >= 1) { this.on.select?.(this.dwell.x, this.dwell.y); this.dwell.t = now + 900; }
        break;
      }
      case 'fist': {
        label = 'fist · hold to go back';
        this.on.explode?.(0);
        if (!this.fistSince) this.fistSince = now;
        const prog = Math.min(1, (now - this.fistSince) / 1200);
        this.ring(g.tip.x, g.tip.y, 38, prog, '#f472b6');
        if (prog >= 1) { this.on.fist?.(); this.fistSince = now + 1800; }
        break;
      }
      default:
        this.on.explode?.(this.openSmooth);
        label = this.openSmooth > 0.6 ? 'open hand · expand' : this.openSmooth < 0.2 ? 'closed · assemble' : 'hand';
    }
    this.on.status?.(label, hands.length, `${Math.round(this.openSmooth * 100)}%`);
  }

  draw(lm) {
    const c = this.ctx;
    const pts = lm.map((p) => this.toScreen(p));
    c.save();
    c.lineCap = 'round';
    c.shadowColor = 'rgba(103,232,249,0.9)';
    c.shadowBlur = 14;
    c.strokeStyle = 'rgba(165,243,252,0.85)';
    c.lineWidth = 2.5;
    c.beginPath();
    for (const [i, j] of BONES) { c.moveTo(pts[i].x, pts[i].y); c.lineTo(pts[j].x, pts[j].y); }
    c.stroke();
    for (let i = 0; i < pts.length; i++) {
      const tip = TIPS.includes(i);
      c.fillStyle = tip ? '#fde68a' : 'rgba(224,242,254,0.9)';
      c.beginPath();
      c.arc(pts[i].x, pts[i].y, tip ? 5 : 3, 0, Math.PI * 2);
      c.fill();
    }
    c.restore();
  }

  ring(x, y, r, progress, color) {
    const c = this.ctx;
    c.save();
    c.shadowColor = color;
    c.shadowBlur = 18;
    c.strokeStyle = color;
    c.globalAlpha = 0.35;
    c.lineWidth = 2;
    c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.stroke();
    c.globalAlpha = 1;
    c.lineWidth = 4;
    c.beginPath(); c.arc(x, y, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * progress); c.stroke();
    c.restore();
  }

  crosshair(x, y) {
    const c = this.ctx;
    c.save();
    c.strokeStyle = 'rgba(103,232,249,0.9)';
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(x - 10, y); c.lineTo(x - 4, y); c.moveTo(x + 4, y); c.lineTo(x + 10, y);
    c.moveTo(x, y - 10); c.lineTo(x, y - 4); c.moveTo(x, y + 4); c.lineTo(x, y + 10);
    c.stroke();
    c.restore();
  }
}
