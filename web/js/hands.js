// Webcam hand tracking (MediaPipe, runs fully in the browser) -> gestures for the 3D scene.
//
//   open hand ......... explode (continuous: how open your hand is)
//   pinch + move ...... rotate the view
//   two-hand pinch .... zoom (pull hands apart / together)
//   point ............. aim the cursor; hold still 1 s to select
//   snap (thumb+middle) next item
//   fist (hold) ....... back to the observatory
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const TIPS = [4, 8, 12, 16, 20];

export const SENSITIVITY = {
  relaxed: { pinch: 0.42, snap: 0.36, open: [0.95, 1.75] },
  standard: { pinch: 0.34, snap: 0.3, open: [1.0, 1.85] },
  precise: { pinch: 0.27, snap: 0.24, open: [1.05, 1.95] },
};

const d2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export class HandController {
  constructor({ video, canvas, on }) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.on = on;                  // callbacks: explode, rotate, zoom, point, pointEnd, select, snap, fist, status
    this.sens = SENSITIVITY.standard;
    this.running = false;
    this.landmarker = null;
    this.prevPinch = null;
    this.prevTwo = null;
    this.dwell = { x: 0, y: 0, t: 0 };
    this.snapCooldown = 0;
    this.fistSince = 0;
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
        minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
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

  analyse(lm) {
    const palm = d2(lm[0], lm[9]) || 1e-3;
    const fingers = [8, 12, 16, 20].map((i) => d2(lm[i], lm[0]) / palm);
    const [lo, hi] = this.sens.open;
    const openness = Math.min(1, Math.max(0, (fingers.reduce((a, b) => a + b, 0) / 4 - lo) / (hi - lo)));
    const pinch = d2(lm[4], lm[8]) / palm;
    const snap = d2(lm[4], lm[12]) / palm;
    const indexOut = fingers[0] > 1.55;
    const othersIn = fingers[1] < 1.25 && fingers[2] < 1.2 && fingers[3] < 1.15;
    // In a fist the thumb rests on the index finger, which looks like a pinch: rule it out.
    const fist = fingers.every((f) => f < 1.05);
    return {
      openness,
      // Index and middle tips sit close together, so decide by which one the thumb is nearer.
      pinching: pinch < this.sens.pinch && pinch <= snap && !fist,
      snapping: snap < this.sens.snap && snap < pinch * 0.6 && !fist,
      pointing: indexOut && othersIn && pinch > this.sens.pinch * 1.4,
      fist,
      pinchPt: this.toScreen({ x: (lm[4].x + lm[8].x) / 2, y: (lm[4].y + lm[8].y) / 2 }),
      tip: this.toScreen(lm[8]),
    };
  }

  handle(res) {
    const c = this.ctx, W = this.canvas.width, H = this.canvas.height;
    c.clearRect(0, 0, W, H);
    const hands = res.landmarks || [];
    const now = performance.now();
    if (!hands.length) {
      this.prevPinch = this.prevTwo = null;
      this.on.pointEnd?.();
      this.on.status?.('Show your hand to the camera', 0, '—');
      return;
    }
    const a = hands.map((lm) => this.analyse(lm));
    hands.forEach((lm, i) => this.draw(lm, a[i]));

    let gesture = 'open hand';
    const g = a[0];

    if (a.length === 2 && a[0].pinching && a[1].pinching) {
      gesture = 'two-hand zoom';
      const dist = Math.hypot(a[0].pinchPt.x - a[1].pinchPt.x, a[0].pinchPt.y - a[1].pinchPt.y);
      if (this.prevTwo) this.on.zoom?.(this.prevTwo / dist);
      this.prevTwo = dist;
      this.prevPinch = null;
    } else if (g.pinching) {
      gesture = 'pinch · rotate';
      this.prevTwo = null;
      if (this.prevPinch) this.on.rotate?.((g.pinchPt.x - this.prevPinch.x) / W, (g.pinchPt.y - this.prevPinch.y) / H);
      this.prevPinch = g.pinchPt;
      this.ring(g.pinchPt.x, g.pinchPt.y, 46, 1, '#fbbf24');
    } else {
      this.prevPinch = this.prevTwo = null;
      if (g.snapping && now > this.snapCooldown) {
        gesture = 'snap · next';
        this.snapCooldown = now + 1200;
        this.on.snap?.();
      } else if (g.pointing) {
        gesture = 'point';
        this.on.point?.(g.tip.x, g.tip.y);
        const moved = Math.hypot(g.tip.x - this.dwell.x, g.tip.y - this.dwell.y);
        if (moved > 28) this.dwell = { x: g.tip.x, y: g.tip.y, t: now };
        const prog = Math.min(1, (now - this.dwell.t) / 1000);
        this.ring(g.tip.x, g.tip.y, 26, prog, '#67e8f9');
        if (prog >= 1) { this.on.select?.(g.tip.x, g.tip.y); this.dwell.t = now + 800; }
      } else {
        this.on.pointEnd?.();
        this.on.explode?.(g.openness);
        gesture = g.openness > 0.6 ? 'open hand · explode' : g.openness < 0.2 ? 'closed · assemble' : 'hand';
      }
      if (g.fist) {
        if (!this.fistSince) this.fistSince = now;
        const prog = Math.min(1, (now - this.fistSince) / 1500);
        this.ring(g.tip.x, g.tip.y, 38, prog, '#f472b6');
        if (prog >= 1) { this.on.fist?.(); this.fistSince = now + 2000; }
        gesture = 'fist · hold to go home';
      } else this.fistSince = 0;
    }
    this.on.status?.(gesture, hands.length, `${Math.round(g.openness * 100)}%`);
  }

  draw(lm, info) {
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
}
