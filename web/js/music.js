// Original generative ambient score, synthesised live in the browser (Web Audio): no audio files, nothing sampled.
// Three tracks, each a different harmony, pace and colour; none of them ever repeats exactly.
const midi = (n) => 440 * Math.pow(2, (n - 69) / 12);
const TRACKS = [
  { name: 'Drift', mood: 'slow and weightless',
    chords: [[38, 45, 53, 57, 64], [34, 41, 50, 57, 60], [41, 48, 57, 60, 67], [36, 43, 52, 55, 62], [33, 40, 52, 57, 60], [38, 45, 52, 57, 65]],
    bells: [62, 65, 67, 69, 72, 74, 77, 79, 81], bar: 14, wave: 'sawtooth', cutoff: [700, 1400], pad: 0.055, sub: 26, bellGap: [1.8, 6.3], bellGain: 0.085, arp: 0 },
  { name: 'Aurora', mood: 'bright and hopeful',
    chords: [[48, 55, 64, 67, 71], [53, 60, 64, 69, 72], [45, 52, 60, 64, 67], [43, 50, 59, 62, 66], [50, 57, 62, 66, 69], [41, 48, 57, 60, 64]],
    bells: [72, 74, 76, 79, 81, 83, 84, 86, 88], bar: 9, wave: 'triangle', cutoff: [1400, 2600], pad: 0.085, sub: 36, bellGap: [0.9, 2.6], bellGain: 0.06, arp: 0.5 },
  { name: 'Abyss', mood: 'dark and vast',
    chords: [[26, 33, 38, 45], [24, 31, 36, 43], [22, 29, 34, 41], [26, 33, 41, 44], [21, 28, 33, 40]],
    bells: [50, 53, 55, 57, 60, 62], bar: 22, wave: 'sawtooth', cutoff: [260, 620], pad: 0.085, sub: 26, bellGap: [6, 15], bellGain: 0.07, arp: 0, wind: true },
];

export function createMusic() {
  let ac = null, master = null, wet = null, dry = null, sub = null, wind = null, timer = null;
  let idx = -1, nextChord = 0, chord = 0, nextBell = 0, arpStep = 0, last = null;
  let volume = 0.5;
  try { volume = Number(localStorage.getItem('cw-music-vol') ?? 0.5); } catch { /* private mode */ }

  function build() {
    ac = new (window.AudioContext || window.webkitAudioContext)();
    master = ac.createGain();
    master.gain.value = 0;
    master.connect(ac.createDynamicsCompressor()).connect(ac.destination);
    // reverb: a 6 s decaying noise impulse
    const len = ac.sampleRate * 6, ir = ac.createBuffer(2, len, ac.sampleRate);
    for (let ch = 0; ch < 2; ch++) { const d = ir.getChannelData(ch); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2.6); }
    const conv = ac.createConvolver();
    conv.buffer = ir;
    wet = ac.createGain(); wet.gain.value = 0.7;
    dry = ac.createGain(); dry.gain.value = 0.5;
    wet.connect(conv).connect(master);
    dry.connect(master);
    // sub drone with a slow swell (retuned per track)
    const o = ac.createOscillator(), g = ac.createGain(), lfo = ac.createOscillator(), lg = ac.createGain();
    g.gain.value = 0.16; lfo.frequency.value = 0.05; lg.gain.value = 0.07;
    lfo.connect(lg).connect(g.gain);
    o.connect(g).connect(dry);
    o.start(); lfo.start();
    sub = o;
    // filtered noise "wind", only audible in the dark track
    const nb = ac.createBuffer(1, ac.sampleRate * 4, ac.sampleRate), nd = nb.getChannelData(0);
    for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
    const src = ac.createBufferSource(), bp = ac.createBiquadFilter(), wl = ac.createOscillator(), wlg = ac.createGain();
    src.buffer = nb; src.loop = true;
    bp.type = 'bandpass'; bp.frequency.value = 420; bp.Q.value = 0.8;
    wl.frequency.value = 0.07; wlg.gain.value = 260;
    wl.connect(wlg).connect(bp.frequency);
    wind = ac.createGain(); wind.gain.value = 0;
    src.connect(bp).connect(wind).connect(wet);
    src.start(); wl.start();
  }

  function pad(tr, note, t, dur) {
    const f = ac.createBiquadFilter(), g = ac.createGain();
    f.type = 'lowpass'; f.Q.value = 0.7;
    f.frequency.setValueAtTime(tr.cutoff[0] * 0.4, t);
    f.frequency.linearRampToValueAtTime(tr.cutoff[0] + Math.random() * (tr.cutoff[1] - tr.cutoff[0]), t + dur * 0.5);
    f.frequency.linearRampToValueAtTime(tr.cutoff[0] * 0.35, t + dur + 5);
    const att = Math.min(5, dur * 0.4);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(tr.pad, t + att);
    g.gain.setValueAtTime(tr.pad, t + dur - 2);
    g.gain.linearRampToValueAtTime(0, t + dur + 6);
    [-7, 6].forEach((cents) => {
      const o = ac.createOscillator();
      o.type = tr.wave; o.frequency.value = midi(note); o.detune.value = cents + (Math.random() - 0.5) * 4;
      o.connect(f); o.start(t); o.stop(t + dur + 7);
    });
    f.connect(g); g.connect(wet); g.connect(dry);
  }

  function bell(tr, note, t, gain = tr.bellGain) {
    const g = ac.createGain(), pan = ac.createStereoPanner(), d = ac.createDelay(2), fb = ac.createGain();
    pan.pan.value = Math.random() * 1.4 - 0.7;
    d.delayTime.value = 0.62; fb.gain.value = 0.38;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 5);
    [[1, 1], [2.01, 0.32], [3.02, 0.1]].forEach(([m, a]) => {
      const o = ac.createOscillator(), og = ac.createGain();
      o.frequency.value = midi(note) * m; og.gain.value = a;
      o.connect(og).connect(g); o.start(t); o.stop(t + 5.2);
    });
    g.connect(pan); pan.connect(wet); pan.connect(d); d.connect(fb).connect(d); d.connect(wet);
    setTimeout(() => { fb.gain.value = 0; }, 9000);
  }

  function schedule() {
    const tr = TRACKS[idx], now = ac.currentTime;
    if (!tr) return;
    while (nextChord < now + 2) {
      last = tr.chords[chord % tr.chords.length];
      last.forEach((n, i) => pad(tr, n, nextChord + i * 0.6, tr.bar));
      if (tr.arp) for (let k = 0; k < Math.floor(tr.bar / tr.arp); k++) {          // a gentle rising arpeggio over the chord
        if (Math.random() < 0.6) bell(tr, last[(arpStep++) % last.length] + 24, nextChord + k * tr.arp, tr.bellGain * 0.45);
      }
      chord = Math.random() < 0.75 ? chord + 1 : Math.floor(Math.random() * tr.chords.length);
      nextChord += tr.bar;
    }
    while (nextBell < now + 2) {
      if (Math.random() < 0.8) { const n = tr.bells[Math.floor(Math.random() * tr.bells.length)]; bell(tr, n, nextBell); if (Math.random() < 0.35) bell(tr, n + (Math.random() < 0.5 ? 7 : 12), nextBell + 0.31); }
      nextBell += tr.bellGap[0] + Math.random() * (tr.bellGap[1] - tr.bellGap[0]);
    }
  }

  function play(i) {
    if (!ac) build();
    clearInterval(timer);
    idx = i;
    const tr = TRACKS[idx];
    if (!tr) {
      master.gain.setTargetAtTime(0, ac.currentTime, 0.6);
      setTimeout(() => { if (idx < 0) ac.suspend(); }, 3000);
      return null;
    }
    ac.resume();
    sub.frequency.setTargetAtTime(midi(tr.sub), ac.currentTime, 2);
    wind.gain.setTargetAtTime(tr.wind ? 0.05 : 0, ac.currentTime, 3);
    chord = 0; arpStep = 0;
    nextChord = ac.currentTime + 0.1; nextBell = ac.currentTime + 3;      // the previous track's pads fade out underneath: a natural crossfade
    schedule();
    timer = setInterval(schedule, 800);
    master.gain.setTargetAtTime(volume, ac.currentTime, 1.5);
    return { name: tr.name, mood: tr.mood };
  }

  return {
    tracks: TRACKS.map((t) => t.name),
    get on() { return idx >= 0; },
    get track() { return TRACKS[idx]?.name || null; },
    get volume() { return volume; },
    setVolume(v) { volume = Math.min(1, Math.max(0, v)); try { localStorage.setItem('cw-music-vol', volume); } catch { /* ignore */ } if (idx >= 0) master.gain.setTargetAtTime(volume, ac.currentTime, 0.2); },
    next() { return play(idx + 1 >= TRACKS.length ? -1 : idx + 1); },         // off -> track 1 -> 2 -> 3 -> off
    toggle() { return !!play(idx >= 0 ? -1 : 0); },
  };
}
