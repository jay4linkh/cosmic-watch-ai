// Deep zoom: the real pixels as a tiled pyramid (OpenSeadragon), so zooming in loads
// sharper tiles instead of stretching one picture.
//
//   image  a JWST / Hubble image at full resolution, with the AI's galaxies on top
//   sky    the sky around a Rubin / ZTF alert from the DESI Legacy Surveys, with the
//          alert (and any others in the field) and its host galaxy, classified by our CNN
const OSD_URL = 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js';
const COARSE_COLOR = { merger: '#f472b6', elliptical: '#fbbf24', spiral: '#67e8f9', 'edge-on': '#a78bfa' };
const CNN_GROUP = {
  Disturbed: 'merger', Merging: 'merger', 'Round smooth': 'elliptical', 'In-between round smooth': 'elliptical',
  'Cigar-shaped smooth': 'elliptical', 'Barred spiral': 'spiral', 'Unbarred tight spiral': 'spiral',
  'Unbarred loose spiral': 'spiral', 'Edge-on without bulge': 'edge-on', 'Edge-on with bulge': 'edge-on',
};
const ALERT_COLOR = { TDE: '#f472b6', AGN: '#a78bfa', QSO: '#38bdf8', Blazar: '#fbbf24', SN: '#4ade80', bogus: '#6b7a93' };
const MAX_ELLIPSES = 400;

let osdPromise = null;
function loadOSD() {
  osdPromise ||= new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = OSD_URL;
    s.onload = () => resolve(window.OpenSeadragon);
    s.onerror = () => { osdPromise = null; reject(new Error('could not load the deep-zoom viewer')); };
    document.head.appendChild(s);
  });
  return osdPromise;
}

const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const pct = (p) => `${Math.round(p * 100)}%`;
function fmtAngle(arcsec) {
  if (arcsec >= 60) return `${(arcsec / 60).toFixed(arcsec >= 600 ? 0 : 1)}′`;
  return `${arcsec.toFixed(arcsec >= 10 ? 0 : 1)}″`;
}
const cnnBars = (top) => top.map(([k, p]) =>
  `<div class="cnn-row"><span>${k}</span><i style="--p:${pct(p)}"></i><b>${pct(p)}</b></div>`).join('');

export function createDeepZoom({ onClose, toast }) {
  const root = el('div', 'dz');
  root.id = 'dz';
  root.innerHTML = `
    <div class="dz-view"></div>
    <div class="dz-top">
      <button class="dz-close">← Back to 3D</button>
      <div class="dz-title"><span class="dz-kicker"></span><b></b></div>
      <button class="dz-ai" hidden title="Add a super-resolution level past the real pixels (E)">✨ AI enhance: off</button>
      <button class="dz-toggle on" title="Show / hide what the AI found (snap)">AI overlay</button>
    </div>
    <div class="dz-card glass"></div>
    <div class="dz-scale"><div class="dz-res"></div><div class="dz-fov"></div><div class="dz-src"></div><div class="dz-credit"></div></div>
    <div class="dz-loading">Loading tiles…</div>`;
  document.body.appendChild(root);
  const $ = (s) => root.querySelector(s);

  let OSD = null, viewer = null, spec = null, info = null;
  let overlays = [], overlaysOn = true, armed = false, openedAt = 0, token = 0, sciTimer = null;
  let aiModel = null;              // null | 'swinir' | 'esrgan'
  const AI_ORDER = [null, 'swinir', 'esrgan'];

  $('.dz-close').addEventListener('click', () => close());
  $('.dz-toggle').addEventListener('click', () => toggleOverlay());
  $('.dz-ai').addEventListener('click', () => cycleAI());

  function ensureViewer() {
    if (viewer) return viewer;
    viewer = OSD({
      element: $('.dz-view'),
      showNavigationControl: false,
      showNavigator: true, navigatorPosition: 'BOTTOM_RIGHT', navigatorSizeRatio: 0.14,
      navigatorBackground: '#000', navigatorBorderColor: 'rgba(103,232,249,0.4)', navigatorDisplayRegionColor: '#67e8f9',
      maxZoomPixelRatio: 4,              // a little past native resolution (smoothly interpolated), not into mush
      minZoomImageRatio: 0.6,
      visibilityRatio: 0.4,
      animationTime: 0.9, springStiffness: 9, blendTime: 0.25,
      imageLoaderLimit: 10, immediateRender: false, smoothTileEdgesMinZoom: 1.1,
      zoomPerScroll: 1.35, zoomPerClick: 1,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
      crossOriginPolicy: false,
    });
    viewer.addHandler('animation', readout);
    viewer.addHandler('zoom', () => {
      readout();
      // zooming back out past the whole picture returns to the 3D view
      if (armed && viewer.viewport.getZoom() < viewer.viewport.getHomeZoom() * 0.7) close();
    });
    viewer.addHandler('tile-load-failed', () => { $('.dz-loading').textContent = 'Some tiles are still on their way…'; });
    viewer.addHandler('tile-loaded', () => $('.dz-loading').classList.remove('show'));
    return viewer;
  }

  function tileSource(i, minSide = 256) {
    const levels = Math.floor(Math.log2(Math.max(i.width, i.height) / minSide));
    const real = (l, x, y) => i.tiles.replace('{level}', l).replace('{x}', x).replace('{y}', y);
    if (aiModel && i.ai?.available) {
      // Twice the size: levels up to maxLevel are the real tiles unchanged, maxLevel + 1 is AI-made.
      return {
        width: i.width * 2, height: i.height * 2, tileSize: i.tileSize, tileOverlap: i.tileOverlap,
        minLevel: Math.max(0, i.maxLevel - levels), maxLevel: i.maxLevel + 1,
        getTileUrl: (l, x, y) => (l > i.maxLevel
          ? i.ai.tiles.replace('{model}', aiModel).replace('{x}', x).replace('{y}', y)
          : real(l, x, y)),
      };
    }
    return {
      width: i.width, height: i.height, tileSize: i.tileSize, tileOverlap: i.tileOverlap,
      minLevel: Math.max(0, i.maxLevel - levels), maxLevel: i.maxLevel, getTileUrl: real,
    };
  }

  // Reopen the same picture with a new tile source, keeping the view where it was.
  function reopen(my, after) {
    const bounds = viewer.viewport.getBounds();
    viewer.addOnceHandler('open', () => {
      if (my !== token) return;
      viewer.viewport.fitBounds(bounds, true);
      viewer.viewport.applyConstraints();      // e.g. AI switched off while zoomed past the real-pixel limit
      imageOverlays(my);
      sourceLine();
      readout();
      after?.();
    });
    viewer.open(tileSource(info));
  }

  function aiButton() {
    const b = $('.dz-ai');
    b.hidden = !(spec?.kind === 'image' && info?.ai?.available);
    b.classList.toggle('on', !!aiModel);
    b.textContent = aiModel ? `✨ AI: ${info.ai.models[aiModel].name}` : '✨ AI enhance: off';
    b.title = aiModel ? `${info.ai.models[aiModel].note}. Click for the next model / off (E)` : 'Add a super-resolution level past the real pixels (E)';
  }

  function cycleAI() {
    if (spec?.kind !== 'image' || !info?.ai?.available) return;
    aiModel = AI_ORDER[(AI_ORDER.indexOf(aiModel) + 1) % AI_ORDER.length];
    aiButton();
    reopen(token, () => {
      if (aiModel) toast?.(`AI enhance on: ${info.ai.models[aiModel].name} (${info.ai.models[aiModel].note}). Zoom past the real pixels to see it; that detail is invented.`, '#f472b6');
      else toast?.('AI enhance off: real pixels only', '#4ade80');
    });
  }

  // ---------- readout: field of view, and whether we are past the real resolution ----------
  function readout() {
    if (!viewer || !info) return;
    const vp = viewer.viewport;
    const iz = vp.viewportToImageZoom(vp.getZoom(true));            // screen px per image px
    const w = viewer.container.clientWidth;
    if (spec.kind === 'sky') {
      const native = iz * info.oversample;                            // screen px per survey px
      $('.dz-fov').textContent = `field ${fmtAngle(w / iz * info.arcsec_per_px)} wide · ${(info.arcsec_per_px / iz).toFixed(3)}″ per screen px`;
      setRes(native);
    } else {
      const k = aiModel && info.ai?.available ? 2 : 1;              // tile source is 2x the real grid with AI on
      const real = iz * k;                                            // screen px per real pixel
      $('.dz-fov').textContent = `${Math.round(w / real)} of ${info.width} px across · ${real >= 1 ? `${real.toFixed(1)}× magnified` : `1:${(1 / real).toFixed(1)}`}`;
      setRes(real, k > 1 ? info.ai.models[aiModel].name : null);
    }
  }
  function setRes(native, ai = null) {
    const r = $('.dz-res');
    root.classList.toggle('ai-active', !!ai && native > 1.15);
    if (ai && native > 1.15) { r.className = 'dz-res ai'; r.textContent = `AI-enhanced (${ai}) ×${native.toFixed(1)} · invented detail, not measured`; }
    else if (native > 1.15) { r.className = 'dz-res beyond'; r.textContent = `beyond native resolution ×${native.toFixed(1)} · smooth interpolation, no new detail`; }
    else { r.className = 'dz-res native'; r.textContent = native > 0.85 ? 'native resolution · every pixel real' : 'real pixels · zoom for more detail'; }
  }

  // ---------- overlays ----------
  function clearOverlays() { overlays.forEach((o) => viewer?.removeOverlay(o)); overlays = []; }
  function addOverlay(element, location, placement) {
    element.style.display = overlaysOn ? '' : 'none';
    viewer.addOverlay({ element, location, placement: placement ?? OSD.Placement.CENTER });
    overlays.push(element);
  }

  async function imageOverlays(my) {
    const det = await fetch(`/api/images/${spec.imageId}/detections?limit=3000`).then((r) => r.json());
    if (my !== token) return;
    const aspect = info.height / info.width;
    const galaxies = det.rows.filter((r) => !/star/.test(det.labels[r[5]])).slice(0, MAX_ELLIPSES);
    let n = 0;
    for (const r of galaxies) {
      const cnn = r[7] >= 0 ? det.cnn_classes[r[7]] : null;
      const color = cnn ? COARSE_COLOR[CNN_GROUP[cnn]] : 'rgba(230,237,247,0.55)';
      const a = Math.max(r[2] * 2.6, 0.0015), b = Math.max(r[3] * 2.6, 0.0015);
      const box = el('button', 'dz-ell');
      box.title = cnn ? `${cnn} · ${pct(r[8])} (our CNN)` : det.labels[r[5]];
      const ring = el('i');
      ring.style.cssText = `border-color:${color};transform:rotate(${r[4]}rad)`;
      box.appendChild(ring);
      if (cnn && r[8] >= 0.6 && a > 0.004) box.appendChild(el('span', 'dz-ell-tag', cnn));
      box.addEventListener('click', (e) => { e.stopPropagation(); focusDetection(r, det, cnn); });
      addOverlay(box, new OSD.Rect(r[0] - a, r[1] * aspect - b, 2 * a, 2 * b));
      n++;
    }
    const withCnn = galaxies.filter((r) => r[7] >= 0).length;
    card(`<div class="dz-card-k">WHAT THE AI FOUND</div>
      <p><b>${n}</b> galaxies outlined · <b>${withCnn}</b> classified by our CNN.</p>
      <div class="dz-legend">${Object.entries(COARSE_COLOR).map(([k, c]) => `<span><i style="border-color:${c}"></i>${k}</span>`).join('')}
      <span><i style="border-color:rgba(230,237,247,0.55)"></i>not classified</span></div>
      <p class="dz-hint">Click (or point at) an outline to zoom in on it. Snap / <kbd>O</kbd> hides the overlay.</p>`);
  }

  async function focusDetection(r, det, cnn) {
    const aspect = info.height / info.width, s = Math.max(r[2] * 14, 0.01);
    viewer.viewport.fitBounds(new OSD.Rect(r[0] - s, r[1] * aspect - s, 2 * s, 2 * s));
    let body = `<div class="dz-card-k">${det.labels[r[5]].toUpperCase()} · ${(r[2] * info.width * 2).toFixed(0)} px</div>`;
    if (cnn) {
      const c = await fetch(`/api/detections/${r[9]}/cnn`).then((x) => x.json()).catch(() => null);
      if (c?.available) body += `<div class="cnn-box"><div class="cnn-head">OUR CNN · ${c.model}</div>${cnnBars(c.top)}</div>`;
    }
    card(body + '<p class="dz-hint">Zoom out to see the whole image again.</p>');
  }

  async function skyOverlays(my) {
    for (const m of info.alerts) {
      const main = m.oid === spec.oid;
      const b = el('button', `dz-alert${main ? ' main' : ''}`);
      b.style.setProperty('--c', ALERT_COLOR[m.cls] || '#e6edf7');
      b.title = `${m.cls} · ${m.survey === 'lsst' ? 'Rubin' : 'ZTF'} ${m.oid}`;
      b.innerHTML = `<i></i>${main ? `<span>${m.cls} · ${m.survey === 'lsst' ? 'Rubin' : 'ZTF'} alert</span>` : ''}`;
      b.addEventListener('click', (e) => { e.stopPropagation(); flyToMark(m); });
      addOverlay(b, new OSD.Point(m.x, m.y));
    }
    const others = info.alerts.length - 1;
    card(`<div class="dz-card-k">${spec.cls} · ${spec.survey === 'lsst' ? 'RUBIN' : 'ZTF'} ALERT</div>
      ${spec.stamp ? `<img class="dz-stamp" src="${spec.stamp}" alt="alert cut-outs">
      <div class="dz-stamp-k"><span>tonight</span><span>before</span><span>change</span></div>` : ''}
      <p>The tiny cut-out is everything the alert contains. Behind it: the same patch of sky, deep and sharp,
      from the <b>${info.layer.toUpperCase()}</b> survey (taken years earlier, before this event).</p>
      <div class="dz-host">Looking for the host galaxy…</div>
      ${others > 0 ? `<p class="dz-hint">${others} other alert${others > 1 ? 's' : ''} we know of ${others > 1 ? 'are' : 'is'} in this field.</p>` : ''}`);
    const h = await fetch(`/api/dz/host?ra=${spec.ra}&dec=${spec.dec}`).then((r) => r.json()).catch(() => null);
    if (my !== token) return;
    const box = $('.dz-host');
    if (!h?.found) { box.innerHTML = `<b>No host galaxy found.</b> ${h?.reason || ''}`; return; }
    const W = info.width, ap = info.arcsec_per_px;
    const x = (W / 2 - h.dx_arcsec / ap) / W, y = (W / 2 - h.dy_arcsec / ap) / W;
    const r = Math.max(h.size_arcsec * 1.4, 2.5) / ap / W;
    const ring = el('div', 'dz-hostring');
    addOverlay(ring, new OSD.Rect(x - r, y - r, 2 * r, 2 * r));
    box.innerHTML = `<div class="cnn-box"><div class="cnn-head">HOST GALAXY · OUR CNN ${h.model ? `(${h.model})` : ''}</div>
      ${h.top ? cnnBars(h.top) : ''}</div>
      <p class="dz-hint">${h.sep_arcsec < 0.6 ? 'The event sits right on this galaxy\'s core' : `The event is ${h.sep_arcsec.toFixed(1)}″ from this galaxy's centre`}
      · galaxy ${h.size_arcsec.toFixed(1)}″ across. ${['TDE', 'AGN', 'QSO', 'Blazar'].includes(spec.cls) && h.sep_arcsec < 1 ? 'A nuclear event: what you\'d expect from a central black hole.' : ''}</p>`;
  }

  function flyToMark(m) {
    const s = 60 / info.arcsec_per_px / info.width;   // a 2-arcminute window
    viewer.viewport.fitBounds(new OSD.Rect(m.x - s, m.y - s, 2 * s, 2 * s));
  }

  // ---------- science data: preview JPEG -> FITS mosaic, swapped in place ----------
  function sourceLine() {
    const sci = info?.science, box = $('.dz-src');
    if (!info || spec?.kind !== 'image') { box.textContent = ''; return; }
    const tel = info.source === 'jwst' ? 'JWST i2d' : 'Hubble drizzled';
    if (info.data === 'science') { box.className = 'dz-src sci'; box.textContent = `science data · ${tel} FITS, float32 · no compression`; return; }
    box.className = 'dz-src';
    if (sci?.state === 'fetching') box.textContent = `preview JPEG · fetching the science FITS from MAST… ${Math.round((sci.progress || 0) * 100)}%`;
    else if (sci?.state === 'processing') box.textContent = 'preview JPEG · processing the science FITS…';
    else if (sci?.state === 'failed') box.textContent = `preview JPEG · science data unavailable (${sci.error || 'MAST error'})`;
    else box.textContent = 'preview JPEG (compressed)';
  }

  async function upgradeToScience(my) {
    clearTimeout(sciTimer);
    if (spec?.kind !== 'image' || !info?.science || ['ready', 'unavailable', 'failed'].includes(info.science.state)) return;
    try {
      info.science = await fetch(`/api/dz/image/${spec.imageId}/science`, { method: 'POST' }).then((r) => r.json());
    } catch { return; }
    if (my !== token) return;
    sourceLine();
    if (info.science.state === 'ready') {
      const fresh = await fetch(`/api/dz/image/${spec.imageId}`).then((r) => r.json());
      if (my !== token) return;
      info = fresh;
      reopen(my, () => toast?.('Switched to the science data: uncompressed JWST/Hubble pixels', '#4ade80'));
      return;
    }
    if (info.science.state !== 'failed') sciTimer = setTimeout(() => upgradeToScience(my), 4000);
  }

  function card(html) { const c = $('.dz-card'); c.innerHTML = html; c.classList.toggle('show', !!html); }

  // ---------- open / close ----------
  async function open(s) {
    const my = ++token;
    spec = s;
    armed = false;
    aiModel = null;
    $('.dz-ai').hidden = true;
    root.classList.add('open');
    document.body.classList.add('deepzoom');
    $('.dz-kicker').textContent = s.kicker || '';
    $('.dz-title b').textContent = s.title || '';
    $('.dz-loading').textContent = 'Loading tiles…';
    $('.dz-loading').classList.add('show');
    card('');
    try {
      OSD ||= await loadOSD();
      info = s.kind === 'sky'
        ? await fetch(`/api/dz/sky?ra=${s.ra}&dec=${s.dec}`).then((r) => r.json())
        : await fetch(`/api/dz/image/${s.imageId}`).then((r) => r.json());
      if (my !== token) return;
      if (s.kind === 'sky' && !info.available) throw new Error(info.reason || 'no survey coverage here');
      $('.dz-credit').textContent = info.credit || '';
      ensureViewer();
      clearOverlays();
      viewer.addOnceHandler('open', () => {
        if (my !== token) return;
        const vp = viewer.viewport;
        if (s.kind === 'sky') {
          const f = 150 / (info.arcsec_per_px * info.width);   // start 5 arcmin across, centred on the alert
          vp.fitBounds(new OSD.Rect(0.5 - f, 0.5 - f, 2 * f, 2 * f), true);
          skyOverlays(my);
        } else {
          const aspect = info.height / info.width, r = s.r ?? 0.5, rh = s.rh ?? r;
          vp.fitBounds(new OSD.Rect((s.u ?? 0.5) - r, (s.v ?? 0.5) * aspect - rh, 2 * r, 2 * rh), true);
          imageOverlays(my);
          upgradeToScience(my);
          aiButton();
        }
        sourceLine();
        openedAt = performance.now();
        setTimeout(() => { if (my === token) armed = true; }, 1200);
        readout();
      });
      viewer.open(tileSource(info));
    } catch (e) {
      toast?.(`Deep zoom unavailable: ${e.message}`, '#fb7185');
      close();
    }
  }

  function close() {
    if (!root.classList.contains('open')) return;
    token++;
    armed = false;
    clearTimeout(sciTimer);
    root.classList.remove('open');
    document.body.classList.remove('deepzoom');
    clearOverlays();
    viewer?.close();
    spec = info = null;
    onClose?.();
  }

  function toggleOverlay() {
    overlaysOn = !overlaysOn;
    overlays.forEach((o) => { o.style.display = overlaysOn ? '' : 'none'; });
    $('.dz-toggle').classList.toggle('on', overlaysOn);
  }

  return {
    get isOpen() { return root.classList.contains('open'); },
    open, close, toggleOverlay, cycleAI,
    // hand / keyboard control
    zoomBy(f) { if (viewer && info) { viewer.viewport.zoomBy(f); viewer.viewport.applyConstraints(); } },
    panBy(dx, dy) {
      if (!viewer || !info) return;
      const b = viewer.viewport.getBounds();
      viewer.viewport.panBy(new OSD.Point(-dx * b.width * 1.6, -dy * b.height * 1.6));
    },
    // open hand = keep zooming in; a closed (not fist) hand zooms back out
    handOpen(v) {
      if (!viewer || !info || performance.now() - openedAt < 800) return;
      if (v > 0.65) this.zoomBy(1 + (v - 0.65) * 0.06);
      else if (v < 0.12) this.zoomBy(0.985);
    },
  };
}
