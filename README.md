# Cosmic Watch AI 🔭

Fetches **live data from real observatories and spacecraft** (Rubin, ZTF, JWST, Hubble, SDO, SOHO, GOES, LIGO/Virgo/KAGRA,
NASA's GCN, the Mars rovers and NASA/JPL's asteroid trackers) and runs **computer vision and our own neural networks** on it:
galaxies, supernovae, black-hole activity, solar flares and CMEs, gravitational waves, gamma-ray bursts, Martian geology
and asteroids passing Earth.
Everything shows up in a **3D observatory** you can fly through, or control with **your bare hands** through the webcam.

> Status: **v2**: 13 live feeds, two CNNs trained on this laptop's GPU (galaxies, Mars terrain), deep zoom into real
> pixels, a learning agent, six 3D worlds and hand-gesture control.

---

## Live data sources

| Source | What we fetch | How fresh | Access |
|---|---|---|---|
| **Rubin (LSST)** via [ALeRCE](https://alerce.online) | Alert cutouts (science / template / difference) + stamp classifier class | Minutes after detection | Free, no key |
| **ZTF** via ALeRCE | Black-hole candidates (TDE, AGN, QSO, Blazar) and supernovae, with cutouts | Last night | Free, no key |
| **JWST + Hubble** via [MAST](https://mast.stsci.edu) | Newly public science images (released in the last 3 days) | Daily | Free, no key |
| **SDO** ([NASA](https://sdo.gsfc.nasa.gov)) | Full-disk Sun in visible light and 3 extreme-UV channels | ~15 min | Free, no key |
| **GOES + NOAA SWPC** | X-ray flux (flare classes), flare list, Kp index, solar wind at L1, active regions with flare odds | 1 min | Free, no key |
| **SOHO LASCO C2/C3** | Coronagraph images of the corona out to 30 solar radii (CMEs) | ~15-30 min | Free, no key |
| **LIGO / Virgo / KAGRA** via [GraceDB](https://gracedb.ligo.org) | Significant real gravitational-wave events: type (BBH/BNS/NSBH), distance, sky map | Minutes (during observing runs) | Free, no key |
| **NASA GCN Circulars** | Gamma-ray bursts, Einstein Probe X-ray flashes, IceCube neutrinos, with positions and redshifts | Minutes to hours | Free, no key |
| **Mars rovers** (Perseverance, Curiosity) | The newest raw images from science and navigation cameras | Every sol | Free, no key |
| **NASA/JPL SSD APIs** | Asteroid close approaches (10 lunar distances), Sentry impact-risk list, fireballs | Hourly | Free, no key |
| **ISS** ([wheretheiss.at](https://wheretheiss.at)) | Live position, speed, altitude | 5 s | Free, no key |
| **SIMBAD (CfA mirror) · Gaia DR3 (ESA) · NED** | What catalogues know about an alert's position: name, type, parallax, redshift | On demand | Free, no key |

Rubin's full camera frames are not public in real time; only alert cutouts are. Full frames come in yearly data releases.

## What the AI does

| Data | Computer vision |
|---|---|
| **JWST / Hubble deep fields** | Background model + source extraction ([SEP](https://github.com/sep-developers/sep)), then each source is labelled **star / galaxy / elongated galaxy / large galaxy** from its half-light radius and ellipticity relative to the image's PSF. Footprint edges and bright-star diffraction spikes are suppressed. |
| **Rubin / ZTF alerts** | Renders science / template / difference cutouts and scores the **difference image** (central signal-to-noise: is there a real new point of light?). Classes come from ALeRCE's classifiers. |
| **Black holes** | No telescope images a black hole directly, so we flag what they **do**: **TDEs** (a star being torn apart), **AGN** flares, **quasars** and **blazars**. |
| **Sun** | Fits the solar disk, removes limb darkening, and detects **sunspots** (visible light) and **active regions** (extreme UV), plus an activity index. |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# terminal 1: keep fetching (alerts every 10 min, Sun every 15 min, JWST/Hubble every 6 h)
python -m cosmic_watch.pipeline

# terminal 2: the 3D observatory at http://localhost:8000
uvicorn cosmic_watch.server:app --port 8000
```

Or run it all in one process: `RUN_PIPELINE=1 uvicorn cosmic_watch.server:app` (that's what the deploy does).
One-off fetches: `python -m cosmic_watch.pipeline --once` (all sources) or `--once --only sdo,alerts`.
The older analyst dashboard is still there: `streamlit run dashboard/app.py` (port 8501).

> **Windows tip:** if `astropy` fails with *"The filename or extension is too long"*, the project path is too deep.
> Create the venv somewhere short (e.g. `C:\venvs\cosmic-watch`).

## 🌞 Space weather · 〰️ gravitational waves · 💥 bursts · 🔴 Mars · 🪨 Near Earth

- **The Sun** view has a live space-weather panel: GOES X-rays with flare classes, the week's flares, Kp (and where aurora
  may be visible), solar-wind speed and Bz, NOAA's flare odds per active region, and the SOHO LASCO coronagraphs as a movie.
  Our **CME detector** runs a running difference on consecutive coronagraph frames and flags large new bright fronts.
- **Black Holes** lists the real significant **LIGO/Virgo/KAGRA** events (mock-data-challenge "MS..." test events and
  retractions are filtered out): inspect one for a 3D model of the merger (inspiral, chirp, flash, ringdown over rippling
  spacetime) next to the real sky map. Below it, **GCN** gamma-ray bursts, X-ray flashes and neutrinos; one with a position
  opens in deep zoom on the survey sky, with its redshift and a host-galaxy search. LIGO's O4 run ended in November 2025;
  new events appear automatically when the next run starts.
- **Mars** shows a 3D Mars with Perseverance (Jezero crater) and Curiosity (Gale crater) at their real sites and today's raw
  photos above them. Each photo is classified by **our Mars CNN** (below) and measured with classical CV (sky fraction via
  regions connected to the top edge, terrain roughness, colour). Click one for deep zoom, NASA's caption and the CNN's call.
- **Near Earth** shows Earth turning with the real time of day, the Moon's orbit, the ISS live, every asteroid within
  10 lunar distances in a two-month window (distances square-root scaled, directions illustrative), fireballs on the globe
  and the Sentry impact-risk table.
- **Alerts** opened in deep zoom now show their **light curve** (ALeRCE) and what **SIMBAD, Gaia and NED** know about
  the spot: known name and type, redshift and light-travel distance (Planck 2018), and whether the "transient" sits on a
  Milky Way star (Gaia parallax).

The agent watches all of it: M/X flares, geomagnetic storms, CME candidates, new gravitational-wave events, gamma-ray
bursts and neutrinos, asteroids inside the Moon's orbit and Mars science finds all become suggestions.

## 🔴 Our Mars CNN

`cosmic_watch/ml/mars_train.py` fine-tunes an ImageNet ConvNeXt-Tiny on NASA/JPL's labelled Curiosity images
([MSL v2.1, doi:10.5281/zenodo.4033453](https://doi.org/10.5281/zenodo.4033453): 6,820 images, 19 classes such as drill
hole, layers, light-toned veins, sand, wheel tracks, distant landscape, rover parts), trained on an RTX 3060 in ~10 minutes.
On the official **held-out test split** (different sols, unseen terrain): **80.0% accuracy, 89.0% top-3**, balanced
accuracy 61% (rare geology classes such as float rocks, veins and layers are the weak spot, so the agent only flags them
when the CNN is confident). Exported to ONNX with float16 weights (56 MB, same accuracy as float32 on a 400-image check).
Curiosity-trained, so expect some drop on Perseverance's cameras.

```bash
python -m cosmic_watch.ml.mars_train     # put msl-labeled-data-set-v2.1 in data/datasets/mars/ first
```

## The 3D observatory

Built with Three.js (WebGL + bloom), served by FastAPI. Four worlds, all from live data:

| View | What you see | "Explode" does |
|---|---|---|
| **Observatory** | Telescopes → AI detectors → discoveries as glowing planets; every pulse along the orbits is real data arriving | spreads the system apart |
| **Deep Field** | The newest public JWST/Hubble image, with every AI detection as a particle | pulls the image apart: galaxies recede by brightness, stars come forward, callouts mark the biggest galaxies |
| **Black Holes** | A live accretion disk with real Rubin/ZTF candidate cut-outs (TDE / AGN / QSO / Blazar) in orbit | widens the orbits |
| **The Sun** | The latest SDO image wrapped on a sphere, AI-detected active regions ringed | raises magnetic loops off the surface and pushes out the solar wind |

**Inspect mode**: click (or point and hold) any galaxy or black-hole card. You get an illustrative 3D model built from the AI's
measurements next to the real telescope pixels. Zoom past halfway and the **real image fills the screen** (sharper crops load
as you zoom; drag or pinch turns it). Then **classify it yourself**; your labels are saved as training data.

**🔭 Deep zoom**: keep zooming past the end of Inspect (or press **Deep zoom** / `Z`) and you drop into the real pixels as a
**tiled image pyramid** (OpenSeadragon), so zooming in loads sharper tiles instead of stretching one picture:

- **JWST / Hubble**: the full-resolution image, with an astronomical asinh display stretch (brightness only, no new detail)
  and every galaxy the AI found outlined and coloured by our CNN's call. Click (or point at) one to zoom onto it.
  Tiles start from MAST's JPEG preview and **upgrade in place to the science mosaic** (JWST `_i2d.fits` / Hubble
  `_drc.fits`, float32) the first time you deep-zoom an image: same pixel grid, but no JPEG blocks, so the faint sky
  shows its real grain. The download is 150–300 MB (a few minutes from MAST); the stretched result is cached as a
  lossless PNG and the FITS is deleted.
- **Rubin / ZTF alerts**: an alert carries only a tiny cut-out, so the background is the same patch of sky, deep and sharp,
  from the [DESI Legacy Imaging Surveys](https://www.legacysurvey.org/) (fetched tile by tile and cached). The alert is
  pinned at its exact position, other alerts in the field show up too, and the **host galaxy** is found and classified by our CNN.
- **🤖 Click any outline and the Cosmic Agent explains it** (`cosmic_watch/explain.py`): a verdict first ("likely a
  galaxy: barred spiral", "probably glowing gas, not a galaxy", "a star", "possibly a satellite trail") with the evidence
  behind it (size in arcseconds, how centrally concentrated the light is, which emission line the filter isolates,
  what the field is, how sure the CNN is), then background knowledge. This catches the CNN's blind spot: it only knows
  galaxies, so it calls nebula filaments "Disturbed". With Claude credentials set (`ANTHROPIC_API_KEY`), Claude Opus 5.5
  also writes a short explanation grounded in the same facts; without them the built-in knowledge is used.
- **✨ AI enhance** (button or `E`, JWST/Hubble): adds one extra 2× level past the real pixels, made by a
  super-resolution network on your GPU: **SwinIR ×2** (sharper, stays close to the data) or **Real-ESRGAN ×2**
  (cleaner, but smooths away faint sources). That detail is *invented*: the view gets a pink frame and an
  "AI-enhanced · invented detail" label, and enhanced pixels never feed the CNN, detectors or anomaly scores.
  Needs PyTorch + [spandrel](https://github.com/chaiNNer-org/spandrel) (`requirements-train.txt`); weights download on
  first use. On a server without them the button simply doesn't appear.
- A readout always says whether you are looking at real pixels or **beyond native resolution** (smoothly interpolated, no
  invented detail). Zoom back out past the whole picture to return to 3D. Hand mode: two-hand pinch zooms, open hand
  keeps zooming in, pinch-drag pans, snap toggles the AI overlay, fist goes back.

Plus a live event trace, source health, a guided tour, and keyboard shortcuts (`1`–`4` views, `Space` explode, `←/→` items, `H` hand mode).

### ✋ Hand mode

Press **Hand mode** (or `H`): your camera becomes the background and the scene floats over it like a hologram.
Hand tracking runs **entirely in your browser** with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker). No video leaves your machine.

| Gesture | Action |
|---|---|
| ✋ Open hand | explode the scene (continuous: how open your hand is) |
| ✊ Close hand | assemble; **hold** to fly home |
| 🤏 Pinch + move | rotate |
| 🙌 Two-hand pinch | zoom (pull apart / together) |
| ☝️ Point, hold 1 s | select what you point at |
| 🫰 Snap (thumb + middle) | next image / channel / class |

The camera needs `localhost` or HTTPS. Good, even lighting helps; there's a sensitivity setting in the side panel.

## 🤖 The Cosmic Agent: anomaly detection that learns what you like

`cosmic_watch/anomaly.py` runs after every fetch:

| Source | Method | Flags |
|---|---|---|
| **Sun** (AIA 131/94/171/193/304 Å) | Frame differencing between consecutive images: exposure-normalised, a big absolute rise **and** ratio **and** area (thresholds tuned on live data so solar rotation doesn't trigger it), plus robust z-scores of each channel's activity vs its own history | sudden brightenings (possible flares), activity jumps |
| **Deep fields** | **Isolation Forest** (unsupervised ML) over every detection's brightness, size, ellipticity and size vs PSF | the odd ones out: possible lensing arcs, mergers, edge-on disks, artifacts |
| **Alerts** | High-probability TDEs; **classifier vs CV disagreement** (broker says *bogus*, our difference-image S/N says real) | candidates worth a second look |

`cosmic_watch/agent.py` then picks what to show you with a **Thompson-sampling bandit** (reinforcement learning for
choosing between options): each anomaly kind has a Beta(α, β) "interest" belief; **Look** is a reward, **Not now** is
not. It explores kinds it's unsure about and exploits the ones you keep opening. The agent card in the app says
*"This looks interesting… It could be X, Y or Z. Want to look?"* and **Look** flies you straight to the object in Inspect mode.

> Why a bandit and not "full" RL? There's no environment to act in or long-horizon reward here, just a choice of
> what to show next and a yes/no signal from you. That's exactly the contextual-bandit setting.

## 🧠 Our galaxy CNN

`cosmic_watch/ml/` trains our own deep-learning classifier and runs it on every new JWST/Hubble image.

- **Data:** [Galaxy10 DECaLS](https://astronn.readthedocs.io/en/latest/galaxy10.html): 17,736 galaxies labelled by
  Galaxy Zoo volunteers into 10 classes (disturbed, merging, round/in-between/cigar smooth, barred spiral,
  tight/loose spiral, edge-on with/without bulge), **plus your own Inspect-mode labels**.
- **Model v2 (in use):** [Zoobot](https://github.com/mwalmsley/zoobot)'s greyscale ConvNeXt-nano encoder, already
  pretrained on hundreds of thousands of Galaxy Zoo galaxies, fine-tuned at 224×224 (encoder at a lower learning rate
  than the new head; mixed precision on an RTX 3060, ~20 min). Predictions average the four 90° rotations.
- **Model v1:** ImageNet-pretrained ResNet-18 on 128×128 cut-outs (kept for comparison: `--arch resnet18`).
  Both use telescope-agnostic augmentation: any rotation, flips, zoom, contrast stretch, PSF blur, noise.

  | Validation (2,656 galaxies) | v1 ResNet-18 | **v2 Zoobot ConvNeXt** |
  |---|---|---|
  | Accuracy (10 classes) | 82.8 % | **88.7 %** |
  | Top-2 accuracy | 93.3 % | **97.2 %** |
  | Coarse (spiral / elliptical / edge-on / merger) | 89.5 % | **92.7 %** |
  | Recall, hardest classes: disturbed / loose spiral | 38 % / 61 % | **57 % / 78 %** |

- **Your labels** enter through a coarse loss (spiral / elliptical / edge-on / merger) on the summed class probabilities.
- **Inference:** exported to ONNX (`models/galaxy_cnn_v2.onnx`, 60 MB) and run with ONNX Runtime on CPU, with no PyTorch
  needed in production (~145 ms per galaxy with rotation averaging). The newest model in `models/` is used; set
  `COSMIC_CNN=galaxy_cnn_v1` to pin the old one. Metrics, classes and checksums are in the `.json` model cards.
- **Host galaxies:** in deep zoom, the galaxy hosting a Rubin/ZTF alert is classified on Legacy Survey imaging, the same
  survey Galaxy10 comes from, so there's no domain gap there.
- **In the app:** Inspect shows the CNN's top-4 probabilities and builds the 3D model from its call (barred spirals get
  a bar, mergers become two colliding disks with tidal tails). The agent flags confident mergers.

```bash
pip install -r requirements-train.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
python -m cosmic_watch.ml.train --arch zoobot   # put Galaxy10_DECals.h5 in data/datasets/galaxy10/ first
python -m cosmic_watch.ml.infer      # classify all stored images
```

> Domain shift: training images are ground-based colour DECaLS cut-outs; our live images are single-filter space
> telescope previews. Expect lower accuracy on JWST/HST than the validation numbers, especially for tiny sources.
> Your labels are how it gets better on our data.

## Computer vision & AI used in this project

| Technique | Where |
|---|---|
| Background estimation + source extraction (segmentation, deblending) | `detectors/galaxies.py` (SEP) |
| Morphological measurement: half-light radius, ellipticity, PSF estimation | `detectors/galaxies.py` |
| Masking: footprint detection, edge rejection, diffraction-spike suppression | `detectors/galaxies.py` |
| Disk fitting, limb-darkening correction, adaptive thresholding, connected components, morphological opening | `detectors/sun.py` |
| Image differencing / change detection, robust statistics (median, MAD, σ-clipping) | `detectors/stamps.py`, `anomaly.py` |
| **Deep learning: our own CNN (Zoobot ConvNeXt fine-tuned; ResNet-18 baseline) for galaxy morphology, rotation test-time augmentation** | `ml/train.py`, `ml/infer.py` |
| Unsupervised anomaly detection (Isolation Forest) | `anomaly.py` |
| Reinforcement learning (Thompson-sampling bandit) | `agent.py` |
| Deep-learning hand-landmark CNN (MediaPipe), One-Euro filtering, gesture state machine with hysteresis | `web/js/hands.js` |
| Running-difference change detection on coronagraph frames (CME detector) | `fetchers/spaceweather.py` |
| Our own Mars terrain CNN (ConvNeXt-Tiny transfer learning), sky segmentation, texture/roughness | `ml/mars_train.py`, `ml/mars.py` |
| Catalogue cross-matching (SIMBAD, Gaia DR3, NED), light curves, cosmological distances | `context.py` |
| Multi-resolution image pyramids (deep zoom), gnomonic (TAN) sky projection, asinh display stretch, super-resolution (SwinIR transformer, Real-ESRGAN GAN), labelled as invented detail | `deepzoom.py`, `web/js/deepzoom.js` |
| Human-in-the-loop labelling (Galaxy Zoo style) → training data | Inspect mode, `/api/labels` |
| Third-party ML classifiers (ALeRCE stamp + light-curve classifiers) | alert classes |

Next step: an autoencoder for image-level anomalies, and retraining the CNN as your labels grow.

## Deploy (live link)

The repo includes a Render blueprint (`render.yaml`): in Render choose **New + → Blueprint**, pick this repo, and it builds and
starts the web app with the pipeline running inside it. Render gives you HTTPS, so hand mode works on the live link.
On the free plan the service sleeps when idle and its disk is temporary, so data is re-fetched after each wake-up (about 5 min).

## Project layout

```
cosmic_watch/
├── config.py            # intervals, limits, paths
├── db.py                # SQLite: images, detections, alerts, runs
├── pipeline.py          # fetch -> detect -> store loop (CLI)
├── server.py            # FastAPI: JSON API + serves the 3D web app
├── anomaly.py           # Sun flares/activity, deep-field outliers, alert disagreements
├── agent.py             # the Cosmic Agent: suggestions + bandit learning
├── deepzoom.py          # tile pyramids (JWST/HST + Legacy Survey sky) + host-galaxy CNN
├── explain.py           # the agent's "what is this, really?" for a clicked detection
├── context.py           # alert light curves + SIMBAD / Gaia / NED cross-match
├── fetchers/
│   ├── alerts.py        # Rubin + ZTF via ALeRCE
│   ├── mast.py          # JWST + Hubble new releases
│   ├── spaceweather.py  # GOES, NOAA SWPC, SOHO LASCO (+ CME detector)
│   ├── gw.py            # LIGO/Virgo/KAGRA (GraceDB)
│   ├── gcn.py           # GCN circulars: bursts, neutrinos, X-ray flashes
│   ├── mars.py          # Perseverance + Curiosity raw images
│   ├── neo.py           # JPL close approaches, Sentry, fireballs, ISS
│   └── sdo.py           # latest Sun images
└── detectors/
    ├── galaxies.py      # star/galaxy detection in deep fields
    ├── stamps.py        # alert cutout rendering + difference-image score
    └── sun.py           # sunspots + active regions
web/                     # 3D observatory (Three.js) + hand tracking (MediaPipe)
├── index.html
├── css/style.css
└── js/
    ├── main.js          # app shell, camera, picking, live polling, tour
    ├── hands.js         # webcam hand tracking -> gestures
    ├── inspect.js       # Inspect mode: 3D models + real pixels + full-screen zoom
    ├── deepzoom.js      # tiled deep-zoom viewer with AI overlays
    ├── fx.js            # glow shaders, starfield, helpers
    └── views/           # observatory · deep · blackholes · sun
dashboard/app.py         # Streamlit analyst dashboard
render.yaml              # one-click deploy
notebooks/               # exploration & training (next phase)
data/                    # downloaded images + database (git-ignored)
```

## Roadmap

- [x] **Phase 1, Fetchers**: Rubin + ZTF alerts, JWST/Hubble new releases, SDO
- [x] **Phase 2, Detection v0**: classical CV for galaxies, sunspots and alert cutouts
- [x] **Phase 3, Black-hole watch**: TDE / AGN / QSO / Blazar candidates from broker classes
- [x] **Phase 4, Live dashboard**
- [x] **Phase 4b, 3D observatory + hand-gesture control**
- [x] **Phase 4c, Inspect mode, your own classifications, anomaly detection + learning agent**
- [x] **Phase 5, Our own CNN galaxy classifier** (Galaxy10 DECaLS + your labels)
- [ ] **Phase 5b**: our own real/bogus CNN on alert cut-outs
- [ ] **Phase 6**: gravitational-lens finder, anomaly detection (autoencoder), galaxy similarity search (DINOv2/CLIP)
- [ ] Rubin via Fink as a second broker (its API was timing out when v0 was built)

## Known limits (v0)

- Small, faint galaxies can be labelled as stars (size-based rules); counts are indicative, not science-grade.
- Detection runs on MAST's JPEG previews, not the calibrated FITS data.
- ZTF cutouts are from each object's first alert, not the latest one.
- Broker classifications are probabilistic; a "TDE" candidate is a candidate.

## Credits

Data: Vera C. Rubin Observatory and ZTF alerts via the ALeRCE broker; JWST and HST data from MAST (STScI);
solar images courtesy of NASA/SDO and the AIA and HMI science teams; SOHO/LASCO (ESA & NASA); GOES and space-weather
data from NOAA SWPC; gravitational-wave alerts from the LIGO Scientific Collaboration, Virgo and KAGRA via GraceDB;
GCN Circulars (NASA); Mars raw images NASA/JPL-Caltech (with ASU and MSSS); asteroid data NASA/JPL SSD/CNEOS; ISS position
wheretheiss.at; Mars training data Lu & Wagstaff (NASA/JPL); SIMBAD (CDS, via the CfA mirror), Gaia DR3 (ESA), NED (IPAC);
planet textures Solar System Scope (CC BY 4.0); sky backgrounds from the DESI Legacy
Imaging Surveys (legacysurvey.org). Galaxy CNN v2 starts from [Zoobot](https://github.com/mwalmsley/zoobot)
(Walmsley et al.), and the deep-zoom viewer is [OpenSeadragon](https://openseadragon.github.io/).
