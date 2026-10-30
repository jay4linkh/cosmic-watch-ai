# Cosmic Watch AI 🔭

Fetches **live data from real observatories** (Rubin, ZTF, JWST, Hubble and SDO) and runs
**computer vision** on it to find galaxies, supernovae, solar active regions and **signs of black-hole activity**.
Everything shows up in a **3D observatory** you can fly through, or control with **your bare hands** through the webcam.

> Status: **v1**: live data, computer vision, 3D web app and hand-gesture control all work end to end.
> The galaxy classifier is rule-based for now; see the [roadmap](#roadmap).

---

## Live data sources

| Source | What we fetch | How fresh | Access |
|---|---|---|---|
| **Rubin (LSST)** via [ALeRCE](https://alerce.online) | Alert cutouts (science / template / difference) + stamp classifier class | Minutes after detection | Free, no key |
| **ZTF** via ALeRCE | Black-hole candidates (TDE, AGN, QSO, Blazar) and supernovae, with cutouts | Last night | Free, no key |
| **JWST + Hubble** via [MAST](https://mast.stsci.edu) | Newly public science images (released in the last 3 days) | Daily | Free, no key |
| **SDO** ([NASA](https://sdo.gsfc.nasa.gov)) | Full-disk Sun in visible light and 3 extreme-UV channels | ~15 min | Free, no key |

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
- **Model:** ImageNet-pretrained ResNet-18 fine-tuned on grey 128×128 cut-outs (mixed precision on an RTX 3060).
  Augmentation makes it telescope-agnostic: any rotation, flips, zoom, contrast stretch, PSF blur, noise.
- **Your labels** enter through a coarse loss (spiral / elliptical / edge-on / merger) on the summed class probabilities.
- **Inference:** exported to ONNX (`models/galaxy_cnn_v1.onnx`) and run with ONNX Runtime on CPU, with no PyTorch needed in production.
  Metrics, classes and the file checksum are in `models/galaxy_cnn_v1.json`.
- **In the app:** Inspect shows the CNN's top-4 probabilities and builds the 3D model from its call (barred spirals get
  a bar, mergers become two colliding disks with tidal tails). The agent flags confident mergers.

```bash
pip install -r requirements-train.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
python -m cosmic_watch.ml.train      # downloads nothing itself: put Galaxy10_DECals.h5 in data/datasets/galaxy10/
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
| **Deep learning: our own CNN (ResNet-18 transfer learning) for galaxy morphology** | `ml/train.py`, `ml/infer.py` |
| Unsupervised anomaly detection (Isolation Forest) | `anomaly.py` |
| Reinforcement learning (Thompson-sampling bandit) | `agent.py` |
| Deep-learning hand-landmark CNN (MediaPipe), One-Euro filtering, gesture state machine with hysteresis | `web/js/hands.js` |
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
├── fetchers/
│   ├── alerts.py        # Rubin + ZTF via ALeRCE
│   ├── mast.py          # JWST + Hubble new releases
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
solar images courtesy of NASA/SDO and the AIA and HMI science teams.
