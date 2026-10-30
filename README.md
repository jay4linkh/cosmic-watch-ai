# Cosmic Watch AI 🔭

Fetches **live data from real observatories** (Rubin, ZTF, JWST, Hubble and SDO) and runs
**computer vision** on it to find galaxies, supernovae, solar active regions and **signs of black-hole activity**.
Everything shows up on a dashboard that refreshes itself.

> Status: **v0 works end to end** on live data. The galaxy classifier is rule-based for now; see the [roadmap](#roadmap).

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

# terminal 2: dashboard at http://localhost:8501
streamlit run dashboard/app.py
```

One-off runs: `python -m cosmic_watch.pipeline --once` (all sources) or `--once --only sdo,alerts`.
The dashboard sidebar also has **Fetch now** buttons.

> **Windows tip:** if `astropy` fails with *"The filename or extension is too long"*, the project path is too deep.
> Create the venv somewhere short (e.g. `C:\venvs\cosmic-watch`).

## Dashboard tabs

- **🌌 Deep Space**: newest public JWST/Hubble images, with the AI's detections toggled on or off and star/galaxy counts
- **🕳️ Black Hole Watch**: TDE / AGN / QSO / Blazar candidates from Rubin and ZTF, with cutouts and broker links
- **📡 Rubin Live**: the latest Rubin alerts, class breakdown and cutout grid
- **☀️ Sun**: latest SDO images with sunspots and active regions marked

## Project layout

```
cosmic_watch/
├── config.py            # intervals, limits, paths
├── db.py                # SQLite: images, detections, alerts, runs
├── pipeline.py          # fetch -> detect -> store loop (CLI)
├── fetchers/
│   ├── alerts.py        # Rubin + ZTF via ALeRCE
│   ├── mast.py          # JWST + Hubble new releases
│   └── sdo.py           # latest Sun images
└── detectors/
    ├── galaxies.py      # star/galaxy detection in deep fields
    ├── stamps.py        # alert cutout rendering + difference-image score
    └── sun.py           # sunspots + active regions
dashboard/app.py         # Streamlit live dashboard
notebooks/               # exploration & training (next phase)
data/                    # downloaded images + database (git-ignored)
```

## Roadmap

- [x] **Phase 1, Fetchers**: Rubin + ZTF alerts, JWST/Hubble new releases, SDO
- [x] **Phase 2, Detection v0**: classical CV for galaxies, sunspots and alert cutouts
- [x] **Phase 3, Black-hole watch**: TDE / AGN / QSO / Blazar candidates from broker classes
- [x] **Phase 4, Live dashboard**
- [ ] **Phase 5, Trained models**: replace the rule-based galaxy labels with a CNN trained on Galaxy Zoo (or Zoobot); our own real/bogus CNN on alert cutouts
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
