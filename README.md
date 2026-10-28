# Cosmic Watch AI 🔭

Fetch newly published images from real observatories and run AI on them to find
**galaxies, supernovae, gravitational lenses and black-hole activity**.

> Status: **idea / planning stage**. Nothing is built yet. This repo holds the plan.

---

## The idea

Telescopes like Rubin, ZTF, JWST and Hubble publish data every day. Cosmic Watch AI will:

1. **Fetch** the newest public images and alerts automatically
2. **Detect & classify** what's in them with computer vision
3. **Show** the results on a live dashboard: "today's new sky + what the AI found"

### A note on black holes
No telescope sees a black hole directly. We detect their **signatures** instead:
- **TDEs** (tidal disruption events): a black hole tearing apart a star, which shows up as a sudden flare in a galaxy core
- **AGN flares**: an active supermassive black hole brightening
- **Quasars**: extremely bright, distant black-hole-powered galaxy cores (point-like, unusual colour)

---

## Data sources

| Source | What we get | Delay | Access |
|---|---|---|---|
| **Rubin (LSST)** via ALeRCE / Fink / Lasair | Alert cutouts (science, template, difference) of anything that changed | ~minutes | Free APIs |
| **ZTF** via ALeRCE | Same alert format, years of labelled history | ~minutes | `pip install alerce` |
| **JWST / Hubble** via MAST | Full images, many public immediately | daily | `astroquery.mast` |
| **DESI Legacy Survey / hips2fits** | Any patch of sky on demand (deep galaxy fields) | static | HTTP cutouts |
| **SDO** (bonus) | Full Sun images | ~15 min | Direct URLs |

Note: Rubin's full camera frames are not public in real time (only alerts are); full data comes in yearly releases.

---

## Architecture (planned)

```
Every few minutes:  Rubin/ZTF alert brokers  ──► alert classifier
                    (SN, TDE, AGN flare, variable, bogus)

Every day:          MAST new JWST/Hubble images ──► tile ──► galaxy detector (YOLO)
                    (spiral / elliptical / merger / lens / quasar candidate)

                                     ▼
                    Database ──► Live dashboard
```

## What the AI will detect

- [ ] **Galaxies**: detect + classify (spiral / elliptical / merger), trained on Galaxy Zoo labels
- [ ] **Black-hole events**: TDE / AGN flares from alerts, quasar candidates in images
- [ ] **Supernovae**: new point sources next to galaxies
- [ ] **Gravitational lenses**: arcs around massive galaxies
- [ ] **Anomalies**: an autoencoder flags anything unusual

---

## Roadmap

- [ ] **Phase 1, Fetchers**: pull newly public JWST/Hubble images from MAST, plus the latest Rubin/ZTF alerts
- [ ] **Phase 2, Galaxy detector**: YOLO on tiled deep-field images
- [ ] **Phase 3, Black-hole event filter**: use broker classes (TDE/AGN/SN) plus our own cutout CNN
- [ ] **Phase 4, Dashboard**: latest images, bounding boxes, live "black hole flare candidates" feed
- [ ] **Phase 5 (stretch)**: galaxy similarity search (DINOv2/CLIP embeddings), text-to-sky search

## Project layout

```
cosmic-watch-ai/
├── fetchers/     # MAST, ALeRCE/Fink, Legacy Survey clients
├── models/       # galaxy detector, alert classifier, anomaly model
├── pipeline/     # scheduled jobs tying fetch → detect → store
├── dashboard/    # web UI
├── notebooks/    # exploration & training
└── data/         # local cache (git-ignored)
```

## Useful links

- ALeRCE: https://alerce.science
- Fink: https://fink-broker.org
- Lasair: https://lasair.lsst.ac.uk
- MAST: https://mast.stsci.edu
- Legacy Survey viewer: https://www.legacysurvey.org/viewer
- hips2fits: https://alasky.cds.unistra.fr/hips-image-services/hips2fits
- Galaxy Zoo data: https://data.galaxyzoo.org
- Rubin gallery: https://rubinobservatory.org/gallery
