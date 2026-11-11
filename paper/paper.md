---
title: 'Cosmic Watch: a live, explorable observatory that applies computer vision and machine learning to public astronomy data streams'
tags:
  - Python
  - astronomy
  - computer vision
  - deep learning
  - time-domain astronomy
  - galaxy morphology
  - solar physics
  - Mars
authors:
  - name: Jay Gupta
    orcid: 0009-0002-7127-6146
    affiliation: 1
affiliations:
  - name: SVKM's NMIMS Mukesh Patel School of Technology Management & Engineering, Mumbai, India
    index: 1
date: 11 November 2026
bibliography: paper.bib
---

# Summary

`Cosmic Watch` is an open-source Python and WebGL application that continuously
collects newly released public data from observatories and spacecraft, analyses it
with computer vision and trained neural networks, checks the results against
archival catalogues, and presents everything in one interactive 3D environment
(\autoref{fig:app}). Its thirteen live feeds include Rubin [@Ivezic2019] and ZTF
[@Bellm2019] alerts via the ALeRCE broker [@Forster2021], new JWST and Hubble images
from MAST, SDO solar images [@Pesnell2012; @Lemen2012], SOHO/LASCO coronagraph data
[@Brueckner1995], NOAA space weather, gravitational-wave events, NASA GCN circulars,
Mars rover images and NASA/JPL near-Earth-object data.

Deep-field sources are extracted with `SEP` [@Barbary2016; @Bertin1996] and
classified by a galaxy-morphology CNN fine-tuned from `Zoobot` [@Walmsley2023];
solar images are searched for sunspots, active regions and coronal holes; rover
images are classified by a terrain CNN; and detections are cross-matched against
Gaia DR3 [@Gaia2023], Pan-STARRS1 [@Chambers2016], SDSS, the Legacy Surveys
[@Dey2019] and SIMBAD [@Wenger2000], with every claim stored alongside its evidence.
Users can open any detection in a tiled deep-zoom view of the real pixels and label
it, and those labels drive an active-learning loop that promotes a retrained model
only when it passes held-out checks.

![The running application, from a live screen recording. (a) Observatory overview:
each feed and detector is a node. (b) Inspect mode for a JWST detection: CNN
probabilities beside the real pixels. (c) SDO/AIA 171 Å with detected active regions
and live NOAA space weather. (d) A Perseverance image classified by the Mars CNN, with
rock candidates outlined.\label{fig:app}](figures/fig1_app.png)

# Statement of need

Most of this data is free and public within minutes to days of being taken, but it
is spread across many archives, formats and APIs. A student, educator or amateur
astronomer who wants to follow the sky (what JWST released today, whether the Sun is
flaring, which transients Rubin flagged last night) must visit many services and has
no easy way to run modern analysis on what they find. Brokers and archive tools
excel at their own data type but are not built to put many data types side by side
or to explain a result to a non-specialist.

`Cosmic Watch` provides one reproducible pipeline that runs on a laptop with a
consumer GPU, needs no API keys, and keeps a clear line between measurement and
model output; for example, super-resolved pixels from SwinIR [@Liang2021] or
Real-ESRGAN [@Wang2021] are always labelled "invented detail" and never reach a
detector or classifier. It is aimed at teaching and outreach, student and
citizen-science projects that need a readable end-to-end example, and quick-look
triage of new releases and alerts.

# Design

A FastAPI server and a scheduled pipeline write to SQLite; each source has a fetcher
and each data type a detector. On calibrated JWST/HST mosaics, detection runs in
several passes (cold, hot and binned low-surface-brightness), and each pass is also
run on the sign-flipped image, where any detection is noise; thresholds rise until
that spurious rate is below 3 %, and completeness is measured by injecting synthetic
galaxies. The galaxy classifier is a Zoobot ConvNeXt-nano [@Liu2022] pre-trained on
Galaxy Zoo [@Walmsley2022] and fine-tuned on Galaxy10 DECaLS [@Leung2019], with a
ResNet-18 [@He2016] baseline. Active learning adds user labels, weak labels and an
eleventh "not a galaxy" class, and a challenger replaces the champion only if
Galaxy10 accuracy falls by at most 1.5 points, held-out stars and gas are rejected
with at least 60 % recall, and at most 10 % of held-out galaxies are rejected. An
Isolation Forest [@Liu2008] scores deep-field outliers, and a Thompson-sampling
bandit [@Thompson1933] ranks findings for the user. Cross-matches use a Bayesian
likelihood ratio of separation against local source density, with Gaia proper
motions applied. Rover images are classified by a ConvNeXt-Tiny trained on @Lu2020.
The front end uses Three.js and OpenSeadragon, with optional in-browser hand-gesture
control through MediaPipe [@Lugaresi2019].

# Validation

All numbers come from model cards and result files in the repository, and
`paper/make_figures.py` rebuilds every figure from them.

**Galaxy CNN.** On the 2,656-galaxy Galaxy10 validation split the model reaches
88.7 % ten-class, 97.2 % top-2 and 92.7 % coarse accuracy, against 82.8 %, 93.3 % and
89.5 % for ResNet-18; *Disturbed* (57 % recall) and *Unbarred loose spiral* (78 %)
are the hardest classes (\autoref{fig:confusion}). Accuracy on JWST and HST is
expected to be lower because of the domain shift (\autoref{fig:deepfield}). Of three
active-learning rounds, two challengers were correctly rejected and one was promoted
(88.1 % Galaxy10 accuracy, 82 % recall on held-out stars and gas, 4.9 % of real
galaxies rejected).

![Galaxy CNN confusion matrix on the Galaxy10 DECaLS validation split, normalised
by row.\label{fig:confusion}](figures/fig3_confusion.png){ width=70% }

![A public JWST NIRCam F150W image (ESO 350-IG038) processed by the pipeline. Right:
the boxed region with each CNN-classified detection coloured by coarse class; the
many merger/disturbed calls on small, faint sources show the domain shift from the
training images.\label{fig:deepfield}](figures/fig2_deepfield.png)

**Detection on calibrated data.** On a JWST NIRCam F460M mosaic the detector found
12,029 sources with measured spurious rates of 0.24 % and 0.42 % (cold and hot
passes); the low-surface-brightness passes failed the purity test and were dropped
automatically. The 5σ point-source depth is AB 27.3, and 50 % completeness falls
from AB 25.6 for the smallest injected galaxies to AB 21.7 for the largest
(\autoref{fig:completeness}).

![Injection-recovery completeness on a calibrated JWST NIRCam F460M mosaic, by size
class (half-light radius r$_{50}$).\label{fig:completeness}](figures/fig4_completeness.png){ width=72% }

**Cross-matching and Mars.** Using Gaia DR3's Pan-STARRS1 best-neighbour table as an
answer key, the matcher recovered 175 of 177 sources in 23 fields from Galactic
latitude −74° to +76° (precision 98.9 %, recall 100 %). The Mars CNN reaches 80.0 %
accuracy (89.0 % top-3, 61.1 % balanced) on the test split of @Lu2020, which uses
sols unseen in training.

**Operation.** In its first twelve days (30 October to 10 November 2026) the
pipeline ingested 824 images, 1,870 Rubin and ZTF alerts and 46,455 detections,
built 3,857 cross-survey objects from 29,191 archive matches, and raised 744
candidate anomalies.

**Audit of anomaly candidates.** The 744 candidates are not discoveries, and an
audit of them shows why (\autoref{tab:audit}). Only 394 concern individual
astronomical objects; the rest are solar-activity changes or relays of events
announced by other facilities, and only 56 (7.5 %) had been reviewed by a person.
Of the 211 deep-field outliers, 96 are point sources and 26 more are rejected by the
CNN as "not a galaxy", so the Isolation Forest mostly selects the brightest and
largest sources. Of 29 broker tidal-disruption candidates, 23 were first detected
two or more years earlier, which a flare lasting months cannot explain. Half of the
"hostless" transients (8 of 16) are single-detection alerts that the broker classes
as asteroids, and 29 of 57 supernova candidates carry an AGN, QSO or blazar class.

We therefore added vetting rules (`cosmic_watch/vetting.py`, with unit tests) that
the pipeline and the re-measurement share: deep-field "normal" is defined over
extended sources only; tidal-disruption candidates must be young and not identified
as stars or AGN by the archives; a "new, hostless" verdict needs repeat detections, a
successful check against known asteroids, and a recent first detection; and
supernova candidates must not carry an AGN-type broker class. Re-applied to the
same data (\autoref{tab:audit}), the rules retracted 248 of 313 transient and outlier
candidates. In a visual inspection of 24 randomly chosen outliers before and after,
the fraction that were real galaxies rose from 9 of 24 to 14 of 24; the remaining
contaminants are nebulae, satellite trails and detector artefacts. None of the
surviving candidates has yet been confirmed as scientifically unusual. The next
steps are adjudication against spectroscopic classifications from the Transient
Name Server and a measured precision on a larger, independently vetted sample.

| Candidate type | Raised | After vetting | Main reason for retraction |
|---|---:|---:|---|
| Deep-field outliers | 211 | 33 (+51 newly selected) | star (96), CNN "not a galaxy" (28) |
| Tidal-disruption candidates | 29 | 4 | first detected years earlier (23) |
| Supernova candidates | 57 | 28 | broker class AGN/QSO/blazar (29) |
| "New, hostless" transients | 16 | 0 | broker asteroid (8), not new (6) |
| CNN mergers, nuclear transients | 81 | 81 | not yet vetted |
| Solar, relayed and other events | 350 | 350 | not object discovery |
| **Total** | **744** | **496 + 51** | 56 had been reviewed by a person |

: Anomaly candidates raised from 30 October to 10 November 2026, before and after
the vetting rules.\label{tab:audit}

**Limitations.** Classifications are probabilistic and outside the training domain;
preview-image detections are indicative only; Rubin full frames are not public in
real time. `Cosmic Watch` complements, and does not replace, the archives and
brokers it draws on.

# Acknowledgements

This work uses public data and services from NASA, ESA, STScI/MAST, the Vera C.
Rubin Observatory, ZTF and ALeRCE, LIGO, Virgo and KAGRA via GraceDB, NASA GCN,
NASA/SDO and the AIA and HMI teams, SOHO/LASCO, NOAA SWPC, NASA/JPL-Caltech, SDSS,
Pan-STARRS, ESA Gaia, CDS/SIMBAD, NED, the DESI Legacy Imaging Surveys and the
NOIRLab Astro Data Lab, and builds on Zoobot and the Galaxy Zoo volunteers. This
project is dedicated to the author's late mother.

# References
