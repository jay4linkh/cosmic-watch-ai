# arXiv submission — copy/paste sheet

Upload file: `paper/cosmic-watch-arxiv.zip` (TeX source, .bbl, 4 PNG figures; compiles with pdflatex)

- **Primary category:** astro-ph.IM (Instrumentation and Methods for Astrophysics)
- **Cross-list (optional):** cs.CV
- **License:** CC BY 4.0 (recommended) or arXiv's default non-exclusive license

**Title**

Cosmic Watch: a live, explorable observatory that applies computer vision and machine learning to public astronomy data streams

**Authors**

Jay Gupta (SVKM's NMIMS Mukesh Patel School of Technology Management and Engineering, Mumbai, India)

**Comments**

11 pages, 4 figures, 1 table. Code: https://github.com/jay4linkh/cosmic-watch-ai (ASCL submission 4908)

**Abstract** (ASCII, under arXiv's 1,920-character limit)

Public astronomy data now arrive within minutes to days of being taken, but they are spread across many archives, formats and interfaces. We present Cosmic Watch, an open-source Python and WebGL system that continuously collects newly released data from thirteen public feeds (Rubin and ZTF alerts, JWST and Hubble images, solar, space-weather, gravitational-wave, Mars rover and near-Earth-object data), analyses it with computer vision and neural networks, cross-matches the results against archival catalogues, and presents them in one interactive 3D environment. On calibrated JWST imaging the source detector controls purity with sign-flipped images and measures completeness by injecting synthetic galaxies: on a NIRCam F460M mosaic it recovered 12,029 sources with spurious rates of 0.24% and 0.42% and a 5-sigma point-source depth of AB 27.3. A galaxy morphology classifier fine-tuned from Zoobot reaches 88.7% ten-class accuracy on Galaxy10 DECaLS, and the catalogue matcher recovers 175 of 177 reference sources across 23 fields. We also audit the 744 anomaly candidates raised in the first twelve days of operation, show that most were stars, long-lived active nuclei, asteroids or relayed events, and describe vetting rules that retract 248 of 313 transient and outlier candidates. No discoveries are claimed. The software runs on a laptop, needs no API keys, and is released under the MIT licence.

## Steps

1. Log in at https://arxiv.org/user/ (register first if needed; an institutional email speeds up endorsement).
2. Start a new submission, choose astro-ph.IM. If arXiv asks for an endorsement, it emails you a code: send it to one astronomer who has posted to astro-ph (for example a professor) and ask them to endorse you.
3. Upload `cosmic-watch-arxiv.zip`, let arXiv compile it, and check the PDF it shows.
4. Paste the title, author, abstract and comments above.
5. Preview and submit. Papers announce the next weekday evening (US Eastern) if submitted before the 14:00 ET cutoff.
6. When you have the arXiv ID, add it to the ASCL entry (record 4908) Ref List and email the ASCL editor.
