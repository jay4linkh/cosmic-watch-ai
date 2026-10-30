"""Cosmic Watch AI: live dashboard.

    streamlit run dashboard/app.py
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cosmic_watch import db  # noqa: E402
from cosmic_watch.config import INTERVALS  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Cosmic Watch AI", page_icon="🔭", layout="wide")

SOURCE_NAMES = {"alerts": "Rubin + ZTF alerts", "sdo": "SDO (Sun)", "mast": "JWST / Hubble"}
BH_INFO = {
    "TDE": "Tidal disruption event: a star torn apart by a supermassive black hole",
    "AGN": "Active galactic nucleus: a feeding supermassive black hole",
    "QSO": "Quasar: an extremely luminous, distant AGN",
    "Blazar": "Blazar: an AGN with its jet pointed at Earth",
}


# ---------- helpers ----------

def resolve(p):
    if not p:
        return None
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def mjd_to_dt(mjd: float) -> datetime:
    return datetime.fromtimestamp((mjd - 40587.0) * 86400.0, tz=timezone.utc)


def ago(dt: datetime) -> str:
    s = (datetime.now(timezone.utc) - dt).total_seconds()
    if s < 0:
        return "just now"
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if s >= size:
            return f"{s / size:.0f} {unit} ago"
    return f"{s:.0f} s ago"


def parse_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def broker_url(survey: str, oid: str) -> str:
    return f"https://alerce.online/object/{oid}" if survey == "ztf" else f"https://lsst.fink-portal.org/{oid}"


@st.cache_data(show_spinner=False, max_entries=500)
def thumb(path: str, size: int = 480):
    im = Image.open(path)
    im.thumbnail((size, size))
    return im.convert("RGB")


def query(sql, params=()):
    con = db.connect()
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def launch(source: str):
    """Start a one-off pipeline run in the background so the UI stays responsive."""
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen([sys.executable, "-m", "cosmic_watch.pipeline", "--once", "--only", source],
                     cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)


# ---------- sidebar ----------

with st.sidebar:
    st.header("🔭 Cosmic Watch AI")
    st.caption("Live observatory data + computer vision")
    auto = st.toggle("Auto-refresh every 60 s", value=True)
    st.divider()
    st.subheader("Fetch now")
    st.caption("Runs one pipeline pass in the background. For continuous updates, "
               "run `python -m cosmic_watch.pipeline` in a terminal.")
    for key, name in SOURCE_NAMES.items():
        if st.button(f"↻ {name}", key=f"fetch_{key}", width="stretch"):
            launch(key)
            st.toast(f"Fetching {name}… results appear on the next refresh.")
    st.divider()
    st.caption("Data: [ALeRCE](https://alerce.online) (Rubin/LSST, ZTF) · "
               "[MAST](https://mast.stsci.edu) (JWST, Hubble) · [NASA SDO](https://sdo.gsfc.nasa.gov)")


# ---------- sections ----------

def status_row():
    runs = query("""SELECT r.* FROM runs r JOIN (SELECT source, MAX(id) id FROM runs GROUP BY source) m ON r.id=m.id""")
    ok = query("""SELECT source, MAX(finished_at) finished_at FROM runs WHERE status='ok' GROUP BY source""")
    ok = dict(zip(ok.source, ok.finished_at))
    cols = st.columns(3)
    for col, (key, name) in zip(cols, SOURCE_NAMES.items()):
        r = runs[runs.source == key]
        last_ok = parse_utc(ok.get(key))
        with col.container(border=True):
            if r.empty:
                st.markdown(f"**{name}**  \n⚪ never fetched")
                continue
            r = r.iloc[0]
            running = pd.isna(r.finished_at)
            icon = "🟡 fetching…" if running else ("🟢 live" if r.status == "ok" else "🔴 last run failed")
            st.markdown(f"**{name}** · {icon}")
            every = INTERVALS[key] // 60
            st.caption(f"Last success: {ago(last_ok) if last_ok else 'n/a'} · polls every {every} min")
            if r.status == "error" and not running:
                st.caption(f"⚠️ {str(r.message).splitlines()[0][:140]}")


def metrics_row():
    img = query("SELECT source, stats FROM images")
    gal = sum(json.loads(s).get("n_galaxies", 0) for s in img[img.source.isin(["jwst", "hst"])].stats)
    n_deep = int(img.source.isin(["jwst", "hst"]).sum())
    al = query("SELECT survey, bh_candidate, lastmjd FROM alerts")
    now_mjd = time.time() / 86400.0 + 40587.0
    rubin_24h = int(((al.survey == "lsst") & (al.lastmjd > now_mjd - 1)).sum())
    c = st.columns(4)
    c[0].metric("Deep-space images analysed", n_deep)
    c[1].metric("Galaxies detected by AI", f"{gal:,}")
    c[2].metric("Black-hole activity candidates", int(al.bh_candidate.sum()))
    c[3].metric("Rubin objects (last 24 h)", rubin_24h)


def tab_deep_space():
    df = query("""SELECT * FROM images WHERE source IN ('jwst','hst') ORDER BY obs_time DESC, id DESC""")
    if df.empty:
        st.info("No JWST/Hubble images yet. Click **↻ JWST / Hubble** in the sidebar.")
        return
    df["label"] = df.apply(lambda r: f"{r.source.upper()} · {r.target} · {r.instrument} {r['filter']} · released {r.obs_time[:10]}", axis=1)
    n_gal = df.stats.map(lambda s: json.loads(s or "{}").get("n_galaxies", 0))
    choice = st.selectbox("Newly public image (newest first; opens on the richest field)",
                          df.label.tolist(), index=int(n_gal.values.argmax()))
    r = df[df.label == choice].iloc[0]
    stats = json.loads(r.stats or "{}")
    left, right = st.columns([3, 1])
    with right:
        st.markdown(f"### {r.target}")
        st.caption(f"{r.source.upper()} · {r.instrument} · {r['filter']}  \nPublic since {r.obs_time} UTC  \n`{r.external_id}`")
        st.metric("Sources found", stats.get("n_sources", 0))
        st.metric("Galaxies", stats.get("n_galaxies", 0))
        st.metric("Stars", stats.get("n_stars", 0))
        counts = stats.get("counts", {})
        if counts:
            st.bar_chart(pd.Series(counts, name="count"), horizontal=True, height=180)
        show_ai = st.toggle("Show AI detections", value=True)
        st.caption("🟡 star · 🔵 galaxy · 🟣 elongated galaxy · 🟢 large galaxy")
    with left:
        path = resolve(r.annotated_path if show_ai else r.path)
        if path and path.exists():
            st.image(str(path), width="stretch")
    with st.expander("All analysed deep-space images"):
        cols = st.columns(4)
        for i, row in enumerate(df.itertuples()):
            p = resolve(row.annotated_path)
            if p and p.exists():
                s = json.loads(row.stats or "{}")
                cols[i % 4].image(thumb(str(p), 360), caption=f"{row.target}: {s.get('n_galaxies', 0)} galaxies")


def alert_cards(df, n_cols=3):
    cols = st.columns(n_cols)
    for i, a in enumerate(df.itertuples()):
        with cols[i % n_cols].container(border=True):
            p = resolve(a.stamp_path)
            if p and p.exists():
                st.image(str(p), width="stretch")
            seen = mjd_to_dt(a.lastmjd)
            st.markdown(f"**{a.cls}** · p = {a.probability:.2f} · [{a.oid}]({broker_url(a.survey, a.oid)})")
            snr = f" · AI diff S/N {a.diff_snr:.0f}" if pd.notna(a.diff_snr) else ""
            st.caption(f"{a.survey.upper()} · last seen {ago(seen)} · RA {a.ra:.4f} Dec {a.dec:+.4f}{snr}")


def tab_black_holes():
    st.markdown("No telescope sees a black hole directly. We watch for what they **do** to their surroundings:")
    st.markdown("\n".join(f"- **{k}**: {v.split(': ', 1)[1]}" for k, v in BH_INFO.items()))
    df = query("SELECT * FROM alerts WHERE bh_candidate=1 ORDER BY lastmjd DESC")
    if df.empty:
        st.info("No candidates yet. Click **↻ Rubin + ZTF alerts** in the sidebar.")
        return
    c1, c2, c3 = st.columns(3)
    classes = c1.multiselect("Class", sorted(df.cls.unique()), default=sorted(df.cls.unique()))
    surveys = c2.multiselect("Survey", sorted(df.survey.unique()), default=sorted(df.survey.unique()))
    only_stamps = c3.toggle("Only with images", value=True)
    f = df[df.cls.isin(classes) & df.survey.isin(surveys)]
    if only_stamps:
        f = f[f.stamp_path.notna()]
    st.caption(f"{len(f)} candidates · TDEs first")
    f = f.assign(_o=(f.cls != "TDE")).sort_values(["_o", "lastmjd"], ascending=[True, False])
    alert_cards(f.head(30))


def tab_rubin():
    df = query("SELECT * FROM alerts WHERE survey='lsst' ORDER BY lastmjd DESC")
    if df.empty:
        st.info("No Rubin alerts yet. Click **↻ Rubin + ZTF alerts** in the sidebar.")
        return
    newest = mjd_to_dt(df.lastmjd.max())
    st.markdown(f"Newest Rubin detection: **{newest:%Y-%m-%d %H:%M} UTC** ({ago(newest)}). "
                "Classes from ALeRCE's Rubin stamp classifier; *AI diff S/N* is our score of how "
                "strong and centred the new light is in the difference image.")
    left, right = st.columns([1, 2])
    with left:
        st.bar_chart(df.cls.value_counts(), horizontal=True, height=240)
    with right:
        show = df[["oid", "cls", "probability", "n_det", "ra", "dec", "diff_snr"]].head(200)
        st.dataframe(show, hide_index=True, height=240, width="stretch")
    pick = st.multiselect("Show stamps for", sorted(df.cls.unique()),
                          default=[c for c in ("SN", "AGN", "VS", "bogus") if c in set(df.cls)])
    alert_cards(df[df.cls.isin(pick) & df.stamp_path.notna()].head(24), n_cols=4)


def tab_sun():
    df = query("""SELECT i.* FROM images i JOIN (SELECT filter, MAX(obs_time) t FROM images WHERE source='sdo' GROUP BY filter) m
                  ON i.filter=m.filter AND i.obs_time=m.t WHERE i.source='sdo' ORDER BY i.filter""")
    if df.empty:
        st.info("No SDO images yet. Click **↻ SDO (Sun)** in the sidebar.")
        return
    show_ai = st.toggle("Show AI detections", value=True, key="sun_ai")
    cols = st.columns(len(df))
    for col, r in zip(cols, df.itertuples()):
        stats = json.loads(r.stats or "{}")
        with col:
            p = resolve(r.annotated_path if show_ai else r.path)
            if p and p.exists():
                st.image(str(p), width="stretch")
            st.markdown(f"**{r.title}**")
            obs = parse_utc(r.obs_time)
            st.caption(f"{r.obs_time} UTC ({ago(obs) if obs else ''})")
            if "n_sunspots" in stats:
                st.metric("Sunspots", stats["n_sunspots"])
            else:
                st.metric("Active regions", stats.get("n_active_regions", 0),
                          help="Bright magnetically active regions on the disk")
                st.caption(f"Activity index {stats.get('activity_index', 0)} % of disk")


def tab_about():
    st.markdown("""
### How it works
1. **Fetchers** poll the observatories: Rubin and ZTF alerts through the ALeRCE broker (every 10 min),
   NASA SDO full-disk Sun images (every 15 min), and newly public JWST/Hubble images from MAST (every 6 h).
2. **Detectors** run computer vision on everything new:
   - *Deep space*: background modelling + source extraction (SEP), then star/galaxy classification from
     each source's size and shape relative to the image's PSF. Bright-star spikes and edges are suppressed.
   - *Alerts*: science / template / difference cutouts are rendered, and the difference image is scored
     (central signal-to-noise). Classes come from the broker's classifiers.
   - *Sun*: solar-disk fit, limb-darkening correction and thresholding for sunspots (visible light) and
     active regions (extreme UV).
3. Results go into SQLite and this dashboard refreshes automatically.

### Honest limits (v0)
- The galaxy classifier is rule-based. Small faint galaxies can look like stars. Next step: a trained model (Zoobot / YOLO).
- Rubin's full camera frames are not public in real time; only the alert cutouts are.
- ZTF stamps are from each object's first alert.
""")


# ---------- layout ----------

def body():
    st.title("🔭 Cosmic Watch AI")
    st.caption("Images from Rubin, ZTF, JWST, Hubble and SDO, fetched live and analysed by computer vision.")
    status_row()
    metrics_row()
    t1, t2, t3, t4, t5 = st.tabs(["🌌 Deep Space (JWST/Hubble)", "🕳️ Black Hole Watch", "📡 Rubin Live",
                                  "☀️ Sun (SDO)", "ℹ️ How it works"])
    with t1:
        tab_deep_space()
    with t2:
        tab_black_holes()
    with t3:
        tab_rubin()
    with t4:
        tab_sun()
    with t5:
        tab_about()


if auto:
    st.fragment(run_every="60s")(body)()
else:
    body()
