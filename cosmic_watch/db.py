import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA, DB_PATH, ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,            -- sdo | jwst | hst
    external_id TEXT NOT NULL,
    title TEXT,
    target TEXT,
    instrument TEXT,
    filter TEXT,
    obs_time TEXT,                   -- when the data was taken / released (UTC ISO)
    fetched_at TEXT NOT NULL,
    path TEXT NOT NULL,
    annotated_path TEXT,
    stats TEXT,                      -- JSON summary from the detector
    UNIQUE (source, external_id)
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    x REAL, y REAL, a REAL, b REAL, theta REAL,
    score REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    survey TEXT NOT NULL,            -- lsst | ztf
    oid TEXT NOT NULL,
    cls TEXT NOT NULL,
    classifier TEXT,
    probability REAL,
    lastmjd REAL,
    ra REAL, dec REAL,
    n_det INTEGER,
    bh_candidate INTEGER DEFAULT 0,  -- TDE / AGN / QSO / Blazar
    diff_snr REAL,                   -- our CV score on the difference stamp
    stamp_path TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE (survey, oid, cls)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT,                     -- ok | error
    n_new INTEGER DEFAULT 0,
    message TEXT
);
CREATE INDEX IF NOT EXISTS idx_det_image ON detections(image_id);
CREATE INDEX IF NOT EXISTS idx_alerts_mjd ON alerts(lastmjd);
"""


def rel(path) -> str | None:
    """Store file paths relative to the project root so the data folder is portable."""
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def connect() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")   # dashboard reads while pipeline writes
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def image_exists(con, source: str, external_id: str) -> bool:
    row = con.execute("SELECT 1 FROM images WHERE source=? AND external_id=?",
                      (source, external_id)).fetchone()
    return row is not None


def insert_image(con, rec: dict, annotated_path: str | None, stats: dict, detections: list[dict]) -> int:
    cur = con.execute(
        """INSERT OR IGNORE INTO images
           (source, external_id, title, target, instrument, filter, obs_time, fetched_at, path, annotated_path, stats)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (rec["source"], rec["external_id"], rec.get("title"), rec.get("target"), rec.get("instrument"),
         rec.get("filter"), rec.get("obs_time"), utcnow(), rel(rec["path"]), rel(annotated_path), json.dumps(stats)),
    )
    image_id = cur.lastrowid
    if cur.rowcount and detections:
        con.executemany(
            "INSERT INTO detections (image_id, label, x, y, a, b, theta, score) VALUES (?,?,?,?,?,?,?,?)",
            [(image_id, d["label"], d.get("x"), d.get("y"), d.get("a"), d.get("b"), d.get("theta"), d.get("score"))
             for d in detections],
        )
    con.commit()
    return image_id


def upsert_alert(con, a: dict) -> bool:
    """Insert or refresh an alert row. Returns True if it is new."""
    new = con.execute("SELECT 1 FROM alerts WHERE survey=? AND oid=? AND cls=?",
                      (a["survey"], a["oid"], a["cls"])).fetchone() is None
    con.execute(
        """INSERT INTO alerts (survey, oid, cls, classifier, probability, lastmjd, ra, dec, n_det,
                               bh_candidate, diff_snr, stamp_path, fetched_at)
           VALUES (:survey,:oid,:cls,:classifier,:probability,:lastmjd,:ra,:dec,:n_det,
                   :bh_candidate,:diff_snr,:stamp_path,:fetched_at)
           ON CONFLICT (survey, oid, cls) DO UPDATE SET
               probability=excluded.probability, lastmjd=excluded.lastmjd, n_det=excluded.n_det,
               fetched_at=excluded.fetched_at,
               diff_snr=COALESCE(excluded.diff_snr, alerts.diff_snr),
               stamp_path=COALESCE(excluded.stamp_path, alerts.stamp_path)""",
        {"diff_snr": None, "stamp_path": None, "n_det": None, "ra": None, "dec": None,
         "classifier": None, **a, "fetched_at": utcnow()},
    )
    con.commit()
    return new


def alert_has_stamp(con, survey: str, oid: str) -> bool:
    row = con.execute("SELECT 1 FROM alerts WHERE survey=? AND oid=? AND stamp_path IS NOT NULL",
                      (survey, oid)).fetchone()
    return row is not None


def start_run(con, source: str) -> int:
    # A run still "open" after 2 h was killed (Ctrl+C, crash); close it so the dashboard doesn't show it as running.
    con.execute("""UPDATE runs SET finished_at=?, status='error', message='interrupted'
                   WHERE finished_at IS NULL AND started_at < datetime('now', '-2 hours')""", (utcnow(),))
    cur = con.execute("INSERT INTO runs (source, started_at) VALUES (?, ?)", (source, utcnow()))
    con.commit()
    return cur.lastrowid


def finish_run(con, run_id: int, status: str, n_new: int = 0, message: str = "") -> None:
    con.execute("UPDATE runs SET finished_at=?, status=?, n_new=?, message=? WHERE id=?",
                (utcnow(), status, n_new, message[:500], run_id))
    con.commit()
