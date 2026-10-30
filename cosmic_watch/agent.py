"""The Cosmic Watch agent: watches the anomaly feed and decides what to show you.

It learns what *you* find interesting with a Thompson-sampling bandit (a simple,
well-founded form of reinforcement learning):

  * arms    = anomaly kinds (sun brightening, deep-field outlier, TDE candidate, ...)
  * reward  = you open it ("Look") or classify it  -> +1 ;  you dismiss it -> 0
  * policy  = for each unseen anomaly, sample interest ~ Beta(alpha, beta) of its kind,
              multiply by the detector's anomaly score, suggest the best one

So it explores kinds it is unsure about and exploits the ones you keep opening.
"""
import json
import random

from . import db

KIND_NAMES = {
    "sun_brightening": "sudden brightening on the Sun",
    "sun_activity_change": "a change in solar activity",
    "deep_outlier": "an unusual object in a deep field",
    "tde_candidate": "a black hole shredding a star",
    "classifier_disagreement": "a case where the AI and the classifier disagree",
    "cnn_merger": "a possible galaxy collision (CNN)",
}
OPENERS = ["This looks interesting.", "Something caught my eye.", "Have a look at this.", "I found an odd one."]


def _prefs(con):
    rows = con.execute("SELECT kind, alpha, beta FROM agent_prefs").fetchall()
    return {r["kind"]: (r["alpha"], r["beta"]) for r in rows}


def suggest(con, exclude: set[int] | None = None) -> dict | None:
    """Pick the next anomaly to show (or None)."""
    exclude = exclude or set()
    rows = con.execute("""SELECT * FROM anomalies WHERE status='new' ORDER BY id DESC LIMIT 200""").fetchall()
    rows = [r for r in rows if r["id"] not in exclude]
    if not rows:
        return None
    prefs = _prefs(con)
    best, best_v, best_interest = None, -1.0, 0.5
    for r in rows:
        a, b = prefs.get(r["kind"], (1.0, 1.0))
        interest = random.betavariate(a, b)
        v = interest * (0.35 + 0.65 * r["score"])
        if v > best_v:
            best, best_v, best_interest = r, v, interest
    a, b = prefs.get(best["kind"], (1.0, 1.0))
    hyp = json.loads(best["hypotheses"] or "[]")
    could = ", ".join(hyp[:-1]) + (f" or {hyp[-1]}" if len(hyp) > 1 else (hyp[0] if hyp else ""))
    learned = a + b - 2
    why_me = (f"You've opened {int(a - 1)} of {int(learned)} {KIND_NAMES.get(best['kind'], best['kind'])} suggestions, so I rank these "
              f"{'higher' if a > b else 'lower'}." if learned >= 2 else "I'm still learning what you like.")
    return {
        "id": best["id"], "kind": best["kind"], "source": best["source"], "score": best["score"],
        "title": best["title"], "reason": best["reason"], "hypotheses": hyp,
        "message": f"{random.choice(OPENERS)} {best['reason']} It could be {could}. Want to look?",
        "why_me": why_me, "interest": round(a / (a + b), 2),
        "ref": json.loads(best["ref"] or "{}"),
        "pending": len(rows),
    }


def feedback(con, anomaly_id: int, action: str) -> dict:
    """action: look | label (reward 1) or dismiss (reward 0)."""
    row = con.execute("SELECT kind FROM anomalies WHERE id=?", (anomaly_id,)).fetchone()
    if not row:
        return {"ok": False}
    reward = 1 if action in ("look", "label") else 0
    con.execute("""INSERT INTO agent_prefs (kind, alpha, beta) VALUES (?, ?, ?)
                   ON CONFLICT (kind) DO UPDATE SET alpha = alpha + ?, beta = beta + ?""",
                (row["kind"], 1 + reward, 2 - reward, reward, 1 - reward))
    con.execute("UPDATE anomalies SET status=?, feedback_at=? WHERE id=?",
                ("looked" if reward else "dismissed", db.utcnow(), anomaly_id))
    con.commit()
    return {"ok": True, "prefs": {k: round(a / (a + b), 2) for k, (a, b) in _prefs(con).items()}}


def stats(con) -> dict:
    prefs = _prefs(con)
    counts = {r["kind"]: r["n"] for r in con.execute(
        "SELECT kind, COUNT(*) n FROM anomalies GROUP BY kind").fetchall()}
    return {
        "anomalies": counts,
        "new": con.execute("SELECT COUNT(*) FROM anomalies WHERE status='new'").fetchone()[0],
        "interest": {k: {"name": KIND_NAMES.get(k, k), "p": round(a / (a + b), 2), "seen": int(a + b - 2)}
                     for k, (a, b) in prefs.items()},
    }
