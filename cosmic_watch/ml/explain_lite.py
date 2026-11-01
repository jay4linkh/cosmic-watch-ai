"""Cheap field-level checks shared by active learning (no per-detection image reads)."""
from ..explain import LINES, _concentration as concentration, _field  # noqa: F401


def field_suspicion(target, filt):
    """0..1: how likely an extended source in this field is NOT a galaxy (nebula fields, emission-line filters)."""
    kind, text = _field(target)
    f = (filt or "").split(";")[-1].strip().upper()
    if kind == "nebula":
        return 1.0, f"this field is {text}"
    if f in LINES:
        return 0.8, f"the filter {f} isolates {LINES[f].split(':')[0]}"
    return 0.0, ""
