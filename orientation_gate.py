"""B0 predicate and explicit C1 diagnostic controls, not a validated social model."""
import numpy as np

MODES = ("B0-original", "proximity-only", "signed-dyad-diagnostic")


def heading_or_none(value):
    """None/zero -> absent; malformed or nonfinite -> ValueError. No renormalization."""
    if value is None:
        return None
    v = np.asarray(value, dtype=float)
    if v.shape != (2,) or not np.isfinite(v).all():
        raise ValueError("heading must be None or a finite planar vector of shape (2,)")
    norm = np.linalg.norm(v)
    if not np.isfinite(norm):
        raise ValueError("heading norm must be finite")
    return None if norm == 0 else v


def original_absolute_predicate(cos_i_to_j, cos_j_to_i, cos_i_j):
    """Exact B0 comparisons; scalar or arrays. Inputs are cosines, not raw vectors."""
    return np.logical_or(np.abs(cos_i_j) > 0.7,
                         np.logical_or(np.abs(cos_i_to_j) > 0.3,
                                       np.abs(cos_j_to_i) > 0.3))


def proximity_only(distance, eps):
    if not np.isfinite([distance, eps]).all() or distance < 0 or eps <= 0:
        raise ValueError("distance must be finite >= 0 and eps finite > 0")
    return bool(distance <= eps)


def signed_direction_dyad(cos_i_to_j, cos_j_to_i):
    """Diagnostic mutual-facing AND; deliberately excludes a parallel-pair rule."""
    return bool(cos_i_to_j > 0.3 and cos_j_to_i > 0.3)


def pair_compatibility(pi, pj, displacement, eps, mode="B0-original", *, close_distance=0.5):
    """Directions entering here are BEV directions; preserve B0 valid-input dots.

    All modes retain distance gate and <.5 close bypass. B0 retains missing
    direction proximity fallback; signed diagnostic rejects missing directions
    outside the bypass. Invalid arrays raise even if the pair would be close.
    """
    if mode not in MODES:
        raise ValueError(f"orientation mode must be one of {MODES}")
    if not np.isfinite(close_distance) or close_distance <= 0:
        raise ValueError('close_distance must be finite and positive')
    pi, pj = heading_or_none(pi), heading_or_none(pj)
    delta = np.asarray(displacement, dtype=float)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("displacement must be a finite planar vector")
    distance = float(np.linalg.norm(delta))
    if not proximity_only(distance, eps):
        return {"compatible": False, "reason": "outside_radius"}
    if mode == "proximity-only":
        return {"compatible": True, "reason": "proximity_only"}
    if distance < close_distance:
        return {"compatible": True, "reason": "close_bypass"}
    if pi is None or pj is None:
        return {"compatible": mode == "B0-original", "reason": "missing_direction"}
    u = delta / distance
    dots = (float(np.dot(pi, u)), float(np.dot(pj, -u)), float(np.dot(pi, pj)))
    connected = (original_absolute_predicate(*dots) if mode == "B0-original"
                 else signed_direction_dyad(dots[0], dots[1]))
    return {"compatible": bool(connected), "reason": mode, "cosines": dots}
