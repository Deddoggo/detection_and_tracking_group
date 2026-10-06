"""Explicit orientation modes. Image pose cues are not verified ground headings."""
import math


def unit(v):
    if v is None or len(v) != 2 or not all(math.isfinite(x) for x in v):
        return None
    n = math.hypot(*v)
    return tuple(x / n for x in v) if n > 1e-6 else None


def compatible(pi, pj, displacement, mode='legacy', confidence_i=None,
               confidence_j=None, minimum_confidence=0.6, facing_cosine=0.3):
    """Proximity and legacy accept missing cues; experimental facing falls back too.

    facing_experimental requires BOTH signed cues toward the other person.
    It assumes correctly signed ground-plane directions and supplied reliability;
    it is an inspection candidate, not validated social grouping.
    """
    if mode not in ('legacy', 'proximity', 'facing_experimental'):
        raise ValueError('unknown orientation mode')
    if mode == 'proximity':
        return True
    pi, pj, v = unit(pi), unit(pj), unit(displacement)
    if pi is None or pj is None or v is None:
        return True
    a = sum(x*y for x, y in zip(pi, v))
    b = -sum(x*y for x, y in zip(pj, v))
    mutual = sum(x*y for x, y in zip(pi, pj))
    if mode == 'legacy':
        return abs(mutual) > 0.7 or abs(a) > 0.3 or abs(b) > 0.3
    if confidence_i is None or confidence_j is None:
        return True
    if min(confidence_i, confidence_j) < minimum_confidence:
        return True
    return a > facing_cosine and b > facing_cosine
