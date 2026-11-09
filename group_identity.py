"""C1 bookkeeping only: deterministic one-to-one old-coverage association.

The caller owns (active_groups, max_group_id) per video/run. Empty observations
clear active state, not allocator history, and do not assert social dissolution.
"""
from numbers import Integral

VERSION = "C1-corrected-bookkeeping-v1"


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _groups(groups, name):
    if not hasattr(groups, "items"):
        raise ValueError(f"{name} must map integer labels to member IDs")
    clean, seen = {}, set()
    for label, members in groups.items():
        label = _integer(label, f"{name} label")
        try:
            values = [_integer(m, "person ID") for m in members]
        except TypeError as exc:
            raise ValueError("group members must be an iterable of person IDs") from exc
        if len(values) != len(set(values)) or seen.intersection(values):
            raise ValueError(f"{name} must be disjoint groups with unique members")
        seen.update(values)
        if values:
            clean[label] = tuple(sorted(values))
    return clean


def associate_groups(previous, current, max_group_id, threshold_overlap=0.5,
                     matching_policy="old_coverage"):
    """Return (raw_label_to_persistent_id, active_groups, monotonic_max).

    Eligible: positive intersection AND |current & old| / |old| >= threshold.
    Globally ordered greedy edges: intersection descending, old coverage
    descending, old ID ascending, sorted current membership lexicographic.
    Raw labels and dictionary insertion order never break a tie. This is a
    deterministic greedy objective, not a maximum-total-overlap optimizer.
    """
    if matching_policy != "old_coverage":
        raise ValueError("only explicit matching_policy='old_coverage' is supported")
    if not 0 <= threshold_overlap <= 1:
        raise ValueError("threshold_overlap must be finite and in [0, 1]")
    maximum = _integer(max_group_id, "max_group_id", -1)
    old, new = _groups(previous, "previous"), _groups(current, "current")
    maximum = max([maximum, *old.keys()])
    candidates = []
    for label, members in new.items():
        for gid, old_members in old.items():
            overlap = len(set(members).intersection(old_members))
            coverage = overlap / len(old_members)
            if overlap and coverage >= threshold_overlap:
                candidates.append((-overlap, -coverage, gid, members, label))
    mapping, claimed = {}, set()
    for _, _, gid, _, label in sorted(candidates):
        if gid not in claimed and label not in mapping:
            mapping[label] = gid
            claimed.add(gid)
    for label in sorted(new, key=lambda k: new[k]):
        if label not in mapping:
            maximum += 1
            mapping[label] = maximum
    active = {mapping[label]: list(new[label]) for label in sorted(new, key=lambda k: new[k])}
    return mapping, active, maximum
