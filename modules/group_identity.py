"""Deterministic one-to-one continuation of numerical spatial-cluster IDs.

This is bookkeeping correctness, not social-group membership inference.
"""


def assign_group_ids(previous, clusters, max_group_id, threshold=0.5):
    """Greedy one-to-one matching by intersection, then old-member retention.

    Split: strongest eligible child retains ID; other children receive new IDs.
    Merge: strongest eligible parent supplies ID; other parents end.
    Ties: lower old ID, then lower cluster label. No missing-frame persistence.
    Returns groups, monotonically increasing counter, and overlap lineage events.
    """
    if not 0 <= threshold <= 1:
        raise ValueError('retention threshold must be in [0,1]')
    previous = {int(k): set(v) for k, v in previous.items() if v}
    clusters = {int(k): list(v) for k, v in clusters.items() if v}
    flat = [p for members in clusters.values() for p in members]
    if len(flat) != len(set(flat)):
        raise ValueError('A person ID must occur in only one current cluster')
    max_group_id = max([max_group_id, *previous.keys()], default=max_group_id)
    candidates, lineage = [], {}
    for label, members in clusters.items():
        lineage[label] = []
        for gid, old in previous.items():
            overlap = len(set(members) & old)
            if overlap:
                lineage[label].append(gid)
                ratio = overlap / len(old)
                if ratio >= threshold:
                    candidates.append((-overlap, -ratio, gid, label))
    assigned, consumed = {}, set()
    for _, _, gid, label in sorted(candidates):
        if label not in assigned and gid not in consumed:
            assigned[label] = gid
            consumed.add(gid)
    groups, events = {}, []
    for label in sorted(clusters):
        if label not in assigned:
            max_group_id += 1
            assigned[label] = max_group_id
        gid = assigned[label]
        groups[gid] = clusters[label]
        parents = sorted(lineage[label])
        events.append({'cluster': label, 'group_id': gid, 'overlap_parents': parents,
                       'continued_from': gid if gid in consumed else None,
                       'split_parents': [p for p in parents if
                           sum(p in v for v in lineage.values()) > 1],
                       'merge': len(parents) > 1})
    return groups, max_group_id, events
