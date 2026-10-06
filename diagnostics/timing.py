"""Use decoded display-order timestamps for VFR pilot ranges."""
import math


def select_interval(timestamps, start_seconds, end_seconds):
    if not 0 <= start_seconds < end_seconds:
        raise ValueError('invalid time interval')
    times = [f['timestamp_seconds'] for f in timestamps['frames']]
    if any(f['frame'] != n for n, f in enumerate(timestamps['frames'])):
        raise ValueError('timestamps must cover every zero-based decoded frame')
    if any(not math.isfinite(t) for t in times) or any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('timestamps must be finite and strictly increasing')
    start = next((n for n,t in enumerate(times) if t >= start_seconds), None)
    end = next((n for n,t in enumerate(times) if t >= end_seconds), None)
    if start is None or end is None or start >= end:
        raise ValueError('interval not covered by timestamp table')
    return start, end, times[start], times[end]
