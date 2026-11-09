"""C1 descriptive observations, not social GT or inferred formation events."""
from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral


@dataclass
class SourceTimebase:
    fps: float = None
    timestamps: dict = None
    frame_origin: int = 0
    timestamp_origin: float = 0.0

    def __post_init__(self):
        if self.fps is not None and (not isfinite(self.fps) or self.fps <= 0):
            raise ValueError("supplied source fps must be finite and positive")
        if not isinstance(self.frame_origin, Integral) or self.frame_origin < 0:
            raise ValueError("frame_origin must be a nonnegative integer")
        if not isfinite(self.timestamp_origin):
            raise ValueError("timestamp_origin must be finite")
        if self.timestamps is not None:
            if not self.timestamps or any(not isinstance(f, Integral) or f < 0 for f in self.timestamps):
                raise ValueError("timestamps require nonnegative integer source-frame keys")
            times = [self.timestamps[f] for f in sorted(self.timestamps)]
            if any(not isfinite(t) for t in times) or any(b <= a for a,b in zip(times,times[1:])):
                raise ValueError("source timestamps must be finite and strictly increasing")
        elif self.fps is None or not isfinite(self.fps) or self.fps <= 0:
            raise ValueError("provide verified timestamps or explicit positive source fps")

    def seconds(self, source_frame):
        if not isinstance(source_frame, Integral) or source_frame < self.frame_origin:
            raise ValueError("source frame must be an integer at/after frame_origin")
        if self.timestamps is not None:
            if source_frame not in self.timestamps:
                raise ValueError(f"missing timestamp for source frame {source_frame}; no FPS fallback")
            value = self.timestamps[source_frame] - self.timestamp_origin
        else:
            value = (source_frame-self.frame_origin)/self.fps
        if not isfinite(value) or value < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        return float(value)


def convex_hull_area(points):
    """Untrimmed planar hull area; <3 people undefined, collinear triple zero."""
    pts = list(points)
    if any(len(p)!=2 or not all(isfinite(x) for x in p) for p in pts):
        raise ValueError("area requires finite planar positions")
    if len(pts)<3:
        return None
    pts = sorted(set(tuple(p) for p in pts))
    if len(pts)<3:
        return 0.0
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower,upper=[],[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:
            upper.pop()
        upper.append(p)
    hull=lower[:-1]+upper[:-1]
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(hull,hull[1:]+hull[:1])))/2


def build_report(raw_data,timebase,*,coordinate_units,expected_stride,
                 interval_seconds=30.0,max_gap_seconds=2.0):
    """C1-report-v1. Supports only adjacent equal-membership observed spans.

    Source frames are never re-enumerated as sampled frames. Gaps, membership
    transitions, terminal and single samples receive no imputed time support.
    """
    if coordinate_units not in ('m','pixels','arbitrary'):
        raise ValueError("declare coordinate_units as m, pixels, or arbitrary")
    if not isinstance(expected_stride,Integral) or expected_stride<1:
        raise ValueError("expected_stride must be a positive source-frame step")
    if not all(isfinite(x) and x>0 for x in (interval_seconds,max_gap_seconds)):
        raise ValueError("interval_seconds and max_gap_seconds must be positive and finite")
    frames={}
    for key,people in raw_data.items():
        if not isinstance(key,str) or not key.startswith('frame_') or not key[6:].isdigit():
            raise ValueError("keys must be frame_<nonnegative source index>")
        f=int(key[6:])
        if f in frames or not isinstance(people,list):
            raise ValueError("frames must be unique lists of person observations")
        frames[f]=people
    frames=dict(sorted(frames.items()))
    persons,frame_rows,groups=[],[],[]
    for observation_index,(f,people) in enumerate(frames.items()):
        t=timebase.seconds(f);seen=set();by_group={}
        for p in people:
            pid,gid=p.get('id_p'),p.get('id_g')
            if isinstance(pid,bool) or not isinstance(pid,Integral) or pid<0 or pid in seen:
                raise ValueError("id_p must be a unique nonnegative integer per frame")
            if gid is not None and (isinstance(gid,bool) or not isinstance(gid,Integral) or gid< -1):
                raise ValueError("id_g must be integer >= -1 or None for unknown membership")
            seen.add(pid);pos=p.get('real_pos')
            if pos is not None and (len(pos)!=2 or not all(isfinite(x) for x in pos)):
                raise ValueError("real_pos must be finite planar coordinates or None")
            persons.append({**p,'source_frame':f,'observation_index':observation_index,'elapsed_seconds':t})
            if gid is not None and gid>=0:
                by_group.setdefault(gid,[]).append(p)
        frame_rows.append({'source_frame':f,'elapsed_seconds':t,'person_count':len(people),
            'detected_group_count':len(by_group),'ungrouped_person_count':sum(p.get('id_g')== -1 for p in people),
            'unknown_membership_count':sum(p.get('id_g') is None for p in people)})
        for gid,members in sorted(by_group.items()):
            positions=[p.get('real_pos') for p in members]
            area=convex_hull_area(positions) if all(p is not None for p in positions) else None
            groups.append({'source_frame':f,'elapsed_seconds':t,'group_id':gid,
                'member_ids':sorted(p['id_p'] for p in members),'cardinality':len(members),
                'unknown_attribute_count':sum(str(p.get('gender','Unknown')).capitalize() not in ('Male','Female') for p in members),
                'area':area,'area_status':'defined' if area is not None else 'insufficient_or_missing_geometry',
                'area_units':coordinate_units+'^2','observed_support_seconds':0.0})
    lookup={(g['source_frame'],g['group_id']):g for g in groups}
    frame_order=list(frames);next_frame=dict(zip(frame_order,frame_order[1:]))
    sessions,last=[],{}
    for g in groups:
        f,gid,t=g['source_frame'],g['group_id'],g['elapsed_seconds']
        nxt=next_frame.get(f);future=lookup.get((nxt,gid))
        if future is not None:
            dt=future['elapsed_seconds']-t
            if nxt-f==expected_stride and 0<dt<=max_gap_seconds and future['member_ids']==g['member_ids']:
                g['observed_support_seconds']=dt
        prev=last.get(gid)
        if prev is None or prev['observed_support_seconds']==0:
            session={'group_id':gid,'session_number':1+sum(s['group_id']==gid for s in sessions),
                'first_source_frame':f,'last_source_frame':f,'start_seconds':t,'end_seconds':t,
                'observed_span_seconds':0.0,'member_ids':g['member_ids'],'cardinality':g['cardinality']}
            sessions.append(session)
        else:
            session=next(s for s in reversed(sessions) if s['group_id']==gid)
            session.update(last_source_frame=f,end_seconds=t,observed_span_seconds=t-session['start_seconds'])
        last[gid]=g
    intervals,areas={},{}
    def interval(iv):
        return intervals.setdefault(iv,{'interval_index':iv,'start_seconds':iv*interval_seconds,
            'end_seconds_exclusive':(iv+1)*interval_seconds,'distinct_persistent_group_ids':0,
            'framewise_group_observation_count':0,'group_sessions_starting':0,
            'group_formation_events':None,'formation_event_status':'undefined_without_event_annotations',
            'observed_group_person_seconds':0.0})
    for frame in frame_rows:
        interval(floor(frame['elapsed_seconds']/interval_seconds))
    ids={}
    for g in groups:
        t,support,gid=g['elapsed_seconds'],g['observed_support_seconds'],g['group_id']
        first=floor(t/interval_seconds)
        interval(first)['framewise_group_observation_count']+=1
        ids.setdefault(first,set()).add(gid)
        stop=floor((t+support)/interval_seconds) if support else first
        for iv in range(first,stop+1):
            weight=max(0.,min(t+support,(iv+1)*interval_seconds)-max(t,iv*interval_seconds))
            if iv!=first and weight==0:
                continue
            r=areas.setdefault((iv,gid),{'interval_index':iv,'group_id':gid,'valid_area_support_seconds':0.,
                'area_time_integral':0.,'maximum_observed_area':None,'observed_group_person_seconds':0.})
            r['observed_group_person_seconds']+=weight*g['cardinality']
            interval(iv)['observed_group_person_seconds']+=weight*g['cardinality']
            if g['area'] is not None:
                r['valid_area_support_seconds']+=weight;r['area_time_integral']+=weight*g['area']
                if iv==first:
                    r['maximum_observed_area']=max(g['area'],r['maximum_observed_area'] or 0.)
    for iv,values in ids.items():
        interval(iv)['distinct_persistent_group_ids']=len(values)
    for s in sessions:
        interval(floor(s['start_seconds']/interval_seconds))['group_sessions_starting']+=1
    for row in areas.values():
        denominator=row['valid_area_support_seconds']
        row['temporal_average_area']=row['area_time_integral']/denominator if denominator else None
        row['area_units']=coordinate_units+'^2'
    return {'metadata':{'schema':'C1-report-v1','timebase':'verified_timestamps' if timebase.timestamps is not None else 'explicit_nominal_source_fps',
        'source_fps':timebase.fps,'frame_origin':timebase.frame_origin,'timestamp_origin':timebase.timestamp_origin,
        'expected_source_stride':expected_stride,'max_gap_seconds':max_gap_seconds,'coordinate_units':coordinate_units,
        'interval_seconds':interval_seconds,
        'duration_policy':'adjacent_equal_membership_observed_span; no gap/terminal imputation'},
        'person_observations':persons,'frame_metrics':frame_rows,'group_observations':groups,'group_sessions':sessions,
        'interval_metrics':[intervals[k] for k in sorted(intervals)],'interval_group_metrics':[areas[k] for k in sorted(areas)]}
