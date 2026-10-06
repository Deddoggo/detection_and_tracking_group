"""Import an explicitly reviewed prefix of a full-video CVAT rectangle export.

No model runs, no interpolation guesses, and no modification of the source export.
Native IDs are namespaced to this exact export; identity is not inferred.
"""
import argparse
import collections
import copy
import csv
import math
from pathlib import Path

from .core import digest, iou, read, validate, write
from .cvat_annotations import boolean, load_xml


def import_range(root, timing, end, export_hash, review=None):
    task = root.find('meta/task')
    count = len(timing['frames'])
    if task is None or int(task.findtext('size', '-1')) != count:
        raise ValueError('Full-video task and timestamp frame counts differ')
    if int(task.findtext('start_frame', '-1')) != 0 or int(task.findtext('stop_frame', '-1')) != count-1:
        raise ValueError('Expected full source video task starting at zero')
    if task.findtext('frame_filter', '').strip() not in ('', 'step=1'):
        raise ValueError('Frame skipping is not supported')
    if boolean(task.findtext('flipped', 'false')):
        raise ValueError('Flipped task is unsupported')
    w, h = [int(task.findtext('original_size/'+k)) for k in ('width', 'height')]
    if not 0 < end < count:
        raise ValueError('Reviewed prefix must have a timestamp end boundary')
    times = timing['frames']
    if any(x['frame'] != n for n, x in enumerate(times)):
        raise ValueError('Timestamp indices must be contiguous source indices')
    if any(b['timestamp_seconds'] <= a['timestamp_seconds'] for a,b in zip(times,times[1:])):
        raise ValueError('Non-monotonic source timestamps')
    frames = [{'frame':n, 'timestamp_seconds':times[n]['timestamp_seconds'],
               'coverage':'pending', 'items':[]} for n in range(end)]
    tracks = root.findall('track'); ids = []; summaries = []; outside_count = 0
    for track in tracks:
        raw = int(track.attrib['id'])
        if raw < 0 or raw in ids: raise ValueError('Invalid or duplicate native track ID')
        ids.append(raw)
        if track.get('label') != 'person' or track.get('source', 'manual') != 'manual':
            raise ValueError('Only manual person rectangle tracks accepted')
        if any(b.tag != 'box' for b in track): raise ValueError('Non-rectangle track')
        rows = track.findall('box'); numbers = [int(b.attrib['frame']) for b in rows]
        if not rows or numbers != sorted(set(numbers)) or numbers[0] < 0 or numbers[-1] >= count:
            raise ValueError('Invalid track frame order/range')
        # This export is dense. Do not approximate omitted visible frame labels.
        if any(right != left+1 and not boolean(a.get('outside','0'))
               for a,left,right in zip(rows,numbers,numbers[1:])):
            raise ValueError('Sparse export: request CVAT dense export before importing')
        if numbers[-1] < end-1 and not boolean(rows[-1].get('outside','0')):
            raise ValueError('Visible track ends before reviewed boundary without Outside')
        seen_visible = []; outside_frames = []; reentries = []; previous_outside = False
        for b, n in zip(rows, numbers):
            if n >= end: break
            outside = boolean(b.get('outside','0'))
            if outside:
                outside_count += 1; outside_frames.append(n)
            else:
                coords = [float(b.attrib[k]) for k in ('xtl','ytl','xbr','ybr')]
                x1,y1,x2,y2 = coords
                if not all(math.isfinite(x) for x in coords) or not 0<=x1<x2<=w or not 0<=y1<y2<=h:
                    raise ValueError(f'Invalid bbox: track {raw}, frame {n}')
                if float(b.get('rotation','0')) != 0: raise ValueError('Rotated box')
                attrs = {a.attrib['name']:(a.text or '').strip() for a in b.findall('attribute')}
                uncertain = boolean(attrs.get('identity_uncertain','false'))
                frames[n]['items'].append({'bbox':[x1,y1,x2-x1,y2-y1],
                    'person_id':None if uncertain else raw+1, 'candidate_person_id':raw+1,
                    'annotation_track_id':str(raw), 'identity_uncertain':uncertain,
                    'occluded':boolean(b.get('occluded','0')), 'ignore':False,
                    'keyframe':boolean(b.get('keyframe','1')),
                    'origin':'CVAT exported per-frame manual/interpolated rectangle'})
                seen_visible.append(n)
                if previous_outside: reentries.append(n)
            previous_outside = outside
        summaries.append({'native_track_id':raw, 'person_id':raw+1,
                          'visible_frames_in_range':len(seen_visible),
                          'first_visible':min(seen_visible) if seen_visible else None,
                          'last_visible':max(seen_visible) if seen_visible else None,
                          'outside_frames':outside_frames,'reappearance_frames':reentries,
                          'last_export_frame':numbers[-1]})
    overlaps = []
    for f in frames:
        items = f['items']
        for j, a in enumerate(items):
            for b in items[j+1:]:
                overlap = iou(a['bbox'], b['bbox'])
                if overlap >= .8:
                    overlaps.append({'frame':f['frame'],'native_ids':[a['annotation_track_id'],b['annotation_track_id']], 'iou':overlap})
    required = ('source_alignment_checked','all_frames_reviewed','all_visible_people_labeled','ids_reviewed')
    confirmed = bool(review and review.get('export_sha256') == export_hash and
                     review.get('end_frame_exclusive') == end and review.get('reviewer') and
                     all(review.get(k) is True for k in required))
    if review and not confirmed: raise ValueError('Review confirmation incomplete or not bound to this export/range')
    if confirmed:
        for f in frames: f['coverage']='full'
    source = {'video':timing['video'],'video_sha256':timing['video_sha256'],
              'fps':timing['average_fps'],'width':w,'height':h,'total_frame_count':count,
              'start_frame':0,'end_frame_exclusive':end,
              'duration_seconds':times[end]['timestamp_seconds']-times[0]['timestamp_seconds'],
              'frame_timestamps_seconds':[x['timestamp_seconds'] for x in times[:end]],
              'timestamp_basis':'decoded source PTS','scene':None,'session':task.findtext('name')}
    doc = {'schema_version':1,'kind':'annotations','source':source,'frames':frames,
           'annotation_status':'human_reviewed' if confirmed else 'pending',
           'review':review,'conversion':{'export_sha256':export_hash,
             'native_track_id_to_person_id':{str(n):n+1 for n in ids},
             'identity_namespace':'exact CVAT export hash + native track ID; no cross-export continuity assumed',
             'bbox_convention':'as exported; visible-versus-amodal convention not encoded in XML',
             'range_policy':'explicit prefix only; labels beyond boundary discarded',
             'source_alignment':'user confirmation required; matching name/count/resolution alone is not proof'}}
    validate(doc)
    audit = {'task_id':task.findtext('id'),'task_frames':count,'task_name':task.findtext('name'),
             'reviewed_candidate_range':[0,end-1],'mot_range':[1,end],
             'first_timestamp':times[0]['timestamp_seconds'],'last_timestamp':times[end-1]['timestamp_seconds'],
             'end_boundary_timestamp':times[end]['timestamp_seconds'],'export_sha256':export_hash,
             'tracks_in_export':len(ids),'tracks_visible_in_range':sum(bool(s['visible_frames_in_range']) for s in summaries),
             'visible_boxes':sum(len(f['items']) for f in frames),'outside_records_excluded':outside_count,
             'uncertain_identity_boxes':sum(i['identity_uncertain'] for f in frames for i in f['items']),
             'people_per_frame_min':min(len(f['items']) for f in frames),
             'people_per_frame_max':max(len(f['items']) for f in frames),
             'high_overlap_pairs_iou_at_least_0_8':overlaps,'tracks':summaries,
             'annotation_status':doc['annotation_status'],
             'identity_and_completeness_proven_by_script':False,'predictions_available':False}
    return doc, audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',required=True);p.add_argument('--timestamps',required=True)
    p.add_argument('--end-frame-exclusive',type=int,required=True);p.add_argument('--review')
    p.add_argument('--out-dir',required=True)
    a=p.parse_args();out=Path(a.out_dir)
    if out.exists():raise FileExistsError('Choose a new output directory')
    timing=read(a.timestamps)
    if digest(timing['video']) != timing['video_sha256']:raise ValueError('Source hash mismatch')
    sha=digest(a.export)
    doc,audit=import_range(load_xml(a.export),timing,a.end_frame_exclusive,sha,read(a.review) if a.review else None)
    write(out/'annotations.json',doc);write(out/'annotation_audit.json',audit)
    with (out/'frame_mapping.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['cvat_frame','source_frame','mot_frame','source_timestamp_seconds'])
        writer.writerows((n,n,n+1,timing['frames'][n]['timestamp_seconds']) for n in range(a.end_frame_exclusive))
    if digest(a.export)!=sha:raise ValueError('Original export changed during import')
    print({k:v for k,v in audit.items() if k not in ('tracks','high_overlap_pairs_iou_at_least_0_8')})
    print('High-overlap pairs requiring inspection:',len(audit['high_overlap_pairs_iou_at_least_0_8']))


if __name__=='__main__':main()
