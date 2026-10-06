"""python -m diagnostics.cli --help (no inference dependencies for cache tools)."""
import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

from .core import (digest, evaluate_detection, freeze, match, read, require_dense,
                   same_source, validate, write)


def inventory(root):
    entries = []
    for p in sorted(Path(root).rglob('*')):
        if p.suffix.lower() not in ('.mp4', '.avi', '.webm', '.mov', '.mkv'):
            continue
        e = {'path': str(p.resolve()), 'bytes': p.stat().st_size,
             'role': 'unknown; human confirmation required', 'camera_motion': 'unknown',
             'scene': None, 'session': None, 'metadata': None}
        try:
            if shutil.which('ffprobe'):
                r = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames:format=duration',
                    '-of', 'json', str(p)], text=True)
                e['metadata'] = json.loads(r)
            else:
                import cv2
                cap = cv2.VideoCapture(str(p))
                if not cap.isOpened():
                    raise ValueError('video cannot be opened')
                fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                e['metadata'] = {'fps': fps, 'frame_count': n,
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'duration_seconds': n/fps if fps else None}
                cap.release()
        except (ImportError, OSError, ValueError, subprocess.CalledProcessError) as exc:
            e['metadata_error'] = str(exc)
        entries.append(e)
    return {'videos': entries, 'selection': 'No pilot selected automatically; inspect raw inputs and confirm sessions.'}


def convert_legacy(path, kind, source):
    doc = {'schema_version': 1, 'kind': kind, 'source': source, 'frames': [],
           'provenance': {'imported_from': str(Path(path).resolve()), 'sha256': digest(path),
                          'historical_run_config': 'unknown'}}
    for key, records in sorted(read(path).items()):
        n = int(key.removeprefix('frame_'))
        if not source['start_frame'] <= n < source['end_frame_exclusive']:
            continue
        items = []
        for r in records:
            i = {'bbox': r['bbox'], 'confidence': r.get('conf')}
            if kind == 'tracks':
                i['person_id'] = int(r.get('id', r.get('id_p')))
            # Cached keypoints/directions are observations, never reference annotations.
            for k in ('keypoints', 'direction', 'direction_confidence'):
                if k in r:
                    i[k] = r[k]
            items.append(i)
        doc['frames'].append({'frame': n, 'items': items})
    if kind == 'detections':
        doc['cache'] = {'confidence_floor': None, 'nms_iou': None,
                        'warning': 'historical thresholds unknown; sweep completeness cannot be certified'}
    return validate(doc)


def filter_cache(doc, threshold):
    validate(doc)
    floor = doc.get('cache', {}).get('confidence_floor')
    if doc['kind'] != 'detections' or floor is None:
        raise ValueError('sweep requires a detection cache with verified confidence_floor')
    if not floor <= threshold <= 1:
        raise ValueError('threshold below cached floor or above 1')
    out = copy.deepcopy(doc)
    for f in out['frames']:
        if any(i.get('confidence') is None for i in f['items']):
            raise ValueError('sweep requires confidence scores')
        f['items'] = [i for i in f['items'] if i['confidence'] >= threshold]
    out['cache']['operating_threshold'] = threshold
    return out


def annotation_template(source):
    return {'schema_version': 1, 'kind': 'annotations', 'source': source,
            'annotation_status': 'pending', 'annotator': None,
            'frames': [{'frame': n, 'coverage': 'pending', 'items': []}
                       for n in range(source['start_frame'], source['end_frame_exclusive'])]}


def oracle(annotations, mode, detections=None, threshold=0.5):
    if annotations['kind'] != 'annotations':
        raise ValueError('oracle requires human annotations')
    require_dense(annotations, annotated=True)
    out = copy.deepcopy(annotations)
    out['kind'] = 'detections' if mode == 'gt_boxes' else 'tracks'
    out['oracle'] = {'mode': mode, 'annotation_source': 'human reference',
                     'iou_threshold': threshold, 'not_a_deployed_pipeline': True}
    if mode == 'matched_detection_ids':
        if detections is None or detections['kind'] != 'detections':
            raise ValueError('matched_detection_ids requires detections')
        require_dense(detections); same_source(detections, annotations)
        df = {f['frame']: f['items'] for f in detections['frames']}
    counts = {'dropped_unmatched_detections': 0, 'missed_gt_people': 0}
    for f in out['frames']:
        if mode == 'matched_detection_ids':
            d = df[f['frame']]
            pairs = match(d, f['items'], threshold)
            counts['dropped_unmatched_detections'] += len(d)-len(pairs)
            counts['missed_gt_people'] += len(f['items'])-len(pairs)
            f['items'] = [{**copy.deepcopy(d[p]), 'person_id': f['items'][g]['person_id']}
                          for p, g in pairs]
        else:
            f['items'] = [{'bbox': i['bbox'], 'confidence': 1.0,
                          **({'person_id': i['person_id']} if out['kind'] == 'tracks' else {})}
                          for i in f['items']]
        f.pop('coverage', None)
    out['oracle']['exclusions'] = counts
    if mode == 'matched_detection_ids':
        out['oracle']['policy'] = 'misses stay missing; unmatched FP and duplicate detections dropped; optimistic matched-only association diagnostic'
    return validate(out)


def mot_export(doc, target):
    annotated = doc['kind'] == 'annotations'
    if not annotated and doc['kind'] != 'tracks':
        raise ValueError('MOT export requires tracks or annotations')
    require_dense(doc, annotated=annotated)
    rows = []
    # MOT uses local 1-based frames and positive IDs; source uses absolute 0-based frames.
    ids = sorted({i['person_id'] for f in doc['frames'] for i in f['items']})
    id_map = {i: n+1 for n, i in enumerate(ids)}
    for f in sorted(doc['frames'], key=lambda f: f['frame']):
        n = f['frame']-doc['source']['start_frame']+1
        for i in f['items']:
            x, y, w, h = i['bbox']
            c = 1 if annotated else i.get('confidence')
            c = -1 if c is None else c
            suffix = f'1,{i.get("visibility", -1)}' if annotated else '-1,-1,-1'
            rows.append(f'{n},{id_map[i["person_id"]]},{x},{y},{w},{h},{c},{suffix}')
    path = Path(target)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(rows)+'\n', encoding='utf8')
    write(str(path)+'.metadata.json', {'source': doc['source'], 'kind': doc['kind'],
                                     'identity_map': id_map, 'full_contiguous_coverage': True,
                                     'unknown_visibility_sentinel': -1,
                                     'visibility_is_not_used_for_gt_filtering': True})


def prepare_trackeval(annotations, tracks, target, sequence):
    """Prepare strict fully annotated pilot for the external official evaluator."""
    import re
    require_dense(annotations, annotated=True); require_dense(tracks)
    same_source(annotations, tracks)
    if annotations['kind'] != 'annotations' or tracks['kind'] != 'tracks':
        raise ValueError('Expected annotations and tracks')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', sequence):
        raise ValueError('Use a simple sequence name')
    root = Path(target)
    if root.exists():
        raise FileExistsError('Choose a fresh TrackEval folder')
    s = annotations['source']
    if any(not isinstance(s.get(k), (int,float)) or s[k] <= 0 for k in ('fps','width','height')):
        raise ValueError('TrackEval preparation requires source fps/width/height')
    mot_export(annotations, root/'gt'/sequence/'gt'/'gt.txt')
    mot_export(tracks, root/'trackers'/'estimated'/'data'/f'{sequence}.txt')
    (root/'gt'/sequence/'seqinfo.ini').write_text(
        f'[Sequence]\nname={sequence}\nimDir=img1\nframeRate={s["fps"]}\n'
        f'seqLength={s["end_frame_exclusive"]-s["start_frame"]}\n'
        f'imWidth={s["width"]}\nimHeight={s["height"]}\nimExt=.jpg\n',encoding='utf8')
    (root/'seqmap.txt').write_text('name\n'+sequence+'\n',encoding='utf8')
    write(root/'evaluation_manifest.json',{'source':s,'coverage':'contiguous fully annotated pilot',
          'gt_status':annotations['annotation_status'],'tracking_provenance':tracks.get('provenance'),
          'metrics_to_run':['HOTA','CLEAR','Identity'], 'preprocessing':'disabled; strict person-only no-ignore export',
          'execution_status':'prepared only; official TrackEval not executed'})


def grouping(doc, config, raw_baseline=False):
    require_dense(doc)
    if doc['kind'] != 'tracks':
        raise ValueError('grouping requires tracks')
    import numpy as np
    from modules.identify_group_social import cluster_bboxes_with_ids
    matrix = None
    g = config['grouping']
    if raw_baseline:
        import types
        if g['orientation_mode'] != 'legacy':
            raise ValueError('Raw source baseline has the legacy gate; set orientation_mode=legacy')
        archived_source = Path(__file__).parent/'baselines'/'identify_group_social.py'
        module = types.ModuleType('archived_grouping')
        exec(compile(archived_source.read_bytes(), str(archived_source), 'exec'), module.__dict__)
        cluster_bboxes_with_ids = module.cluster_bboxes_with_ids
    if g['calibration_file'] is not None:
        if g.get('calibration_source_video') != doc['source']['video'] and not g.get('allow_unverified_calibration', False):
            raise ValueError('Confirm calibration_source_video or explicitly allow unverified calibration for inspection only')
        matrix = np.array(read(g['calibration_file'])['homography_matrix'], dtype=np.float32)
        if matrix.shape != (3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError('invalid/singular homography')
    if matrix is None and g['coordinate_unit'] != 'pixels':
        raise ValueError('without homography eps must explicitly be in pixels')
    if g['scale_factor'] <= 0 or g['eps'] <= 0 or not isinstance(g['min_samples'],int) or g['min_samples'] < 1:
        raise ValueError('positive grouping scale/eps and integer min_samples required')
    previous, counter = {}, -1
    out = {'schema_version': 1, 'kind': 'groups', 'source': doc['source'], 'frames': [],
           'provenance': freeze(config, [g['calibration_file']] if g['calibration_file'] else []),
           'input_oracle': doc.get('oracle', doc.get('input_oracle')), 'tracking_provenance': doc.get('provenance'),
           'implementation': 'raw_2db7aac' if raw_baseline else 'corrected_worktree',
           'cues': {'orientation_mode': g['orientation_mode'], 'coordinate_unit': g['coordinate_unit'],
                    'directions': 'cached image-pose proxies; not verified ground heading',
                    'direction_projection': 'homography(footpoint + 10px * direction) minus homography(footpoint), normalized',
                    'missing_direction_policy': 'proximity fallback', 'close_pair_bypass': 0.5}}
    for f in sorted(doc['frames'], key=lambda f: f['frame']):
        records = f['items']; old = previous
        orientation_options = {} if raw_baseline else {
            'pose_confidences': [i.get('direction_confidence') for i in records],
            'orientation_mode': g['orientation_mode']}
        results, previous, counter = cluster_bboxes_with_ids(
            previous, [i['bbox'] for i in records], [i['person_id'] for i in records], counter,
            pose_vectors=[i.get('direction') for i in records], homography_matrix=matrix,
            scale_factor=g['scale_factor'], eps=g['eps'], min_samples=g['min_samples'],
            threshold_overlap=g['retention_threshold'], **orientation_options)
        events = []
        for gid, members in previous.items():
            parents = sorted(p for p, om in old.items() if set(members) & set(om))
            events.append({'group_id': gid, 'overlap_parents': parents,
                           'continued_from': gid if gid in old else None,
                           'split_parents': [p for p in parents if sum(
                               bool(set(om) & set(old[p])) for om in previous.values()) > 1],
                           'merge': len(parents) > 1})
        out['frames'].append({'frame': f['frame'], 'items': [
            {'group_id': gid, 'members': members} for gid, members in previous.items()],
            'people': [{'person_id': int(r['id_p']), 'group_id': int(r['id_g']),
                        'bbox': list(map(float, r['bbox'])),
                        'position': r['real_pos'].tolist() if r['real_pos'] is not None else None,
                        'direction': records[j].get('direction'),
                        'direction_confidence': records[j].get('direction_confidence')}
                       for j, r in enumerate(results)], 'overlap_lineage': events})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inventory'); p.add_argument('--root', default='.'); p.add_argument('--out', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('convert'); p.add_argument('--input', required=True); p.add_argument('--kind', choices=['detections','tracks'], required=True); p.add_argument('--source', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('validate'); p.add_argument('--input', required=True)
    p = sub.add_parser('annotation-template'); p.add_argument('--source', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('evaluate-detection'); p.add_argument('--input', required=True); p.add_argument('--annotations', required=True); p.add_argument('--iou', type=float, default=0.5); p.add_argument('--out', required=True)
    p = sub.add_parser('sweep'); p.add_argument('--input', required=True); p.add_argument('--thresholds', type=float, nargs='+', required=True); p.add_argument('--out-dir', required=True)
    p = sub.add_parser('oracle'); p.add_argument('--annotations', required=True); p.add_argument('--mode', choices=['gt_boxes','gt_boxes_ids','matched_detection_ids'], required=True); p.add_argument('--detections'); p.add_argument('--iou', type=float, default=0.5); p.add_argument('--out', required=True)
    p = sub.add_parser('mot-export'); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('prepare-trackeval'); p.add_argument('--annotations', required=True); p.add_argument('--tracks', required=True); p.add_argument('--sequence', required=True); p.add_argument('--out-dir', required=True)
    p = sub.add_parser('group'); p.add_argument('--input', required=True); p.add_argument('--config', required=True); p.add_argument('--out', required=True); p.add_argument('--raw-baseline', action='store_true')
    a = parser.parse_args()
    if a.command == 'inventory': result = inventory(a.root)
    elif a.command == 'freeze':
        c = read(a.config)
        assets = [str(p) for p in Path('.').glob('calibration*.json')]
        assets += [c.get('detector', {}).get('checkpoint'), c.get('tracker', {}).get('config_file')]
        result = freeze(c, [p for p in assets if p])
    elif a.command == 'convert': result = convert_legacy(a.input, a.kind, read(a.source))
    elif a.command == 'validate': validate(read(a.input)); print('valid'); return
    elif a.command == 'annotation-template': result = annotation_template(read(a.source))
    elif a.command == 'evaluate-detection': result = evaluate_detection(read(a.input), read(a.annotations), a.iou)
    elif a.command == 'sweep':
        d = read(a.input)
        for t in a.thresholds: write(Path(a.out_dir)/f'detections_{t:.3f}.json', filter_cache(d, t))
        return
    elif a.command == 'oracle': result = oracle(read(a.annotations), a.mode, read(a.detections) if a.detections else None, a.iou)
    elif a.command == 'mot-export': mot_export(read(a.input), a.out); return
    elif a.command == 'prepare-trackeval': prepare_trackeval(read(a.annotations), read(a.tracks), a.out_dir, a.sequence); return
    elif a.command == 'group': result = grouping(read(a.input), read(a.config), a.raw_baseline)
    write(a.out, result)


if __name__ == '__main__':
    main()
