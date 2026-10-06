"""Bounded, explicit detector/tracker runner. No checkpoint substitution/download."""
import argparse
import copy
from pathlib import Path

from .core import digest, freeze, read, require_dense, validate, write


def open_pilot(config):
    import cv2
    p = config['pilot']
    if not p.get('video') or not Path(p['video']).is_file():
        raise FileNotFoundError('Set pilot.video to the confirmed raw source')
    if p.get('start_frame') is None or p.get('end_frame_exclusive') is None:
        raise ValueError('Select a reviewed pilot frame range first')
    start, end = p['start_frame'], p['end_frame_exclusive']
    if not isinstance(start,int) or not isinstance(end,int) or not 0 <= start < end:
        raise ValueError('pilot frame indices must be increasing nonnegative integers')
    cap = cv2.VideoCapture(p['video'])
    if not cap.isOpened():
        raise ValueError('Cannot open source video')
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_times = None
    duration = (end-start)/fps if fps > 0 else float('inf')
    timing = None
    if p.get('frame_timestamp_file'):
        timing = read(p['frame_timestamp_file'])
        if Path(timing['video']).resolve() != Path(p['video']).resolve():
            cap.release()
            raise ValueError('timestamp table belongs to another video')
        times = timing['frames']
        if len(times) != count or any(f['frame'] != n for n,f in enumerate(times)) or end >= len(times):
            cap.release()
            raise ValueError('timestamp table/count/range mismatch; need end-boundary frame')
        duration = times[end]['timestamp_seconds']-times[start]['timestamp_seconds']
        clip_times = [f['timestamp_seconds'] for f in times[start:end]]
    limit = p.get('max_duration_seconds', 20)
    if not isinstance(limit, (int, float)) or not 0 < limit <= 60:
        cap.release()
        raise ValueError('Explicit diagnostic duration limit must be in (0,60] seconds')
    if fps <= 0 or not 0 <= start < end <= count or not 0 < duration <= limit:
        cap.release()
        raise ValueError('Pilot exceeds its explicit bounded duration; no full-video inference')
    video_hash = digest(p['video'])
    if timing is not None and timing.get('video_sha256') != video_hash:
        cap.release()
        raise ValueError('timestamp table video hash mismatch')
    source = {'video': str(Path(p['video']).resolve()), 'video_sha256': video_hash,
              'fps': timing.get('average_fps', fps) if timing else fps,
              'opencv_reported_fps': fps, 'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 'total_frame_count': count,
              'start_frame': start, 'end_frame_exclusive': end,
              'duration_seconds': duration, 'frame_timestamps_seconds': clip_times,
              'timestamp_basis': 'decoded PTS' if clip_times is not None else 'nominal OpenCV FPS; unverified constant-rate assumption',
              'scene': p.get('scene'), 'session': p.get('session')}
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if abs(position-start) > 0.1:
        cap.release()
        raise ValueError('Video backend cannot seek to exact pilot frame')
    return cap, source


def detect(config):
    d = config['detector']
    checkpoint = d.get('checkpoint')
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError('Explicit local detector checkpoint required; downloads/fallback disabled')
    from ultralytics import YOLO
    cap, source = open_pilot(config)
    try:
        model = YOLO(str(Path(checkpoint).resolve()))
        model.to(d['device'])
        out = {'schema_version': 1, 'kind': 'detections', 'source': source, 'frames': [],
               'cache': {'confidence_floor': d['confidence_floor'], 'nms_iou': d['nms_iou'],
                         'imgsz': d['imgsz'], 'max_det': d['max_det'], 'classes': [0],
                         'stage': 'post-NMS YOLO observations; not pre-NMS proposals'},
               'provenance': freeze(config, [checkpoint, *([config['pilot']['frame_timestamp_file']] if config['pilot'].get('frame_timestamp_file') else [])])}
        for n in range(source['start_frame'], source['end_frame_exclusive']):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Video ended before expected frame {n}')
            r = model.predict(image, conf=d['confidence_floor'], iou=d['nms_iou'],
                              imgsz=d['imgsz'], max_det=d['max_det'], classes=[0],
                              half=d['half'], device=d['device'], verbose=False)[0]
            boxes, scores = r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist()
            out['frames'].append({'frame': n, 'items': [
                {'bbox': [x1,y1,x2-x1,y2-y1], 'confidence': score}
                for (x1,y1,x2,y2), score in zip(boxes,scores)]})
        return validate(out)
    finally:
        cap.release()


def track(config, detections):
    t = config['tracker']
    if detections['kind'] != 'detections':
        raise ValueError('Tracker requires detection observations, including oracle GT boxes')
    require_dense(detections)
    if t['type'] != 'bytetrack':
        raise ValueError('First pilot adapter intentionally supports ByteTrack only')
    if not t.get('config_file') or not Path(t['config_file']).is_file():
        raise FileNotFoundError('Provide the actual installed-version ByteTrack YAML; no implicit defaults')
    import inspect
    import numpy as np
    from boxmot.trackers.tracker_zoo import create_tracker
    from .core import same_source
    cap, source = open_pilot(config)
    try:
        same_source({'source': source}, detections)
        tracker = create_tracker(tracker_type='bytetrack', tracker_config=Path(t['config_file']),
                                 reid_weights=None, device=t['device'], half=t['half'], per_class=False)
        implementation = inspect.getfile(type(tracker))
        out = {'schema_version': 1, 'kind': 'tracks', 'source': source, 'frames': [],
               'provenance': freeze(config, [t['config_file'], implementation]),
               'detection_provenance': detections.get('provenance'),
               'detection_cache': detections.get('cache'), 'input_oracle': detections.get('oracle'),
               'tracker_effective_scalars': {k: v for k,v in vars(tracker).items()
                                            if isinstance(v,(str,int,float,bool)) or v is None}}
        frames = {f['frame']: f['items'] for f in detections['frames']}
        floor = detections.get('cache', {}).get('confidence_floor')
        if floor is not None and t['input_confidence'] < floor:
            raise ValueError('Tracker input threshold below cached detector floor')
        if any(i.get('confidence') is None for items in frames.values() for i in items):
            raise ValueError('Tracker input requires measured detection confidence')
        for n in range(source['start_frame'], source['end_frame_exclusive']):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Video ended before frame {n}')
            ds = [i for i in frames[n] if i['confidence'] >= t['input_confidence']]
            det = np.array([[x,y,x+w,y+h,i['confidence'],0] for i in ds
                            for x,y,w,h in [i['bbox']]], dtype=float).reshape(-1,6)
            result = np.asarray(tracker.update(det, image))
            if result.size and (result.ndim != 2 or result.shape[1] < 7):
                raise ValueError('Unverified BoxMOT output layout; expected xyxy,id,conf,class[,det_index]')
            items = []
            for r in result:
                if r[4] < 0:
                    continue
                x1,y1,x2,y2 = map(float,r[:4])
                items.append({'bbox': [x1,y1,x2-x1,y2-y1], 'person_id': int(r[4]),
                              'confidence': float(r[5]),
                              'detection_index': int(r[7]) if len(r)>7 else None})
            out['frames'].append({'frame': n, 'items': items})
        return validate(out)
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--stage', choices=['detect','track'], required=True)
    p.add_argument('--detections'); p.add_argument('--out', required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        raise FileExistsError('Choose a new output path')
    c = read(a.config)
    result = detect(c) if a.stage == 'detect' else track(c, read(a.detections))
    write(a.out, result)


if __name__ == '__main__':
    main()
