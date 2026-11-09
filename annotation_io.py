"""CVAT person observations and source PTS; no neural or grouping dependencies."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_cvat(path):
    """Read explicit exported boxes only. Never fill outside/absent intervals."""
    import math
    root = ET.parse(path).getroot()
    meta = root.find('meta')
    if root.findtext('version') != '1.1' or meta is None:
        raise ValueError('expected CVAT 1.1 video-track XML')
    job = meta.find('job')
    if job is None:
        job = meta.find('task')
    if job is None:
        raise ValueError('missing CVAT job/task metadata')
    size = meta.find('original_size')
    if size is None:
        size = job.find('original_size')
    if job is None or size is None or job.findtext('frame_filter', '').strip():
        raise ValueError('missing video metadata or unsupported CVAT frame_filter')
    metadata = {'schema': 'cvat-1.1', 'job_id': job.findtext('id'),
                'start_frame': int(job.findtext('start_frame')),
                'stop_frame': int(job.findtext('stop_frame')),
                'width': int(size.findtext('width')), 'height': int(size.findtext('height')),
                'source_sha256': file_sha256(path)}
    frames, seen_tracks = {}, set()
    for track in root.findall('track'):
        if track.get('label') != 'person':
            continue
        tid = int(track.attrib['id'])
        if tid < 0 or tid in seen_tracks:
            raise ValueError('duplicate or negative person track ID')
        seen_tracks.add(tid)
        seen_frames = set()
        for box in track.findall('box'):
            f = int(box.attrib['frame'])
            if f in seen_frames or not metadata['start_frame'] <= f <= metadata['stop_frame']:
                raise ValueError('duplicate track/frame or out-of-range annotation')
            seen_frames.add(f)
            x1, y1, x2, y2 = (float(box.attrib[k]) for k in ('xtl', 'ytl', 'xbr', 'ybr'))
            if not all(map(math.isfinite, (x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
                raise ValueError('invalid CVAT bbox')
            flags = {}
            for key in ('outside', 'occluded', 'keyframe'):
                value = box.get(key)
                if value not in ('0', '1'):
                    raise ValueError('CVAT flags must be 0 or 1')
                flags[key] = value == '1'
            attrs = {a.attrib['name']: a.text or '' for a in box.findall('attribute')}
            frames.setdefault(f, []).append({'person_id': tid, 'bbox_xywh': [x1, y1, x2-x1, y2-y1],
                                             **flags, 'attributes': attrs})
    for people in frames.values():
        people.sort(key=lambda p: p['person_id'])
    return metadata, frames


def probe_video(path):
    """Read demuxed PTS, refusing B-frame/reordered packet assumptions."""
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_streams', '-show_packets', '-show_entries',
        'stream=width,height,nb_frames,time_base,has_b_frames,avg_frame_rate,r_frame_rate,duration:packet=pts,dts',
        '-of', 'json', str(path)], text=True)
    data = json.loads(raw)
    stream, = data['streams']
    packets = data['packets']
    if stream.get('has_b_frames', 0) != 0 or any(p.get('pts') != p.get('dts') for p in packets):
        raise ValueError('reordered video requires a decoded presentation-frame index')
    pts = [int(p['pts']) for p in packets]
    if not pts or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('missing/nonmonotonic PTS')
    if int(stream.get('nb_frames', len(pts))) != len(pts):
        raise ValueError('packet/frame count mismatch; decoded index required')
    return {'video_sha256': file_sha256(path), 'width': int(stream['width']),
            'height': int(stream['height']), 'time_base': stream['time_base'],
            'frame_count': len(pts), 'pts_ticks': pts, 'stream': stream,
            'mapping_basis': 'ordered_packets_no_b_frames; decoder verification required'}


def pts_seconds(ticks, time_base):
    return float(int(ticks) * Fraction(time_base))


def verify_decode(video, timing, end, sample_frames=(), output=None, annotations=None):
    """Sequential decode verifies count, dimensions and OpenCV PTS through end.

    Optional review images are evidence for humans, not an automatic alignment verdict.
    """
    import cv2
    if not 0 < end <= timing['frame_count']:
        raise ValueError('invalid exclusive source-frame end')
    output = Path(output) if output is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(video))
    samples, maximum_error = [], 0.0
    try:
        for index in range(end):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'decode ended before source frame {index}')
            if image.shape[:2] != (timing['height'], timing['width']):
                raise ValueError('decoder dimensions differ from metadata')
            expected = pts_seconds(timing['pts_ticks'][index], timing['time_base'])
            observed = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            error = abs(observed - expected)
            maximum_error = max(maximum_error, error)
            if error > 0.002:
                raise ValueError(f'decoder/packet PTS mismatch at {index}: {observed} vs {expected}')
            if index in sample_frames:
                if annotations is not None:
                    for p in annotations.get(index, []):
                        if p['outside']:
                            continue
                        x, y, w, h = p['bbox_xywh']
                        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                        label = f"{p['person_id']} {p['attributes'].get('social_group', '')}"
                        cv2.putText(image, label, (int(x), max(20, int(y)-3)), 0, .45, (0,255,255), 1)
                cv2.putText(image, f'source={index} PTS={expected:.6f}s', (20, 35), 0, 1, (0,0,255), 2)
                name = f'frame_{index:05d}.jpg'
                if output is not None and not cv2.imwrite(str(output/name), image):
                    raise OSError('could not write review image')
                samples.append({'source_frame': index, 'pts_seconds': expected, 'image': name if output else None})
    finally:
        cap.release()
    return {'verified_start': 0, 'verified_end_exclusive': end, 'decoded_frames': end,
            'maximum_pts_error_seconds': maximum_error, 'samples': samples,
            'visual_alignment': 'requires review of sample images'}
