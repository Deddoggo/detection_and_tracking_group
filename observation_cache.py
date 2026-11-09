"""Versioned raw 2D observations. Standard-library only; no model initialization."""
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction

SCHEMA = 'observations.v1'
COCO17 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
          'right_knee', 'left_ankle', 'right_ankle']


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def finite(value, count):
    return (isinstance(value, list) and len(value) == count and
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value))


def validate_manifest(m):
    if m.get('schema') != SCHEMA or m.get('joint_order') != COCO17:
        raise ValueError('unsupported schema/joint order')
    for key in ('video_sha256', 'input_sha256'):
        value = m.get(key, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'invalid {key}')
    for key in ('width', 'height', 'stride'):
        integer(m[key], key, 1)
    integer(m['start_frame'], 'start_frame')
    integer(m['end_frame_exclusive'], 'end_frame_exclusive', m['start_frame']+1)
    if Fraction(m['time_base']) <= 0 or not m.get('id_namespace'):
        raise ValueError('time base and ID namespace required')
    if m.get('coordinate_units') != 'pixels' or m.get('bbox_format') != 'xywh':
        raise ValueError('raw observations must use original image pixels/xywh')
    if m.get('auxiliary_attributes') != 'disabled':
        raise ValueError('v1 cache does not run auxiliary attribute inference')
    if m.get('inference_kind') not in ('real', 'synthetic-test'):
        raise ValueError('explicit inference provenance required')
    if not isinstance(m.get('pose_model'), dict) or not m.get('provenance'):
        raise ValueError('pose model and code/environment provenance required')


def validate_frame(row, manifest):
    integer(row['source_frame'], 'source_frame')
    f = row['source_frame']
    if not manifest['start_frame'] <= f < manifest['end_frame_exclusive'] or (f-manifest['start_frame']) % manifest['stride']:
        raise ValueError('frame outside scheduled source-frame range')
    integer(row['pts_ticks'], 'pts_ticks')
    expected = float(row['pts_ticks'] * Fraction(manifest['time_base']))
    if not isinstance(row['timestamp_seconds'], (int,float)) or not math.isfinite(row['timestamp_seconds']) or abs(expected-row['timestamp_seconds']) > 1e-9:
        raise ValueError('timestamp does not match rational PTS')
    if row.get('decode_status') != 'ok' or not isinstance(row['persons'], list):
        raise ValueError('decoded frame/person list required')
    ids = set()
    for p in row['persons']:
        integer(p['person_id'], 'person_id')
        if p['person_id'] in ids:
            raise ValueError('duplicate person in frame')
        ids.add(p['person_id'])
        if not finite(p['bbox_xywh'], 4) or min(p['bbox_xywh'][2:]) <= 0:
            raise ValueError('invalid bbox')
        conf = p['detection_confidence']
        if conf is not None and (isinstance(conf,bool) or not isinstance(conf,(int,float)) or not math.isfinite(conf) or not 0 <= conf <= 1):
            raise ValueError('invalid detection confidence')
        for flag in ('outside', 'occluded', 'keyframe'):
            if not isinstance(p['reference'][flag], bool):
                raise ValueError('reference flags must be Boolean')
        if not isinstance(p['reference']['attributes'], dict):
            raise ValueError('reference attributes required, unknown values preserved')
        outside = p['reference']['outside']
        tiny = p['bbox_xywh'][2] < 15 or p['bbox_xywh'][3] < 30
        reason = 'outside' if outside else 'tiny_bbox' if tiny else None
        if p['exclusion_reason'] != reason or p['eligible'] is not (reason is None):
            raise ValueError('eligibility does not match original size/outside policy')
        status, kpts = p['pose_status'], p['keypoints_xy_score']
        if reason is not None:
            if status != 'excluded' or kpts is not None:
                raise ValueError('excluded person must have explicit missing pose')
        elif status == 'ok':
            # ViTPose heatmap scores are raw model outputs, not probabilities.
            if not isinstance(kpts,list) or len(kpts) != 17 or not all(finite(k,3) for k in kpts):
                raise ValueError('COCO17 finite xy/score required')
        elif status != 'missing' or kpts is not None or not p.get('missing_reason'):
            raise ValueError('missing pose requires explicit status/reason')


class CacheWriter:
    """New directory only. Failure leaves an explicitly incomplete cache."""
    def __init__(self, directory, manifest):
        validate_manifest(manifest)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = dict(manifest, state='incomplete')
        self.rows = 0
        self.previous_pts = None
        self.handle = (self.directory/'frames.jsonl').open('x', encoding='utf8')
        self._manifest()

    def _manifest(self):
        # Only this newly-created run is updated; historical runs are never touched.
        target = self.directory/'manifest.json'
        temporary = self.directory/'manifest.pending'
        with temporary.open('x', encoding='utf8') as f:
            json.dump(self.manifest, f, indent=2, allow_nan=False)
        temporary.replace(target)

    def append(self, row):
        validate_frame(row, self.manifest)
        expected = self.manifest['start_frame'] + self.rows*self.manifest['stride']
        if row['source_frame'] != expected or (self.previous_pts is not None and row['pts_ticks'] <= self.previous_pts):
            raise ValueError('missing/duplicated/reordered scheduled frame or PTS')
        self.handle.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
        self.previous_pts = row['pts_ticks']
        self.rows += 1

    def close(self, complete=False):
        if self.handle.closed:
            return
        self.handle.close()
        if complete:
            expected = len(range(self.manifest['start_frame'], self.manifest['end_frame_exclusive'], self.manifest['stride']))
            if self.rows != expected:
                raise ValueError('cannot complete cache with missing scheduled frames')
            from annotation_io import file_sha256
            self.manifest.update(state='complete', frame_records=self.rows,
                                 frames_sha256=file_sha256(self.directory/'frames.jsonl'))
            self._manifest()


def read_cache(directory, expected_video_sha256=None):
    """Validate the entire cache before yielding it to a stateful replay."""
    from annotation_io import file_sha256
    directory = Path(directory)
    m = json.loads((directory/'manifest.json').read_text(encoding='utf8'))
    validate_manifest(m)
    if m.get('state') != 'complete' or (expected_video_sha256 and m['video_sha256'] != expected_video_sha256):
        raise ValueError('incomplete cache or source hash mismatch')
    if file_sha256(directory/'frames.jsonl') != m['frames_sha256']:
        raise ValueError('cache payload hash mismatch')
    rows = []
    with (directory/'frames.jsonl').open(encoding='utf8') as handle:
        for line in handle:
            row = json.loads(line)
            validate_frame(row,m)
            if row['source_frame'] != m['start_frame'] + len(rows)*m['stride']:
                raise ValueError('missing/duplicated/reordered frame')
            if rows and row['pts_ticks'] <= rows[-1]['pts_ticks']:
                raise ValueError('nonmonotonic PTS')
            rows.append(row)
    if len(rows) != m['frame_records'] or len(rows) != len(range(m['start_frame'],m['end_frame_exclusive'],m['stride'])):
        raise ValueError('cache record count mismatch')
    return m, rows
