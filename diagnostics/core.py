import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}; choose a new run path')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def freeze(config, assets=()):
    """Capture this environment; no imports of inference modules or secrets."""
    files = {}
    for path in assets:
        p = Path(path).resolve()
        files[str(p)] = {'exists': p.is_file(),
                         'sha256': digest(p) if p.is_file() else None}
    # Hash every tracked source/config to identify changes beyond the base commit.
    source = {p: digest(p) for p in git('ls-files').splitlines() if Path(p).is_file()}
    for p in ['project_paths.py',
              *map(str, Path('pipeline').glob('*.py')), *map(str, Path('modules').glob('*.py')),
              *map(str, Path('experiments').glob('*.py')), *map(str, Path('evaluation').glob('*.py')),
              *map(str, Path('tools').glob('*.py')), *map(str, Path('legacy').glob('*.py')),
              *map(str, Path('diagnostics').rglob('*.py')), *map(str, Path('tests').glob('*.py'))]:
        if Path(p).is_file():
            source[p] = digest(p)
    return {'utc': datetime.now(timezone.utc).isoformat(), 'commit': git('rev-parse', 'HEAD'),
            'branch': git('branch', '--show-current'), 'git_status': git('status', '--short'),
            'command': sys.argv, 'python': sys.executable, 'python_version': platform.python_version(),
            'platform': platform.platform(),
            'packages': dict(sorted((d.metadata['Name'], d.version)
                                   for d in importlib.metadata.distributions())),
            'config': config, 'assets': files, 'source_sha256': source}


def validate(doc):
    if doc.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    if doc.get('kind') not in ('detections', 'tracks', 'annotations', 'groups'):
        raise ValueError('unknown observation kind')
    source = doc['source']
    start, end = source['start_frame'], source['end_frame_exclusive']
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        raise ValueError('invalid absolute zero-based frame range')
    if not isinstance(source.get('video'), str) or not source['video']:
        raise ValueError('source video identity is required')
    seen = set()
    for frame in doc['frames']:
        n = frame['frame']
        if not isinstance(n, int) or not start <= n < end or n in seen:
            raise ValueError('invalid or duplicate frame')
        seen.add(n)
        ids = []
        for item in frame.get('items', []):
            if doc['kind'] == 'groups':
                continue
            b = item['bbox']
            if len(b) != 4 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in b) or min(b[2:]) <= 0:
                raise ValueError('bbox is finite pixel xywh with positive width/height')
            c = item.get('confidence')
            if c is not None and (not math.isfinite(c) or not 0 <= c <= 1):
                raise ValueError('confidence must be null or in [0,1]')
            if doc['kind'] == 'tracks':
                if not isinstance(item.get('person_id'), int) or item['person_id'] < 0:
                    raise ValueError('tracks require nonnegative integer person_id')
                ids.append(item['person_id'])
            if doc['kind'] == 'annotations' and item.get('person_id') is not None:
                if not isinstance(item['person_id'], int) or item['person_id'] < 0:
                    raise ValueError('annotation identity must be nonnegative integer or null')
                ids.append(item['person_id'])
            if doc['kind'] == 'annotations':
                visibility = item.get('visibility', 1)
                if not isinstance(visibility, (int, float)) or not 0 <= visibility <= 1:
                    raise ValueError('visibility must be a fraction in [0,1]')
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate person identity in frame')
    return doc


def require_dense(doc, annotated=False):
    validate(doc)
    expected = set(range(doc['source']['start_frame'], doc['source']['end_frame_exclusive']))
    if {f['frame'] for f in doc['frames']} != expected:
        raise ValueError('full contiguous frame coverage is required (including empty frames)')
    if annotated:
        if doc.get('annotation_status') != 'human_reviewed':
            raise ValueError('human_reviewed annotation_status required; predictions are not ground truth')
        for f in doc['frames']:
            if f.get('coverage') != 'full':
                raise ValueError('human full-frame coverage confirmation required')
            if any(i.get('ignore', False) or i.get('identity_uncertain', False) or i.get('person_id') is None for i in f['items']):
                raise ValueError('tracking/oracle export requires resolved identities and no ignored regions; split pilot at uncertainty')


def same_source(a, b):
    keys = ('video', 'start_frame', 'end_frame_exclusive')
    if any(a['source'].get(k) != b['source'].get(k) for k in keys):
        raise ValueError('prediction and annotation source/frame ranges differ')
    for k in ('video_sha256', 'fps'):
        if a['source'].get(k) is not None and b['source'].get(k) is not None and a['source'][k] != b['source'][k]:
            raise ValueError(f'source {k} mismatch')


def iou(a, b):
    x = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    intersection = x*y
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection)


def match(pred, gt, threshold=0.5):
    """Maximum-cardinality bipartite IoU matching, deterministic augmenting paths.

    Edges visited by descending IoU then GT index; NOT global max-sum-IoU.
    """
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0,1]')
    adjacency = [sorted((j for j, g in enumerate(gt) if iou(p['bbox'], g['bbox']) >= threshold),
                        key=lambda j: (-iou(p['bbox'], gt[j]['bbox']), j)) for p in pred]
    owners = {}

    def augment(p, seen):
        for g in adjacency[p]:
            if g not in seen:
                seen.add(g)
                if g not in owners or augment(owners[g], seen):
                    owners[g] = p
                    return True
        return False

    for p in range(len(pred)):
        augment(p, set())
    return sorted((p, g) for g, p in owners.items())


def evaluate_detection(predictions, annotations, threshold=0.5):
    validate(predictions); validate(annotations); same_source(predictions, annotations)
    if predictions['kind'] != 'detections' or annotations['kind'] != 'annotations':
        raise ValueError('expected detections and human annotations')
    if annotations.get('annotation_status') != 'human_reviewed':
        raise ValueError('annotations must be explicitly human-reviewed')
    pf = {f['frame']: f['items'] for f in predictions['frames']}
    counts = dict(tp=0, fp=0, fn=0, duplicates=0, ignored_predictions=0, evaluated_frames=0)
    for f in annotations['frames']:
        if f.get('coverage') != 'full':
            raise ValueError('detection metrics require full annotation of each evaluated frame')
        if f['frame'] not in pf:
            raise ValueError('missing prediction frame is not assumed to be empty')
        gt = [i for i in f['items'] if not i.get('ignore', False)]
        ignored = [i for i in f['items'] if i.get('ignore', False)]
        pred = pf[f['frame']]
        pairs = match(pred, gt, threshold)
        used = {p for p, _ in pairs}
        counts['tp'] += len(pairs); counts['fn'] += len(gt)-len(pairs)
        for p, item in enumerate(pred):
            if p in used:
                continue
            # Duplicates remain false positives, even if also overlapping ignore boxes.
            duplicate = any(iou(item['bbox'], g['bbox']) >= threshold for g in gt)
            if not duplicate and any(iou(item['bbox'], g['bbox']) >= threshold for g in ignored):
                counts['ignored_predictions'] += 1
                continue
            counts['fp'] += 1
            counts['duplicates'] += int(duplicate)
        counts['evaluated_frames'] += 1
    tp, fp, fn = (counts[x] for x in ('tp', 'fp', 'fn'))
    return {**counts, 'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'iou_threshold': threshold, 'scope': 'annotated frames only',
            'matching': 'maximum cardinality, descending-IoU edge visitation',
            'duplicate_definition': 'unmatched prediction with IoU >= threshold to a scored GT person; subset of FP'}
