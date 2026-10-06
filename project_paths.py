"""Repository-relative paths and lightweight CLI configuration (no ML imports)."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    'video': 'videos/WIN_20260326_10_00_11_Pro.mp4',
    'output_dir': 'playground_results',
    'detector_checkpoint': 'models/yolo26x.pt',
    'calibration': 'configs/calibration/calibration_matrix_playground.json',
    'tracker_config': None,
    'models_dir': 'models',
    'external_dir': 'external',
}


def repo_path(value):
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return str(p if p.is_absolute() else ROOT / p)


def load_paths(config=None):
    values = dict(DEFAULTS)
    configured = config or os.environ.get('PLAYGROUND_CONFIG')
    candidate = Path(repo_path(configured)) if configured else ROOT/'configs/paths.local.json'
    if candidate.is_file():
        supplied = json.loads(candidate.read_text(encoding='utf8'))
        unknown = set(supplied)-set(values)
        if unknown: raise ValueError(f'Unknown path settings: {sorted(unknown)}')
        values.update(supplied)
    elif configured:
        raise FileNotFoundError(candidate)
    for key,value in values.items():
        if value is not None:
            if not isinstance(value,str) or not value: raise ValueError(f'Invalid path setting: {key}')
            values[key] = repo_path(value)
    return values


def stage_paths(stage):
    p = argparse.ArgumentParser(description=f'Playground {stage}; configure paths only, algorithm defaults unchanged.')
    p.add_argument('--config', help='Path JSON; relative paths resolve from repository root')
    p.add_argument('--video');p.add_argument('--output-dir')
    p.add_argument('--checkpoint', dest='detector_checkpoint')
    p.add_argument('--calibration');p.add_argument('--tracker-config')
    args = vars(p.parse_args());values = load_paths(args.pop('config'))
    values.update({k:repo_path(v) for k,v in args.items() if v is not None})
    return values


def checkpoint_path(name):
    return str(Path(load_paths()['models_dir']) / name)


def external_path(name=''):
    return str(Path(load_paths()['external_dir']) / name)


def output_path(name):
    return str(Path(load_paths()['output_dir']) / name)
