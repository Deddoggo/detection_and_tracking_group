# Playground detection, tracking and subgroup analysis

Research project on monocular playground video: detect people, maintain person tracks,
stitch fragments offline, and explore spatial groups using calibration and pose cues.
Edit canonical implementations in `pipeline/`, not the compatibility wrappers at the root.

## Status and evidence

The research project implemented tracking and exploratory playground/group visualizations.
Historical quantitative tracking results used the **MOT15 training split**, not a held-out
playground test set. [Archived results](evaluation/archive/eval_result.txt) are preserved;
this checkout is not established as the exact code/config that generated those runs.

Five private raw playground videos total approximately 74 minutes. Person bbox/ID annotations
were subsequently reviewed for source frames **0–900 inclusive** in one video. Keep annotation
review/exclusion decisions with each export and evaluation. This is a diagnostic/development
set, not evidence of generalization. There is **no group membership GT or quantitative subgroup
validation**. Later code/log audits and diagnostic infrastructure are distinct from thesis experiments.

## Repository map

```text
pipeline/               Four-stage playground pipeline
modules/                Shared grouping, group identity and orientation logic
configs/
  paths.example.json    Copy to ignored paths.local.json and edit for your machine
  calibration/          Camera calibration JSON files
  trackers/             Existing alternative tracker configurations
experiments/            YOLO/pose/MixSTE/SAM and clustering research variants
legacy/                 Earlier Detectron2/DeepSORT route and original requirements
evaluation/             Historical public-MOT scripts and archived result
diagnostics/            Pilot, annotation, conversion, evaluation and audit utilities
tests/                  Lightweight correctness and layout tests
tools/                  Calibration, measurement, reports and visualization
docs/                   Inventory, path mapping and research notes
project_paths.py        Repository-relative path configuration
*.py                    Compatibility wrappers for old root entry points
```

Local assets excluded from Git include `videos/`, `models/`, `external/`, `assets/reference/`,
annotation exports and generated outputs. Existing assets remain on disk. Experiments and
legacy code are preserved, not declared obsolete or interchangeable with the main pipeline.
See the complete [old → new mapping and inventory](docs/REPOSITORY_LAYOUT.md).

## Pipeline and data contracts

| Stage | Input | Output |
|---|---|---|
| `pipeline.detect_yolo` | Video + YOLO checkpoint | `detections.json`, visualization |
| `pipeline.track` | Same video + detections | `tracks_bytetrack_only.json`, visualization |
| `pipeline.stitch_tracks` | Same video + raw tracks | `tracks_stitched.json` |
| `tools.calibration_headless` (camera preparation) | Camera frame + four ground-plane correspondences and measured dimensions | Calibration JSON + warped validation image |
| `pipeline.analyze_subgroups` | Same video + stitched tracks (falls back to raw tracks), calibration, pose models | `final_sociometrics_data.json` |

Pipeline outputs use `output_dir`. Calibration is separate camera/scene preparation,
not an output of stitching; prepare it before subgroup analysis.

Main JSON keys are **zero-based** `frame_00000`, etc.; bbox is pixel `[x, y, width, height]`.
Adapters convert to corners internally. Detections carry confidence; tracks carry person `id`
and bbox, but the current tracking export drops confidence. Stitching rewrites IDs offline.
Do not mix tracks, pose records or caches across videos/runs or ID mappings.

Subgroup analysis processes every third source frame and excludes boxes with width < 15 or
height < 30 pixels. Group continuation uses one-to-one member overlap matching: only one
eligible child/parent retains an old ID during a split/merge, with no missing-frame group
persistence. Missing pose can reuse the person's stored smoothed direction; unavailable
orientation makes pairwise grouping rely on proximity. For variable-frame-rate video,
retain source timestamps rather than inferring elapsed time from FPS.

## Dependencies and paths

Use a Python environment compatible with your server PyTorch/CUDA installation.
`requirements.txt` is a dependency **inventory**, not a validated version lock. Main stages
use Ultralytics, BoxMOT, OpenCV and NumPy; subgroup analysis additionally uses Transformers
ViTPose, PyTorch and Roboflow Inference. Record actual server versions before comparisons.
`legacy/requirements-original.txt` preserves the old requirements, not a new environment lock.
Optional branches need their external repositories/checkpoints (DeepSORT, HRNet, MixSTE, etc.).
Public-MOT evaluation needs TrackEval separately; nothing is installed automatically here.

Copy `configs/paths.example.json` to `configs/paths.local.json` and configure:

- `video`: actual input. The inherited March 26 filename may not exist on your machine.
- `output_dir`: a separate directory per video/run, avoiding stale mixed outputs.
- `detector_checkpoint`: intended checkpoint; inherited model name is `yolo26x.pt`.
- `calibration`: calibration for this camera, not simply the first JSON available.
- `tracker_config`: `null` retains installed BoxMOT ByteTrack YAML selection; an explicit
  path overrides its location. Existing attribute overrides in tracker code still apply.
- `models_dir` and `external_dir`: asset roots for auxiliary/experimental scripts.

Relative paths resolve from the repository root. Absolute paths are supported in local config.
`PLAYGROUND_CONFIG` selects another config; explicit `--config` takes precedence for main
stages. These settings change paths, not thresholds. Older experiments retain individual
settings; inspect their constants before running.

The existing gender-classification component reads `ROBOFLOW_API_KEY` from the environment.
`.env.example` documents variables; `.env` is **not automatically loaded**. Never commit
credentials. ViTPose retains its Hugging Face model identifier and may need downloaded/cache
assets. ViTPose initialization/prediction failures yield missing poses; gender failures yield
`Unknown`. A missing calibration file yields pixel-space grouping, without metric distances;
a malformed existing calibration JSON raises an error.

### Calibration for a camera/scene

[tools/calibration_headless.py](tools/calibration_headless.py) has **no CLI arguments**.
Set `IMG_PATH`, `MY_POINTS`, `REAL_W` and `REAL_H` in its `__main__` block. Extract a frame
from the same camera at analysis resolution, without resizing/cropping, for example:

```bash
ffmpeg -i videos/YOUR_VIDEO.mp4 -frames:v 1 assets/reference/Real_playground.png
```

This requires FFmpeg and an existing parent directory. Use a new filename if a reference
must be preserved, then set `IMG_PATH` to it. Supply four `(x, y)` pixel coordinates in
`MY_POINTS` corresponding to **top-left, top-right, bottom-right, bottom-left** of a ground
rectangle. Perspective can make this differ from simply sorting image coordinates.
`compute_homography` pairs them, in order, with:

```text
(0, 0), (REAL_W, 0), (REAL_W, REAL_H), (0, REAL_H)
```

Set width/height from measured or known dimensions of that ground rectangle, in the same
unit. Use meters for meter coordinates; four image points alone do not establish metric
scale. Current constants are inherited examples, not verified measurements for a new video.
The implementation assumes a rectangle; arbitrary ground correspondences require a different
destination definition rather than forcing that assumption.

Choose distinct, well-spread points on the **playground surface**, avoiding collinear or
nearly collinear arrangements. Roof corners/elevated objects are not ground points. Gazebo
references in the source require checking that points are ground contacts. The script checks
point count but does not validate degeneracy or physical scale. After configuring constants:

```bash
python -m tools.calibration_headless
```

By default, `save_matrix()` writes `configs/calibration/calibration_matrix_playground.json`
with the `homography_matrix` key. `validate_warp_headless()` writes
`validation_result_playground.jpg` in the working directory, at 100 display pixels per
destination unit. Defaults overwrite existing files: choose new filenames in those method
calls for another camera and create the output parent directory. Inspect the warp and point
correspondences; a rectangular warp alone does not verify metric accuracy.

Set `calibration` in `configs/paths.local.json` to the resulting JSON. This calibration
script itself does not read the config's `calibration` output setting. Reuse the matrix only
while camera geometry, image coordinates/resolution and ground plane remain unchanged;
camera movement or resizing/cropping requires reconsidering it. Calibration accuracy has
not been measured.

## Entry points

From the repository root, use the same local config for all stages:

```bash
python -m pipeline.detect_yolo --config configs/paths.local.json
python -m pipeline.track --config configs/paths.local.json
python -m pipeline.stitch_tracks --config configs/paths.local.json
```

Prepare/check calibration as above, then run subgroup analysis:

```bash
python -m pipeline.analyze_subgroups --config configs/paths.local.json
```

All support `--help`, `--video`, `--output-dir`, `--checkpoint`, `--calibration` and
`--tracker-config`; each consumes only relevant settings. These commands process the configured
video, not automatically just the annotated interval. For diagnostic range/provenance handling,
start with [diagnostics/README.md](diagnostics/README.md).

Old commands such as `python detect_yolo.py` still dispatch to canonical modules.
Module commands require the repository root as working directory; direct script/wrapper paths
can be invoked elsewhere. Main-stage `--help` does not load ML dependencies or models.

After subgroup output exists, optional reporting tools are:

```bash
python -m tools.generate_report
python -m tools.render_visuals
```

They read `configs/paths.local.json` or `PLAYGROUND_CONFIG`. Per-command overrides from a
previous stage are not persisted. Most older experiments/utilities execute on import and are
scripts, not import-safe APIs; do not import them merely to discover settings.

## Evaluation, limitations and development

Evaluate detection/tracking on the reviewed GT interval; compare raw and stitched outputs
separately. Standard MOT exports use one-based frames; conversion must preserve source mapping.
Sparse annotations are not full-sequence GT. Group metrics require an independent reference
layer; proximity predictions are not ground truth.

The [pipeline audit](diagnostics/PIPELINE_CODE_AUDIT_2026-10-06_VI.md) records confirmed issues
and hypotheses. The legacy orientation gate is non-selective for valid 2D directions; pose
arrows do not establish ground-plane heading accuracy. Proximity connectivity can bridge
groups, and stitching can join different people. Numeric group-ID correctness does not prove
social membership accuracy. Orientation alternatives remain experiments. Code alone cannot
rank the main bottleneck.

This reorganization preserves model names, thresholds and algorithms, including earlier
uncommitted correctness/diagnostic work. Future changes need comparison against a frozen
baseline/configuration on the annotated development interval.

Lightweight verification (no inference):

```bash
python -m unittest discover -s tests -v
```

Tests cover diagnostic/evaluation fixtures, grouping correctness, paths, main entry-point help
and normalized algorithm AST preservation. They do not verify GPU inference, model availability
or every BoxMOT/Transformers version. `pytest.ini` restricts discovery to `tests/`, avoiding
model-running experimental scripts.

## Sharing

Share code, configuration examples and documentation. Private videos, annotations, weights,
outputs and `.local/` recovery material are ignored; local files were not deleted.
Historical commits may still contain media and a previously hardcoded API key. Untracking
current files does not erase history. Rotate that credential before sharing Git history.
This task performs no history rewrite or publishing.
