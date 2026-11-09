# Active monocular-video subgroup pipeline

The canonical master integrates the clean active pipeline and selected C1 correctness repairs. C2 is implemented on `codex/c2-observation-cache`; real pose-cache validation and trusted metric calibration remain pending. Predictions are pre-annotations, not subgroup ground truth.

## Active path

`detect_yolo.py` -> `track.py` -> optional `stitch_tracks.py` -> `analyze_subgroups.py` -> `identify_group_social.py` / `orientation_utils.py` -> legacy `render_visuals.py` / `generate_report.py`.

The active detector already requests `yolo26x.pt`. C2 does not switch models or tune tracking/grouping. Active helpers are `group_identity.py`, `orientation_gate.py`, `grouping_contracts.py`, `report_metrics.py`, `annotation_io.py`, `observation_cache.py` and `c2_pipeline.py`. Legacy/MixSTE implementations are outside the active boundary.

## C2 commands and environment

Use the existing `paper-research` environment explicitly; never assume base:

```powershell
$py = 'C:\Users\trant\miniconda3\envs\paper-research\python.exe'
& $py -B tests/run_c1.py
& $py -B tests/run_c2.py
& $py -B tests/check_active_layout.py
& $py -B analyze_subgroups.py validate-data --help
& $py -B analyze_subgroups.py cache --help
& $py -B analyze_subgroups.py replay --help
```

C2 uses explicit XML oracle boxes/IDs, original pixel coordinates, zero-based source indices and rational PTS. The intended range is `[0,1800)`, approximately 65.2 seconds in the restored VFR video. It is not literal 60 seconds. Partial CVAT `G1`–`G9` and `unreviewed` values are retained unchanged; `unreviewed` is unknown.

`cache` writes a new `observations.v1` JSON/JSONL directory, retains raw COCO17 keypoints/scores before smoothing/grouping, disables auxiliary gender inference, and requires identified local HF ViTPose files. No checkpoints are downloaded by this path. Missing/excluded observations remain explicit. `replay` requires no neural imports and writes separate predictions with configuration, geometry units and provenance. Its JSONL format is separate from the legacy renderer/report JSON interface.

Metric grouping requires verified `calibration.v1` evidence matching the source video/dimensions/H. Missing calibration cannot silently become pixels. `image-space-diagnostic` requires explicit pixel thresholds. Historical behavior is available only through `legacy-b0`; the live script requires `--compatibility-b0 --tracks EXPLICIT_PATH`. Existing live video/output constants remain hardcoded: inspect them before execution; historical live outputs can be overwritten by that compatibility path.

`requirements-active.in` is an inventory, not an environment lock. `requirements.txt` is historical and must not be installed blindly. No dependency changes were made for C2. Full environment/model provenance and actual validation are recorded below.

## Milestone evidence and next prerequisites

- [C2 implementation and complete change inventory](docs/C2_IMPLEMENTATION_SUMMARY.md)
- [C2 executed validation and reproduction commands](docs/C2_VALIDATION_REPORT.md)
- [Pending checkpoint/calibration inputs and bounded smoke command](docs/C2_HANDOFF_TO_C3.md)
- Machine evidence: `docs/c2_validation.json`; local alignment images and full PTS: `.local/c2/`.

Original broad master remains under `archive/pre-cleanup-master`. User datasets, annotations, checkpoints and existing calibration edits are preserved. C2 does not establish model accuracy, complete subgroup GT or benchmark readiness; C3/H2/H3/H4 have not begun.
