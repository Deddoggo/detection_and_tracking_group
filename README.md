# Active monocular-video subgroup pipeline

This repository keeps the current research pipeline at the repository root. The
clean `master` preserves the original implementation; correctness repairs belong
to `fix/c1-active-pipeline-only`. Cleanup alone does not validate social grouping,
metric calibration or model accuracy.

## Pipeline and local interfaces

| Stage | Source | Input / output |
|---|---|---|
| Person detection | `detect_yolo.py` | Video -> person xywh/confidence JSON; requests `yolo26x.pt` |
| Tracking | `track.py` | Detection JSON + video -> ByteTrack ID/xywh JSON |
| Optional offline stitching | `stitch_tracks.py` | Tracks + video -> remapped track JSON |
| Pose and subgroup analysis | `analyze_subgroups.py` | Tracks + video + calibration -> sampled sociometrics JSON |
| Active grouping | `identify_group_social.py` | Footpoints, headings, IDs -> groups and person records |
| Heading heuristic | `orientation_utils.py` | COCO keypoints -> image-derived direction |
| Calibration utility | `calibration_headless.py` | Reference image/correspondences/dimensions -> homography |
| Visualization | `render_visuals.py` | Sociometrics JSON + original video -> overlay video |
| Reporting | `generate_report.py` | Sociometrics -> descriptive tables |

The analyzer directly imports the active grouping and orientation modules.
Tracking is `track.py`; retired DeepSORT/DeepOCSORT/MixSTE experiments are not
runtime dependencies. C1 additionally uses `group_identity.py`,
`orientation_gate.py`, `grouping_contracts.py` and `report_metrics.py`.

The staged JSON convention is zero-based `frame_00000`. Detection records contain
`bbox=[x,y,w,h]` and `conf`; track records contain `id` and `bbox`. Analyzer records
contain `id_p`, `id_g`, `bbox`, `gender`, `real_pos`, sampled every third source
frame by the original configuration. Predicted groups are pre-annotations, not GT.

## Dependencies, configuration and execution

`requirements-active.in` inventories direct active runtime dependencies; it is
**not a resolved environment lock**. The historical `requirements.txt` is retained
unchanged for provenance and includes old dependencies unrelated to this active
path. Compatible BoxMOT/Transformers/model versions still need recovery before
claiming reproducibility. No dependencies, checkpoints or datasets are bundled or
automatically installed by the structural/regression checks.

Models requested by existing code include YOLO26x, HF
`usyd-community/vitpose-base-simple`, and an auxiliary Roboflow attribute model.
ByteTrack config is resolved from the installed BoxMOT package. Old local
BoostTrack/DeepOCSORT YAMLs are not used by this active implementation and remain
in the archive. This cleanup does not select a new detector/tracker combination.

Run from the repository root. The staged scripts currently use hardcoded paths,
not a shared CLI/config abstraction. Inspect constants before any inference:

- Video: `videos/WIN_20260326_10_00_11_Pro.mp4` (must be supplied locally).
- Stage files: `playground_results/detections.json`,
  `tracks_bytetrack_only.json`, optional `tracks_stitched.json`,
  `final_sociometrics_data.json`.
- Analyzer calibration: `calibration_matrix_playground.json`.
- Calibration utility reference: `Real_playground.png`; alternative reference
  and five calibration JSONs are retained. Their physical scale is not certified.

The normal processing order is detection -> tracking -> optional stitching ->
analysis -> visualization/report. The analyzer currently prefers a stitched file
if present. Most stage scripts execute work on import and can overwrite their
configured output paths. Use explicitly isolated inputs/outputs for an authorized
future inference run; do not import them as smoke tests. No inference is needed
for the checks below.

On clean master, reporting retains the original fixed-30-FPS behavior and other
known B0 limitations. C1 provides an explicit timebase and versioned report schema;
do not reinterpret old outputs under corrected semantics.

## Lightweight verification

```powershell
python -B tests/check_active_layout.py
# On the corrected C1 branch only:
python -B tests/run_c1.py
```

Layout checks compile source without executing imports/models. C1 tests distinguish
native dependencies from explicitly substituted clustering and report skips.
The C1 report CLI requires a new output path, source timing, stride and units; see
`docs/C1_VALIDATION.md` on that branch.

## History and local data

Original master: `2db7aac2fca6f03e1a974b1c46a921ab54330160`, preserved on
`archive/pre-cleanup-master`. Earlier C1 tips have separate archive refs. The user's
`chore/repository-layout` branch remains independent; its directory migration and
candidate integrations were not adopted as clean master.

Historical videos, pose caches, workbook and generated images were untracked,
retained locally and ignored. Their original tracked content remains in archive
history. User video/annotation directories and the untracked audit were preserved.
Prior W1/C1 reports remain on their original/archive refs rather than copied into
the production tree. See [repository cleanup and branch history](docs/REPOSITORY_CLEANUP.md).
