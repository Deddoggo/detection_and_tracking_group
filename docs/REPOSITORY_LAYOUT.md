# Repository layout and inventory

Canonical implementations moved; root Python files are compatibility wrappers. No file is classified as obsolete.

| Previous path | Canonical path |
|---|---|
| `detect_yolo.py` | `pipeline/detect_yolo.py` |
| `track.py` | `pipeline/track.py` |
| `stitch_tracks.py` | `pipeline/stitch_tracks.py` |
| `analyze_subgroups.py` | `pipeline/analyze_subgroups.py` |
| `identify_group.py` | `modules/identify_group.py` |
| `identify_group_social.py` | `modules/identify_group_social.py` |
| `orientation_utils.py` | `modules/orientation_utils.py` |
| `orientation_gate.py` | `modules/orientation_gate.py` |
| `group_identity.py` | `modules/group_identity.py` |
| `detection.py` | `legacy/detection.py` |
| `tracking_group.py` | `legacy/tracking_group.py` |
| `yolo_detection.py` | `experiments/yolo_detection.py` |
| `yolo_detection_poseformer.py` | `experiments/yolo_detection_poseformer.py` |
| `yolo_detection_vitpose.py` | `experiments/yolo_detection_vitpose.py` |
| `yolo_detection_vitpose_draw.py` | `experiments/yolo_detection_vitpose_draw.py` |
| `run_mixste_offline.py` | `experiments/run_mixste_offline.py` |
| `test_clustering.py` | `experiments/test_clustering.py` |
| `test_combine_with_pose_estimation.py` | `experiments/test_combine_with_pose_estimation.py` |
| `test_sam3_video.py` | `experiments/test_sam3_video.py` |
| `run_all_detectron2.py` | `evaluation/run_all_detectron2.py` |
| `run_all_yolov12.py` | `evaluation/run_all_yolov12.py` |
| `tracking_yolov12.py` | `evaluation/tracking_yolov12.py` |
| `eval_tracking_group.py` | `evaluation/eval_tracking_group.py` |
| `calibration.py` | `tools/calibration.py` |
| `calibration_headless.py` | `tools/calibration_headless.py` |
| `measure_tool.py` | `tools/measure_tool.py` |
| `measure_tool_headless.py` | `tools/measure_tool_headless.py` |
| `draw_grid.py` | `tools/draw_grid.py` |
| `generate_report.py` | `tools/generate_report.py` |
| `render_visuals.py` | `tools/render_visuals.py` |
| `draw_3d_orientation.py` | `tools/draw_3d_orientation.py` |
| `render_gif.py` | `tools/render_gif.py` |
| `visualize_3d.py` | `tools/visualize_3d.py` |
| `calibration_matrix.json` | `configs/calibration/calibration_matrix.json` |
| `calibration_matrix_based_on_roof.json` | `configs/calibration/calibration_matrix_based_on_roof.json` |
| `calibration_matrix_PETS09.json` | `configs/calibration/calibration_matrix_PETS09.json` |
| `calibration_matrix_playground.json` | `configs/calibration/calibration_matrix_playground.json` |
| `calibration_matrix_playground_2.json` | `configs/calibration/calibration_matrix_playground_2.json` |
| `custom_boosttrack.yaml` | `configs/trackers/custom_boosttrack.yaml` |
| `custom_deepocsort.yaml` | `configs/trackers/custom_deepocsort.yaml` |
| `eval_result.txt` | `evaluation/archive/eval_result.txt` |
| `MixSTE_without_pretrained_weights.txt` | `docs/research/MixSTE_without_pretrained_weights.txt` |
| `requirements.txt` | `legacy/requirements-original.txt` |
| `playground.png` | `assets/reference/playground.png` |
| `Real_playground.png` | `assets/reference/Real_playground.png` |
| `Real_playground_2.png` | `assets/reference/Real_playground_2.png` |

## Retained local assets

Raw `videos/`, `PETS09-S2L1-raw.webm`, CVAT export folders, session-specific diagnostic artifacts, `playground_results/`, `output_*/`, `runs/`, grid/validation images and pose caches remain on disk. They are ignored and existing tracked media are removed from the index only. Nothing is deleted.
Calibration reference photos moved to ignored `assets/reference/`; supply these separately if needed. Calibration matrices and tracker YAMLs remain versioned.

## Roles and uncertainty

`pipeline/`: the four confirmed playground stages. `modules/`: shared grouping/orientation code, including prior correctness edits.
`experiments/`: integrated YOLO/pose variants, MixSTE, SAM3 and Detectron2/HRNet explorations. The files named test_* here are executable experiments, not lightweight unit tests. Their exact result provenance is not established.
`legacy/`: the separately documented Detectron2 + DeepSORT route; preserved, not declared obsolete. `evaluation/`: MOT batch/evaluation scripts; archive log unchanged.
`tools/`: calibration, measurement, drawing, reporting, 3D visualization. `diagnostics/`: reusable annotation/conversion/cache/evaluation tools and dated research audits; local datasets and environment locks are not distributed.
`docs/research/`: retained research notes. Dated audits refer to their original snapshot; moved-file links point to canonical implementations, but historic line numbers/config claims are not claims about the reorganized checkout.

## Preservation

Pre-existing uncommitted changes were copied before migration to ignored `.local/pre-layout-sources.zip`. It may contain old credentials; do not share it. `layout_inventory.json` records original file hashes/sizes. No Git history rewrite, push, or algorithm change is part of this migration.
The historical API credential was removed from live source and replaced by ROBOFLOW_API_KEY. It may still exist in Git history; the owner should revoke/rotate it before sharing history. No credential value is documented here.
