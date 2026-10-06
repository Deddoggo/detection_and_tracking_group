# Current repository audit — 2026-10-05

Initial HEAD: `2db7aac2fca6f03e1a974b1c46a921ab54330160`, branch `master`,
initial working tree clean. Commit date 2026-06-05 is later than the thesis report.
The present checkout matches the previous audited snapshot. This establishes that
the two grouping bugs apply here, not that these scripts produced thesis results.

Read sources: user-provided `C:/Users/trant/Downloads/Advisor_Notes_and_SCIA_Plan.md`,
previous workspace's advisor notes and SCIA priorities, repository scripts/configs
and archived `eval_result.txt`. No new literature search was performed. The original
notes and archived results were not edited.

## Paths present in code, not confirmed actual commands

| Entry point | Observed implementation/config | Outputs/cache |
|---|---|---|
| `detect_yolo.py` | local/name `yolo26x.pt`; person class0; conf .15, NMS IoU .45, imgsz1280; auto CUDA/CPU | `playground_results/detections.json`, detection overlay; zero-based frame keys, integer xywh+conf |
| `track.py` | reads that cache; BoxMOT ByteTrack package YAML, device auto, half=True; input conf>.15; assigns track_thresh=.25/track_buffer=90 AFTER construction | tracks_bytetrack_only.json, xywh+id, no score |
| `stitch_tracks.py` | optional offline HSV appearance/spatial linking; MAX_TIME_LOST150, distance350px, similarity.60 | tracks_stitched.json; source video needed |
| `analyze_subgroups.py` | stitched cache preferred silently over ByteTrack cache; HF ViTPose `usyd-community/vitpose-base-simple`; `orientation_utils.get_pose_vector`; social module; homography playground; scale1, eps1.75, min2, stride3; ignores w<15/h<30 tracks | final_sociometrics_data.json; bbox/person/group/position/gender, NO raw keypoints or direction cache |
| `yolo_detection_vitpose.py` | CLI video; YOLO12x fallback26x; conf.25/imgsz1280; DeepOCSORT defaults+OSNet; pose+arrows but imports proximity `identify_group` | separate integrated PETS/playground path, not same as offline chain |
| `yolo_detection_vitpose_draw.py` | CLI video; YOLO12x fallback26x; conf.25/imgsz640; BoostTrack custom YAML+clip_market1501; social module; default scale.35/eps1.5 | alternate integrated path, not selected automatically |
| `tracking_yolov12.py`, `run_all_*` | public MOT15 chain; YOLO12x fallback yolov26x; conf.4, DeepOCSORT defaults; custom_deepocsort.yaml NOT passed | benchmark path; archives alone do not recover run/checkpoint/environment |

The first four scripts hardcode `videos/WIN_20260326_10_00_11_Pro.mp4`.
That path and the videos/ directory are absent. Which chain the user actually ran
remains unconfirmed. No raw playground input was inferred from overlay videos.
`analyze_subgroups.py` also initializes external gender classification, outside the
diagnostic question; the new pilot runner does not import/call it.

## Available assets and gaps

Present: PETS raw webm; PETS/pose/tracking output mp4 overlays; one AVI under
`runs/segment/predict/` (derived output by location; source status not confirmed);
PETS 2D/3D pickle caches; one sociometrics spreadsheet; five calibration JSONs;
BoostTrack/DeepOCSORT YAMLs; archived text log. Missing: playground detection/track
JSONs, five confirmed raw inputs, local .pt checkpoints, ByteTrack package YAML,
official TrackEval checkout, DeepSORT dependencies and the actual inference env.

Installed desktop Python has numpy, but not cv2/sklearn/scipy/torch/ultralytics/
boxmot/transformers. `requirements.txt` describes an older Detectron2/DeepSORT stack;
it does not pin the current YOLO/BoxMOT/HF chain. No environment was installed.
The inventory records file sizes and metadata-read failures explicitly where the
current environment cannot read duration/FPS/resolution. Camera motion, session and
failure timestamps cannot be inferred without inspecting confirmed raw inputs.

Calibration files only store matrices, with no camera/session/measurement provenance.
Roof corners are not confirmed ground landmarks. Code constants in calibration
scripts are not independent measured distance validation. Geometry scale remains
unverified. Pose uses shoulder normal + ear/nose proxy, weighted fusion and heuristic
vertical scaling; this is not validated ground-plane heading. History is reset for
absent IDs; stride3 analysis omits intermediate frames. Tiny-track filtering could
remove genuine distant children downstream — a hypothesis to measure, not changed now.

## Tracker features: implementation audit remains conditional

ByteTrack is selected in `track.py`; the local script does not implement association
itself. No BoxMOT package is installed here, so exact high/low association,
confirmation, reactivation, persistence and duplicate-removal behavior cannot be
verified for the user's version. The runtime attributes assigned after construction
may not update derived buffer/persistence values; no assumption is made that they
did. The new runner uses an explicit version-specific YAML and saves implementation
hash/effective scalars instead. YOLO NMS is already used. No duplicate suppression,
low-confidence recovery or new tracker was reimplemented. Inspect the user's actual
installed source/YAML before changing these policies.

## Confirmed correctness findings

1. Both grouping modules independently matched new clusters to old groups, allowing
   multiple assignments to one dictionary key. A 4-person group split into two
   2-person clusters loses one cluster in the archived code. The new one-to-one
   bookkeeping preserves all four people. Split/merge lineage is recorded separately.
2. The social gate uses abs mutual cosine>.7 OR abs facing>.3 for either person.
   For valid unit2D vectors, if both facing terms fail, abs mutual cosine>=.82>.7;
   hence it accepts all otherwise eligible pairs. Small numerical fixtures support
   this mathematical result, not performance on video. Gate retained as raw legacy
   comparator; proximity baseline and disabled-by-default signed candidate are explicit.

## No historical relabeling

Archived log, public MOT counts and thesis figures remain unchanged. New config
locks describe the current desktop environment and current code, not historical
packages/weights. Unavailable historical assets are recorded as unavailable. No
new metric, model inference, orientation accuracy, grouping improvement or research
novelty follows from this audit or correctness fix.
