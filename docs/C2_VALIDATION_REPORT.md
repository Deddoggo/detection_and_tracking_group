# C2 final validation report

2026-11-09 (Europe/Paris). **Executed full-run evidence**, not an extrapolation from the smoke test. Complete machine record: `c2_full_acceptance.json`. Prior records `c2_validation.json` and `c2_vitpose_plus_validation.json` remain historical evidence.

## Configuration and provenance

Interpreter: `C:/Users/trant/miniconda3/envs/paper-research/python.exe`; Python 3.13.0, PyTorch 2.15.0+cu134, CUDA 13.4, Transformers 5.22.2, RTX 3060 Laptop GPU. NumPy 2.5.4, OpenCV distribution 4.11.0.86/runtime 4.11.0, sklearn 1.9.1. Versions were checked against the start snapshot; no packages were installed/changed.

Only `usyd-community/vitpose-plus-base` was loaded via Hugging Face from_pretrained, revision `92be54d7a29e42fad47b6e2ca01dd9e685a61e0d`, standard HF cache, FP32, COCO human-pose dataset_index=0, batch size 1, cuda:0. Its config/processor/weights hashes match the reviewed smoke model. Weight SHA256: `640225e4a9dd544239f1da0ae36af865a6e76e21e5f33a12656aa7b77fa5a5fa`. No separate local-checkpoint loader or unrelated model was used.

Video SHA256: `fa1044aa1f3c9d413bdd608d39f8545c593b34457f5ecb6dc80a28beb62cb119`. XML SHA256: `67addc127c07d98c17d0c160dc03d5f4025e5926dfb4d2d786aa901af37f7f08`. Reviewed alignment: `.local/c2/alignment_20261109/alignment.reviewed.json`; current hashes/dimensions/declared ranges were checked. The export itself verified decoder PTS and 1920x1080 dimensions throughout all 1,800 decoded source frames. Scheduled indices are exactly range(0,1800,3); no fixed FPS or observation-index substitution was used.

The export began before the metadata commit, so its immutable provenance records the parent HEAD plus exact executed source hashes. Those hashes were checked against the committed implementation; this detail is explicit in machine evidence rather than silently rewriting the observation manifest.

## Person and frame acceptance

| Check | Expected | Observed |
|---|---|---|
| Scheduled records | 600, source 0 through 1797 by 3 | PASS, exact sequence and source PTS |
| Input XML records | Preserve every explicit sampled person, including exclusions | 23,734 |
| Eligible / successful / missing eligible | Complete actual inference, with strict failure handling | 23,426 / 23,426 / 0 |
| Exclusions | Original policy, retained records and reasons | 308: 296 tiny_bbox, 12 outside |
| Occluded reference observations | Preserve original flags | 5,316 |
| Source person/box/reference association | Exact equality to XML | PASS, all 23,734 records |
| Extra metadata | Preserve z_order, source/label/track ID strings | PASS; optional extension, not a schema migration |
| Raw output | 17x3 finite COCO17 xy/score per eligible person | PASS, all 23,426 outputs |
| Model expert/input/heatmap | dataset [0], [1,3,256,192], [1,17,64,48] | PASS for all 23,426 logged calls |
| Raw JSON round-trip | No numeric/reference value changes | PASS on every frame |
| Model/cached artifacts | Pinned identity and checksums | PASS; matches reviewed smoke model hashes |
| Completeness/hashes | Complete state only after all expected rows | PASS; partial reader rejection also captured during export |
| Existing smoke preservation | Read-only evidence | PASS; all file hashes unchanged; prefix keypoint/score difference = 0 |

Original social_group values are unchanged: unreviewed 22,077; G1 28; G2 1,200; G3 211; G4 102; G5 4; G6 34; G7 8; G8 14; G9 56. These are unfinished reference attributes, not a complete positive/negative subgroup ground truth. No Setti F1 or subgroup accuracy was computed.

Scores are raw model outputs, not calibrated probabilities. All joint scores are retained. Pose failures are not converted to plausible missing data: the strict predictor aborts and an interrupted export remains incomplete. The existing HF postprocessor bbox metadata caveat remains documented; source XML ROIs and explicit batch-size-1 call linkage determine person association.

## Performance and resources — single measured full run

| Measurement | Observed |
|---|---:|
| Cached model initialization, including artifact hashing | 10.264 s |
| Whole subprocess wall time, including imports/startup/finalization | 996.155 s |
| Decode/inference/raw export stage | 977.403 s |
| Video decoding within the stage | 9.193 s |
| Observation construction/schema/write calls within the stage | 2.365 s |
| Total synchronized model-forward time | 677.355 s |
| Mean / median forward per person | 28.915 / 26.966 ms |
| Mean / median full person call (preprocess+forward+postprocess) | 41.148 / 38.898 ms |
| Cumulative preprocessing / postprocessing | 212.339 / 74.228 s |
| Effective persons/s, processing stage / including startup | 23.968 / 23.516 |
| Effective source frames/s, stage / including startup | 1.842 / 1.807 |
| Effective sampled frames/s, stage / including startup | 0.614 / 0.602 |
| Peak CUDA allocated | 518,845,440 bytes (494.81 MiB) |
| Peak CUDA reserved | 589,299,712 bytes (562.00 MiB) |
| Total cache file lengths | 55,844,896 bytes (53.26 MiB), excluding filesystem metadata |
| Full CPU replay A / B, including loading/validation/output | 9.895 / 9.891 s |

Forward execution is the largest measured component; preprocessing is also material. Decode and record-writing time are small relative to forward time in this run. No optimization/retuning was attempted. Times include dispatch/synchronization and were measured during an audit with lightweight concurrent diagnostics; this single run is not a robust performance benchmark. PyTorch allocator peaks exclude driver/context and other-process memory. There was no OOM and no reason to request Vast.ai.

## Replay validation

Both complete runs use the same recorded recipe: image-space-diagnostic, eps=50 pixels, close_distance=10 pixels, B0-original predicate, alpha=0.7, confidence threshold=0.6, min_samples=2, overlap threshold=0.5, state reset at cache start. Values are diagnostics, not optimized social-group parameters.

- Runs: `.local/c2/full_acceptance_20261109/replay_a/` and `replay_b/`.
- Prediction SHA256 (both): `49235da70d967a40b2f18ddb7ad8256e0de307eacf6ae7c16cea1d85bcabda6f`.
- Every source frame, PTS, person ID, bbox and eligibility/exclusion state checked; complete byte equality also verifies group IDs, heading values and heading freshness/missingness agree.
- Subprocess import guards blocked torch, transformers, ultralytics, boxmot, inference, huggingface_hub and timm throughout each run. No neural model construction/import/download occurred.
- All original cache file hashes were identical before/after replay. unreviewed remains only an unknown reference attribute, never a negative membership input.

## Existing H: separate validation levels

Actual exported file: `calibration_matrix_playground.json`, nested homography_matrix key, finite 3x3 H, rank 3, determinant -0.0004466068444. Exact requested filename homography_matrix.json was not found in the searched roots; a matrix does exist and was not regenerated or replaced.

| Level | Result | Evidence |
|---|---|---|
| 1 Numerical | PASS | Nonsingular/finite; 23,722 visible sampled bbox footpoints project finitely; maximum error reproducing source-declared correspondences 8.53e-8 destination units. |
| 2 Source-video compatibility | PASS with limited provenance | Referenced image and target video both 1920x1080; source-code points reproduce H; static top-of-image registration against reviewed frame 0 yields 1,223 inliers, near-identity affine and median displacement 0 pixels. Original JSON has no explicit video-hash binding; the additive manifest labels its derived binding. |
| 3 Physical metric validity | BLOCKED | Draft section 3.4 declares directly measured 10.42x11.54 m dimensions, but the four physical landmark identities, measurement confirmation/record and accepted physical tolerance/verifier are not established. |

The draft statement is recorded with its file hash in calibration_validation.json; it is not silently promoted to independently verified scale. `.local/c2/full_acceptance_20261109/calibration.existing-H.v1.json` preserves H numerically, references original files/hashes and declares verified=false. The metric contract rejected it with 'unverified metric calibration'. No metric grouping replay was labeled successful. Image registration above was a source-compatibility diagnostic, not a new ground calibration.

## Regressions and hygiene after full replay

- C1: **56 PASS / 1 optional Excel SKIP** (openpyxl absent), 57 methods. Native sklearn chain/split tests executed; explicitly substituted C1 cases remain labeled in their receipts.
- C2: **31 PASS**, including optional old-cache compatibility, z_order preservation, raw schema/round-trip/missingness, geometry rejection, neural-free replay and model/expert contracts.
- Full artifact acceptance command: PASS on all 600 real records and all 23,426 actual pose calls.
- Actual workspace layout: **2 PASS / 1 FAIL**, solely the preserved untracked draw_grid.py outside the unchanged root allowlist.
- Isolated git-archive snapshot of the tracked implementation: **3 PASS** under the exact same layout test. It excludes user-owned untracked/uncommitted work and does not repair/delete that work.
- User inputs/images/matrices/scripts, source video/XML, old smoke artifacts and package versions were verified unchanged. Master/archive refs were not moved; no merge was run.

See C2_HANDOFF_TO_C3.md for exact commands and remaining inputs. Complete logs/receipts are under `.local/c2/full_acceptance_20261109/`; no prior run or report artifact was overwritten. Existing primary reports were updated as requested.
