# C2 final implementation and acceptance

2026-11-09 (Europe/Paris). Branch: `codex/c2-observation-cache`. Starting HEAD: `0e29b47c51b19841aea97ba28cb8cc5a695aeced`. Core metadata/instrumentation commit: `3cac838`. Current complete evidence: `c2_full_acceptance.json`; prior structural/smoke evidence remains unchanged.

**C2 observation-cache engineering is complete. Trusted metric grouping remains BLOCKED.** Curated tracked C2 changes are ready for master integration with the documented limitations; no merge was performed. C3/H2/H3/H4, public benchmarking, model training and Setti F1 were not started.

## Acceptance matrix

| Criterion | Status | Executed evidence |
|---|---|---|
| A Data | PASS | Existing reviewed alignment rechecked against current video/XML hashes. Exactly 600 records at source indices range(0,1800,3); original PTS/person/reference metadata retained. |
| B Pose | PASS | 23,426 real ViTPose+ CUDA poses, 0 missing eligible poses; raw COCO17/score arrays finite; strict model failures cannot become successful missing records. |
| C Cache | PASS | observations.v1 complete; all frame/payload/metrics hashes, person counts, reference values and JSON value round-trips validated. |
| D Replay | PASS | Two full guarded CPU replays have identical complete prediction content; raw cache hashes unchanged. |
| E Geometry | BLOCKED for trusted metric use | Existing H has Level 1 numerical validity and Level 2 source compatibility; Level 3 physical verification remains incomplete. Metric rejection tested. |
| F Regression | PASS under documented-layout policy | 56 C1 PASS / 1 optional Excel SKIP; 31 C2 PASS. Workspace layout: 2 PASS / 1 FAIL from preserved untracked draw_grid.py; isolated tracked snapshot: 3 PASS. |
| G Reproducibility | PASS | Pinned model/artifact hashes, source hashes, effective configuration, environment, commands and output identities recorded. No unsupported accuracy claim. |

## Full artifacts

- Cache: `.local/c2/full_acceptance_20261109/observations/`, schema `observations.v1`, state `complete`, 600 scheduled records.
- Manifest SHA256: `fda85542696711cd50567a20be216ef4d903416423cf4c0b8cec3eb4bcf67366`.
- Raw JSONL SHA256: `7a8efcbc164f6980409e8b3d57a250365152ca44188a3bf463366601a66975b6`.
- Replays: `.local/c2/full_acceptance_20261109/replay_a/` and `replay_b/`.
- Both prediction SHA256 values: `49235da70d967a40b2f18ddb7ad8256e0de307eacf6ae7c16cea1d85bcabda6f`.
- Existing five-frame smoke cache was hashed before/after and preserved. For its five frames, the full cache has exactly zero maximum coordinate and score difference.

This is an **oracle-box/oracle-identity observation path**. No YOLO, BoxMOT or ReID execution is included in its performance or validation.

## Minimal changes in this final phase

| File | Purpose |
|---|---|
| `annotation_io.py::read_cvat` | Preserve additional raw box attributes (z_order) and original track metadata; no interpolation/preprocessing changes. |
| `c2_pipeline.py::observation_frame` / `export_cache` | Carry optional extra reference metadata into observations.v1; measure decode and record-writing time separately. |
| `observation_cache.py::validate_frame` | Validate optional raw metadata while retaining compatibility with the untouched old smoke cache. |
| `tests/test_c2.py` | Add a regression for preservation of extra CVAT metadata. |
| `tests/validate_c2_full.py` | Repeatable full-artifact acceptance: all scheduled frames, exact XML/PTS/person association, finite shape/schema, model/artifact identity, metrics/calls and accounting. |
| `README.md` | Point to full acceptance and current limitations. |
| Three existing C2 Markdown reports | Replace pending-full-export status with executed full-run evidence and scoped closure. |
| `docs/c2_full_acceptance.json` | Current machine-readable acceptance, metrics, geometry/replay/test receipts and preservation manifest. |

No orientation heuristic, B0 predicate, DBSCAN method/parameters, C1 association, model architecture, detector/tracker/ReID, renderer or reporter was changed. The only raw-record extension retains existing reference metadata; the schema remains observations.v1 and old caches still validate.

## Existing H integration

The located exported matrix is `calibration_matrix_playground.json` (nested key `homography_matrix`). A distinct file named `homography_matrix.json` was not found in the repository/ignored outputs, Downloads, Desktop, Documents or configured visualization workspace. This filename discrepancy does not mean H generation is pending.

The additive manifest `.local/c2/full_acceptance_20261109/calibration.existing-H.v1.json` references and copies the existing numerical H unchanged, binds the target video with an explicit image-registration basis, retains source hashes and declared correspondences, and remains `verified=false`. The original matrix/image/script files were never changed or staged.

Level 1: PASS (rank 3, finite/nonsingular, 23,722 visible sampled footpoints project finitely). Level 2: PASS with documented scope (1920x1080 reference/video, source-code correspondences reproduce H, static-scene registration has 1,223 inliers and near-identity transform). Level 3: BLOCKED: the draft declares direct 10.42x11.54 m measurements, but point identities/measurement confirmation and accepted physical-error tolerance are not established. No metric grouping run was represented as verified.

## User work and repository boundary

Preserved pre-existing changes/deletions: `Real_playground.png`, `Real_playground_2.png`, `calibration_headless.py`, `calibration_matrix.json`, `calibration_matrix_PETS09.json`, `calibration_matrix_based_on_roof.json`, `calibration_matrix_playground.json`, `calibration_matrix_playground_2.json`; also the untracked `draw_grid.py`. All initial content hashes/absence states and video/XML/smoke hashes were verified unchanged. No package operations or archive/master ref changes occurred.

The grid script is a standalone, main-guarded visualization utility using the existing image/H; it writes a grid image and can open a GUI. It was inspected statically, never run or modified, and is not imported by the nine active scripts. The source-boundary allowlist was not weakened. Any relocation to a non-active tools/calibration directory requires separate owner authorization.
