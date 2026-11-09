# C2 implementation summary

2026-11-09. Branch: `codex/c2-observation-cache`. Implementation commits: `ed28f32` (core), `45b1d2d` (replay lineage).
Model clarification implementation commit: `650f722`. Current model clarification: only `usyd-community/vitpose-plus-base` is supported. See `c2_vitpose_plus_validation.json` for the executed CUDA smoke and regressions; the initial implementation commits above predate this clarification.

Starting master: `6c641d74de5269c53ea80d0069567d05b546a398`.

## Actual completion

| Phase | Status | Evidence / remaining boundary |
|---|---|---|
| C2.1 XML/video alignment | CONFIRMED for `[0,1800)` | All 1,800 frames decoded sequentially; every decoded timestamp matched packet PTS. XML coverage/flags/partial groups tested. Three sampled overlays visually inspected. This is not exhaustive annotation validation. |
| C2.2 geometry contracts | Implemented and synthetically tested | Metric mode rejects absent/unverified/mismatched provenance. Actual physical calibration remains pending user measurements. |
| C2.3 observations.v1 | CONFIRMED bounded CUDA export | Raw keypoints/scores, boxes, IDs, PTS and missingness round-trip. Real CUDA export CONFIRMED on frames 0,3,6,9,12: 213 eligible poses, 5 explicit tiny-box exclusions; observations.v1 saved. |
| C2.4 grouping-only replay | CONFIRMED controlled and real-cache replay | Native sklearn; replay imports no neural modules. Retained live analyzer EMA/groups agree with replay on controlled test poses. The real five-frame pose cache also replayed twice with identical predictions in explicit image-space diagnostic mode. |
| C2.5 validation | Bounded CUDA smoke and regressions complete | C2: 30 PASS. C1: 56 PASS, one optional Excel SKIP. Layout: 2 PASS / 1 FAIL because pre-existing untracked draw_grid.py lies outside the allowlist; it was preserved. |

Bounded model inference succeeded. No subgroup/pose accuracy, benchmark gain, complete subgroup GT or full-video pose export is claimed.

## Architecture and contracts

`analyze_subgroups.py::main` dispatches `validate-data`, `cache` and `replay`.
`c2_pipeline.py::observation_frame` (line 55) is the raw export boundary, before heading extraction, EMA, projection and grouping.

- `annotation_io.py::read_cvat` (line 18) reads explicit CVAT 1.1 person boxes. It does not interpolate absent/outside intervals. Attributes, including `G1`-`G9` and `unreviewed`, are retained verbatim.
- `annotation_io.py::probe_video` / `verify_decode` (lines 71 / 97) retain rational source PTS. Reordered/B-frame streams are rejected rather than given a speculative packet/frame map.
- `observation_cache.py::validate_manifest`, `validate_frame`, `CacheWriter`, `read_cache` define `observations.v1`. JSON manifest + scheduled-frame JSONL; original-resolution pixel xywh and COCO17 xy/score; new run directory only; incomplete-state protection, payload hashes, ordering/count validation.
- Raw scores are finite heatmap scores, not probabilities. They are neither clipped nor converted to calibrated uncertainty. XML boxes have null detection confidence.
- Outside and tiny persons remain in cache records, with exclusion reasons. Eligible persons with missing poses remain explicit. Unsampled frames are represented by the manifest's source range/stride, not fabricated pose rows.
- Raw keypoints are authoritative. Raw image heading, smoothed heading, stale age, projected heading and positions are derived during replay with a recorded recipe. Changing these derivations does not require pose inference.
- `grouping_contracts.py::geometry_contract` (line 63) separates `metric`, `image-space-diagnostic` and `legacy-b0`. Metric metadata must match video hash/dimensions and H hash, name units/verifier/measurements, and reproduce supplied correspondences within an explicit tolerance. This validates assertions and algebra, not the truth of physical measurements.
- `grouping_contracts.py::project_geometry` (line 114) shares the existing float32 footpoint/10-pixel endpoint projection. B0 predicate, DBSCAN and C1 association are preserved. Image diagnostics explicitly use image headings and pixel thresholds.
- `c2_pipeline.py::GroupingReplay` resets state at cache start and requires consecutive scheduled source frames. It preserves the live EMA, stale-heading fallback, disappearance reset and C1 ID allocation. Predicted group `-1` means clustering noise; excluded persons have null predictions. Neither is derived from `unreviewed` references.
- `cache` requires explicit video/XML/alignment/output; optional `--revision` accepts only an exact HF commit SHA. Gender inference is disabled. Model and processor load directly by the fixed HF model ID, pinned to resolved commit `92be54d7a29e42fad47b6e2ca01dd9e685a61e0d` in the executed smoke. Hub-cached artifacts are hashed for provenance; no local-checkpoint selector exists. Failed inference stops export and leaves partial cache incomplete.
- `replay` reads and validates the entire cache before stateful processing. It writes a separate immutable run containing predictions and provenance. No H2/H3/H4 was implemented.

## Changed files and purpose

| File | Purpose |
|---|---|
| `analyze_subgroups.py` | C2 command dispatch, lazy neural imports, fixed-model pinned HF cache predictor with explicit COCO expert and timing/shape audits; explicit live tracks/B0 compatibility. |
| `annotation_io.py` (new) | CVAT parsing, source PTS, bounded decode verification and review overlays. |
| `observation_cache.py` (new) | Versioned raw schema, exclusive writer and validated reader. |
| `c2_pipeline.py` (new) | Minimal command orchestration, observation boundary, Hub model provenance, performance capture and replay state. |
| `grouping_contracts.py` | Geometry/unit/provenance contracts and shared original projection. |
| `identify_group_social.py` | Enforce geometry contracts and use shared projection; preserve clustering/association. |
| `orientation_gate.py` | Explicit close-distance units; original default 0.5 and B0 comparisons preserved. |
| `tests/c1_cases.py` | Explicit B0 compatibility in historical cases, immutable scope baseline, preload native OpenCV before module substitution. |
| `tests/check_active_layout.py` | Allow the three directly used C2 helpers. |
| `tests/test_c2.py` (new) | 30 tests, including real XML/cache replay, native postprocessing, fixed Hub loading and explicit COCO dataset-index dispatch. |
| `tests/run_c2.py` (new) | paper-research-only runner and timestamped machine evidence. |
| `config/calibration.v1.example.json` (new) | Explicitly unverified template; no fabricated matrix or measurements. |
| `README.md` | Correct integrated C1 state and current C2 commands/environment. |
| `docs/C2_IMPLEMENTATION_SUMMARY.md` | This milestone/change inventory. |
| `docs/C2_VALIDATION_REPORT.md` | Executed evidence and reproduction commands. |
| `docs/C2_HANDOFF_TO_C3.md` | Remaining prerequisites and scope boundary. |
| `docs/c2_validation.json` | Preserved prior structural evidence. |
| `docs/c2_vitpose_plus_validation.json` | Current exact model revision, CUDA smoke, timings/memory, replay and regression evidence. |

The original orientation heuristic and group-ID helper are unchanged. Detection, tracking, optional stitching, visualization and reporting implementations were not changed. C2 JSONL is not directly consumed by the old visualization/report scripts; their legacy JSON interface remains separate.

## Preserved pre-existing work

Master was not clean on arrival. These pre-existing changes/deletions were carried onto the feature branch, never staged or committed by C2:

`Real_playground.png`, `Real_playground_2.png`, `calibration_headless.py`, `calibration_matrix.json`, `calibration_matrix_PETS09.json`, `calibration_matrix_based_on_roof.json`, `calibration_matrix_playground.json`, `calibration_matrix_playground_2.json`.

The existing data and user calibration edits were not overwritten or restored. The original master/archive refs were not moved. Generated alignment/evidence files are isolated under ignored `.local/c2/`; no historical outputs were overwritten.

## ViTPose+ clarification changes

This follow-up modifies only `analyze_subgroups.py`, `c2_pipeline.py`, `tests/test_c2.py`, README and the three existing C2 documents, and adds `docs/c2_vitpose_plus_validation.json`. All homographies and existing user edits remain untouched. The additional pre-existing untracked `draw_grid.py` was neither executed nor modified; it explains the current root-layout check failure. No Conda packages changed. The full 600-sample export and Vast/H2/H3/H4/C3 were not started.

The explicitly requested pose-model change is identified by new observation provenance; comparisons against earlier model outputs require controlled evaluation. Grouping rules and C1 identity behavior remain unchanged.
