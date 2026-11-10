# C2 closed — local master integration

**10 November 2026, Europe/Paris. C2 is closed and its curated implementation is integrated into local master.**

Final C2 implementation commit: `9469fbb03fcd8947830e114aa633af8a366a2552`. Master fast-forwarded from `6c641d74de5269c53ea80d0069567d05b546a398` to that commit, preserving all eight C2 commits and the C1 lineage. This report's containing commit records closure documentation after the integrated implementation passed verification. No push, history rewrite or merge commit was performed. The C2 feature branch remains available at the implementation commit; `archive/pre-cleanup-master` remains unchanged.

## Before and after

| Item | Before closure | After integration |
|---|---|---|
| Checked-out branch | codex/c2-observation-cache | master |
| Implementation HEAD | 2f705c32f21195c58d72829732397550fd198382 | 9469fbb03fcd8947830e114aa633af8a366a2552 |
| Canonical master | 6c641d74de5269c53ea80d0069567d05b546a398 | Fast-forwarded to C2 implementation |
| User changes | Eight unstaged calibration/image modifications or deletions | Same contents/absence states, unstaged |
| draw_grid.py | Already absent on arrival | Not restored, deleted or relocated by this task |
| Tests | Previous accepted evidence rechecked | C1 56 PASS / 1 optional SKIP; C2 35 PASS; layout 3 PASS |

The eight user paths had identical committed blobs on both branches and were disjoint from the integration diff. The integration guard checked content hashes, clean staging, ancestry and working-tree status before/after each Git operation. No stash, personal-file commit, relocation, discard, restore, reset or clean was used. Accepted caches, replay files, calibration artifacts and source data were preserved.

## Calibration accepted at its actual evidence level

The existing `calibration_matrix_playground.json` was read unchanged. Its four code-selected image points map to the following ground rectangle:

| Image x,y (pixels) | Ground X,Y (metres) |
|---|---|
| 1049,595 | 0,0 |
| 1184,533 | 10.42,0 |
| 1666,554 | 10.42,11.54 |
| 1674,651 | 0,11.54 |

The project owner explicitly confirmed the 10.42 m width, 11.54 m height and intended relationship to these four selected landmarks on 2026-11-10. Exact point localization and independent physical projection accuracy were not confirmed or measured.

The additive manifest `.local/c2/closure_20261110/calibration.user-confirmed.v1.json` records that confirmation, source attachment hash, original matrix/image/code/video hashes and correspondence assumptions. It keeps `verified=false`, `independently_surveyed=false`, `physical_projection_error_measured=false` and no physical-error bound. Original H values and the original JSON structure/bytes remain untouched.

The minimal calibration.v1 extension permits explicit user-confirmed scale after all existing matrix/video/units/hash/correspondence checks. Numerical fit is checked against source construction precision (float32), not an invented empirical metre tolerance. The old verified-assertion path remains backward-compatible and is not relabeled as an independent survey.

## Executed closure verification

| Item | Status |
|---|---|
| Numerical H validity / current source compatibility | PASS: finite, rank 3, source/ground correspondences and unchanged registered 1920x1080 scene verified |
| Measurement provenance | USER-CONFIRMED: dimensions and intended landmark relationship recorded |
| Independent physical accuracy | NOT VERIFIED; no survey/error certificate or physical-error tolerance fabricated |
| Bounded metric replay | PASS on source frames 0,3,6,9,12, using eps=1.75 m and close_distance=0.5 m |
| Projected outputs | 213 finite positions, 148 available finite unit-length headings, 5 exclusions retained |
| Existing full cache | PASS: 600 samples; 23,426 successful poses; zero missing eligible; original hashes unchanged |
| Image replay determinism | PASS: fresh full replays before/after integration match both accepted prediction hashes |
| C1/C2/layout on master | PASS, with only the documented optional Excel skip |
| Local integration | PASS: fast-forward; feature/archive refs retained; user work preserved |

C2 engineering is complete and ready for separately authorized C3. User-confirmed metric replay success is not independently demonstrated metric accuracy or social-group accuracy. No C3, W3/W4 research, H2/H3/H4, public benchmark or Setti F1 computation was begun.

## Narrow closure changes

- `grouping_contracts.py`: explicit user-confirmed provenance, precision-derived numerical fit and truthful accuracy metadata; matrix/video/units safety remains enforced.
- `c2_pipeline.py`: optional first-N replay selection and calibration-file hash; default full replay prediction behavior unchanged.
- `config/calibration.v1.example.json`: represent unverified/user-confirmed evidence without claiming a survey.
- `tests/test_c2.py`: positive/negative provenance guards and bounded-selection/immutability regressions.
- The three existing C2 reports and the `closure` section of existing `docs/c2_full_acceptance.json`: actual closure/integration evidence. Original Nov09 export evidence remains historical and its immutable artifacts remain unchanged.

Integration receipt and exact commit/file list: `.local/c2/closure_20261110/integration.json`. Existing original user changes were excluded: Real_playground.png, Real_playground_2.png, calibration_headless.py, calibration_matrix.json, calibration_matrix_PETS09.json, calibration_matrix_based_on_roof.json, calibration_matrix_playground.json and calibration_matrix_playground_2.json.
