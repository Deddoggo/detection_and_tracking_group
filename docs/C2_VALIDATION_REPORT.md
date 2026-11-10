# C2 closure validation — 10 November 2026

**C2 closure passes on local master**, implementation `9469fbb03fcd8947830e114aa633af8a366a2552`. Current evidence is `closure` in `docs/c2_full_acceptance.json`; the existing Nov09 full-export data/receipts remain historical and unchanged. No ViTPose inference or observation-cache generation was rerun in this closure.

## Post-integration tests and artifact integrity

Using `C:/Users/trant/miniconda3/envs/paper-research/python.exe` explicitly, both before and after integration:

| Suite/check | Actual result |
|---|---|
| C1 | 56 PASS, 1 optional Excel SKIP (openpyxl absent), 57 methods |
| C2 | 35 PASS; includes user-confirmed scale, identity/hash/unit rejection, no unsupported accuracy claims, precision fit and bounded replay selection |
| Workspace layout | 3 PASS under the unchanged source-boundary rules |
| Full accepted cache validation | PASS: all 600 scheduled source frames, 23,426 successful poses, 308 exclusions, no missing eligible poses; original model/source/XML/PTS provenance retained |
| Prior artifact preservation | PASS: every file captured under the accepted full-run/smoke artifact directories remains byte-identical |
| Fresh full image replay | PASS with forbidden neural imports guarded; same predictions as both accepted runs |
| Bounded metric replay | PASS before and after integration, with identical metric prediction hashes |
| Packages/user files/archive | PASS: package versions, eight user-owned content/absence states and archived original master ref unchanged |

`draw_grid.py` was already absent when this task started. The agent did not delete, move, restore, execute or stage it. Its prior layout failure is historical; current actual-workspace checks all pass without changing the allowlist.

## Calibration: confirmed scale versus independent accuracy

Original file: `calibration_matrix_playground.json`, SHA256 `cc9c644dffa43cffc647f4e02dec34e46a24003bcf7e9c2164d0def983463d90`. Original matrix values and JSON bytes are unchanged. Reference image/source-video compatibility is rechecked through unchanged hashes of the scene previously registered at 1920x1080. Source constants are recovered statically, without executing calibration generation.

| Source point, pixels | Ground point, m |
|---|---|
| (1049,595) | (0,0) |
| (1184,533) | (10.42,0) |
| (1666,554) | (10.42,11.54) |
| (1674,651) | (0,11.54) |

The owner confirmed the physical dimensions and their intended relationship to these four selected playground landmarks in the closure instruction. The additive `.local/c2/closure_20261110/calibration.user-confirmed.v1.json` stores the confirmation reference/hash and assumptions; it does not claim named survey markers or independently measured localization error.

- Numerical validity: PASS, rank 3, determinant approximately -0.0004466068444.
- Source compatibility: PASS with documented scene/hash/correspondence basis; original export had no video identity field, so the additive manifest records its derived binding explicitly.
- Scale/landmark provenance: accepted as **user-confirmed**; `verified=false` is retained.
- Independent physical accuracy: **NOT VERIFIED**. No empirical error bound or independent survey is asserted.

Observed four-correspondence residual is approximately 8.5299e-8 declared coordinate units. The numeric limit 1.8535e-6 is computed as float32 machine epsilon times the maximum target-coordinate norm, because calibration_headless.py constructs its correspondence arrays as float32. This checks equation agreement at construction precision; it is not an arbitrarily chosen physical-error tolerance, a measured generalization error, or evidence that other world points have micrometre accuracy. The actual manifest leaves correspondence_tolerance_m null. No H was regenerated.

## Executed bounded metric replay on master

```powershell
$py = 'C:/Users/trant/miniconda3/envs/paper-research/python.exe'
& $py -B analyze_subgroups.py replay `
  --cache '.local/c2/full_acceptance_20261109/observations' `
  --geometry-mode metric `
  --calibration '.local/c2/closure_20261110/calibration.user-confirmed.v1.json' `
  --eps 1.75 --close-distance 0.5 --max-frames 5 `
  --output '.local/c2/closure_20261110/metric_replay_master'
```

This already-existing output must not be reused for a new run. The full input cache remains 600 records; replay_selection explicitly records the five selected records at frames 0,3,6,9,12. A first-N limit is validated without mutating/truncating the source cache. State initializes at source frame 0 as before.

Actual results: 213 finite positions; 148 available projected headings are finite and unit length; five ineligible records retain null group predictions and their exclusions. Positions agree with a separate homogeneous-matrix calculation within float32 numerical precision (maximum observed numeric difference 9.1563e-7 declared units, not a physical projection error measurement).

Recipe: eps=1.75 m, close_distance=0.5 m, min_samples=2, B0-original predicate, alpha=0.7, confidence threshold=0.6, overlap threshold=0.5. These preserve current B0/C1 settings, not a retuning for attractive groups. Metric and image-space runs remain separate and their group IDs are not compared as equivalent thresholds.

Metric output: `.local/c2/closure_20261110/metric_replay_master/`; prediction SHA256 `572fa2321730d0cb4f37eefe10991a071d2c858071ef446d8f796b4251a8ae41`. The before-integration metric run has the same prediction SHA. Its manifest records calibration ID, file hash, H hash, units, recipe, user-confirmed evidence level, numerical-fit metadata and independent_physical_accuracy_verified=false.

Fresh post-integration image replay: `.local/c2/closure_20261110/image_replay_master/`; all 600 predictions match SHA256 `49235da70d967a40b2f18ddb7ad8256e0de307eacf6ae7c16cea1d85bcabda6f`, the accepted image diagnostic hash. The import guard blocked torch, transformers, ultralytics, boxmot, inference, huggingface_hub and timm. B0/C1 prediction behavior and accepted raw artifacts remain unchanged.

## Safe integration proof

The C2 source lineage is based on the expected C1 master. All eight unrelated user-modified/deleted paths have identical committed blobs on C1 master and the C2 tip and are disjoint from the curated C2 diff. The guard checked initial staged state, ancestry, exact user hashes/absence and status, then switched to master and ran git -c merge.autoStash=false merge --ff-only codex/c2-observation-cache. It rechecked preservation after both operations. No stash, reset, clean, forced merge, personal-file staging, file relocation, history rewrite or push occurred.

Master integrated at `9469fbb03fcd8947830e114aa633af8a366a2552`; the final closure documentation commit follows verified tests without changing executable source. Feature and archived master refs remain available. Exact commit/file list and command outputs are in `.local/c2/closure_20261110/integration.json`; pre/post tests and preservation are under that directory and embedded/referenced by the existing machine evidence.

## Historical full-export validation — 9 November 2026

The following configuration, person counts, performance and initial image-replay measurements describe the accepted Nov09 full run. They were revalidated from immutable artifacts; inference was not rerun for closure.

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
