# C2 validation report — ViTPose+ Base clarification

2026-11-09. Current evidence: `c2_vitpose_plus_validation.json`. Prior C2 structural evidence remains in `c2_validation.json`; it predates this model clarification.

## Environment and single model

All execution used `C:\Users\trant\miniconda3\envs\paper-research\python.exe`, Python 3.13.0. PyTorch 2.15.0+cu134, CUDA 13.4, Transformers 5.22.2; local RTX 3060 Laptop GPU. NumPy 2.5.4, OpenCV distribution 4.11.0.86/runtime 4.11.0, sklearn 1.9.1. No package installation, downgrade or upgrade was performed.

Only `usyd-community/vitpose-plus-base` is supported. Model and processor load directly using `from_pretrained(model_id, revision=resolved_sha)`, allowing standard Hugging Face automatic download/cache. There is no local-checkpoint mode or alternative model selector.

Executed immutable revision: **`92be54d7a29e42fad47b6e2ca01dd9e685a61e0d`**. The first resolution used Hub `main`; both loads are pinned to its full SHA. The final run explicitly reused this SHA. Manifest artifacts are SHA256-hashed: config, processor config and weights. The cached weights hash is `640225e4a9dd544239f1da0ae36af865a6e76e21e5f33a12656aa7b77fa5a5fa`.

The installed model signature/backbone explicitly support `dataset_index`. Every real call used a CUDA int64 tensor `[0]`, selecting the COCO human-pose expert. The six-expert/192-part-feature configuration, COCO17 label names and returned label order 0–16 were checked. This matches the [official HF ViTPose+ dataset-index documentation](https://huggingface.co/docs/transformers/model_doc/vitpose#notes) and [pinned model configuration](https://huggingface.co/usyd-community/vitpose-plus-base/blob/92be54d7a29e42fad47b6e2ca01dd9e685a61e0d/config.json). Successful native CUDA forwards establish compatibility for this bounded case.

## Executed CUDA smoke

Raw cache: `.local/c2/vitpose_plus_20261109/observations_verified/`.

| Source frame | PTS seconds | XML records retained | Pose arrays | Explicit tiny-box exclusions | Missing eligible poses |
|---|---:|---:|---:|---:|---:|
| 0 | 0.000000000 | 43 | 42 | 1 | 0 |
| 3 | 0.112066667 | 43 | 42 | 1 | 0 |
| 6 | 0.239966667 | 44 | 43 | 1 | 0 |
| 9 | 0.368000000 | 44 | 43 | 1 | 0 |
| 12 | 0.464166667 | 44 | 43 | 1 | 0 |
| Total | | 218 | 213 | 5 | 0 |

Each of the 213 calls used batch size 1, `cuda:0`, FP32, input `[1,3,256,192]`, heatmaps `[1,17,64,48]` and final xy/score array `[17,3]`. Arrays were finite. Each single-person input was associated with exactly one output, and all source IDs, original float boxes, reference flags/attributes and source PTS were checked against the XML/alignment. No detection/tracking inference was used.

Native postprocessing tests confirm that corresponding crops preserve their distinct source-pixel offsets and order. Pose overlays at frames 0 and 12 were visually inspected: confident visible joints broadly follow the intended people. This is not exhaustive identity or keypoint accuracy validation. The HF postprocessor's returned `bbox` field is not used for assignment: in installed Transformers 5.22.2 it is assembled from crop center/scale metadata rather than the original pixel ROI. Original XML boxes remain authoritative and unchanged.

Raw scores ranged 0.0168633–0.9917663; **1,955 of 3,621 scores were below 0.6**. All scores/joints were retained. Low confidence is not silently treated as a missing person or calibrated uncertainty. The five real exclusions exercise the existing tiny-box policy; separate regressions cover absent poses, empty frames and outside intervals. Gender inference and ground-plane projection were not run in raw cache mode.

### Timing and memory

| Measurement | Observed |
|---|---:|
| Final cache-hit model initialization, including artifact hashing | 12.654 s |
| Sequential decode + 213 pose calls + raw export preparation | 12.059 s |
| Sum of synchronized forward wall times | 7.926 s |
| Mean / median synchronized forward per person | 37.212 / 34.405 ms |
| Mean preprocessing + forward + postprocessing per person | 54.722 ms |
| First complete person call, no warm-up discarded | 345.323 ms |
| Peak model-load allocated bytes | 501,608,960 |
| Peak inference allocated bytes | 518,845,440 (494.81 MiB) |
| Peak inference reserved bytes | 589,299,712 (562.00 MiB) |

Times are one bounded run, not a benchmark or throughput claim. Forward times include dispatch/synchronization. Memory is the PyTorch allocator, excluding driver/context/other-process use. Initial download duration was not measured; the successful final run used HF-cached weights. No OOM occurred at batch size 1. Full per-call measurements and person/crop linkage are in `inference_metrics.json`, hashed by the observation manifest.

### Preserved failure evidence

The first initialization attempt downloaded/loaded the model but stopped at an audit check that relied on the removed private config `_commit_hash`. The code now pins the public loading revision and hashes the pinned cached artifacts. A second attempt stopped at an incorrect assumption that the processor's returned bbox equaled the source ROI. Its cache remains explicitly incomplete under `observations/`; no failed partial cache was relabeled complete or overwritten. The final `observations_verified/` run passed all checks. No package changes were needed.

## Regression and replay results

| Check | Actual result |
|---|---|
| C1 regression suite | **56 PASS / 1 SKIP**; optional Excel skip because openpyxl is absent. Native sklearn tests execute. |
| C2 regression suite | **30 PASS**, including fixed Hub ID/revision loading, explicit dataset index per crop, COCO17 config validation, native crop postprocessing, missingness, cache round-trip and real-cache association/replay. |
| Root layout checks | **2 PASS / 1 FAIL**: pre-existing untracked `draw_grid.py` is outside the allowlist. Preserved without execution/modification; the guard was not weakened. |
| Actual five-frame cache replay | Two native CPU runs produced identical prediction hashes; original cache hashes unchanged. |
| Metric calibration | NOT VERIFIED; remains fail-closed. No existing H was modified or used by raw inference. |

Actual replay used explicit image-space diagnostic settings `eps=50 pixels`, `close_distance=10 pixels` solely to test repeatability. No accuracy or physical interpretation is claimed. Neural-free replay remains protected by a subprocess import-guard regression. C1 grouping and group-ID logic remain unchanged.

Reproduce regressions:

```powershell
$py = 'C:\Users\trant\miniconda3\envs\paper-research\python.exe'
& $py -B tests/run_c1.py
& $py -B tests/run_c2.py
& $py -B tests/check_active_layout.py
```

Real-cache-only tests skip if the local smoke artifact is absent; they never download weights or rerun inference. See `C2_HANDOFF_TO_C3.md` for the exact bounded inference command.

## Real data evidence

Source: `WIN_20260303_10_09_56_PRO_GT_First60s/WIN_20260303_10_09_56_Pro.mp4`.

- Video SHA256: `fa1044aa1f3c9d413bdd608d39f8545c593b34457f5ecb6dc80a28beb62cb119`.
- XML SHA256: `67addc127c07d98c17d0c160dc03d5f4025e5926dfb4d2d786aa901af37f7f08`.
- 1920x1080, 2,956 frames; time base 1/30000; variable PTS increments. No fixed-30-FPS assumption.
- XML: CVAT 1.1 job 4549037, 76 persons, declared frames 0–2955.
- First 1,800 frames: 71,157 visible person boxes, including 66,223 `unreviewed` and 4,934 assigned G1–G9 attribute values. These remain partial, unfinished reference annotations.
- Source frame 1799 is at 65.18396666666666 seconds. `[0,1800)` is a frame interval, not literal 60 seconds.
- Decoding verified all first 1,800 frame positions/PTS. Overlays were generated for 0,1,3,30,300,319,334,899,900,1799. Frames 0,334,1799 were visually inspected: broad person-box correspondence and no obvious global offset/resizing error. Individual box precision, identity continuity and group correctness are not certified.

Local timing/review evidence: `.local/c2/alignment_20261109/alignment.json` and the separately created `alignment.reviewed.json`. The latter retains all source PTS and records the limited visual review. Neither changes the XML/video.

Reproduce alignment into a **new** output directory:

```powershell
& $py -B analyze_subgroups.py validate-data `
  --video 'WIN_20260303_10_09_56_PRO_GT_First60s/WIN_20260303_10_09_56_Pro.mp4' `
  --xml 'WIN_20260303_10_09_56_PRO_GT_First60s/annotations.xml' `
  --end 1800 --output '.local/c2/alignment_NEW_RUN'
```

Review newly generated overlays and record a separate reviewed alignment file before real cache export. No full-video pose/grouping inference or public benchmark was run.
