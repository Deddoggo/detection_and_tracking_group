# C2 bounded validation handoff

The ViTPose+ clarification is implemented and the authorized five-frame CUDA test passed. **C3, the 600-sample export and H2/H3/H4 have not begun.** This establishes bounded execution and observation integrity, not scientific accuracy or full benchmark readiness.

## Fixed Hugging Face workflow

Only `usyd-community/vitpose-plus-base` is used. `AutoProcessor.from_pretrained()` and `VitPoseForPoseEstimation.from_pretrained()` receive that exact model ID and the same immutable revision. Hugging Face automatically downloads/caches weights. No manually prepared checkpoint directory or alternative model option is required or supported.

Default loading resolves `main` once through the Hub API. For exact reproduction, pass the executed full revision **`92be54d7a29e42fad47b6e2ca01dd9e685a61e0d`**. Each forward uses COCO human-pose `dataset_index=0`. Architecture/joint ordering and artifact hashes are recorded in `observations.v1` provenance.

## Executed bounded command

Run from the repository root using the existing environment. Use a **new** output directory if repeating; the completed cache below already exists and will be refused as an output destination.

```powershell
$py = 'C:\Users\trant\miniconda3\envs\paper-research\python.exe'
& $py -u -B analyze_subgroups.py cache `
  --video 'WIN_20260303_10_09_56_PRO_GT_First60s/WIN_20260303_10_09_56_Pro.mp4' `
  --xml 'WIN_20260303_10_09_56_PRO_GT_First60s/annotations.xml' `
  --alignment '.local/c2/alignment_20261109/alignment.reviewed.json' `
  --revision 92be54d7a29e42fad47b6e2ca01dd9e685a61e0d `
  --start 0 --end 13 --stride 3 --max-pose-frames 5 --pose-batch-size 1 `
  --device cuda:0 --output '.local/c2/vitpose_plus_20261109/observations_verified'
```

Frames 0,3,6,9,12 yielded 213 finite COCO17 pose arrays and 5 explicit tiny-box exclusions among 218 retained records. Person IDs/boxes/attributes/source PTS match the XML/alignment. Raw model scores are retained without clipping. XML identities are oracle inputs; this is not detector/tracker evaluation or complete subgroup GT.

Artifacts:

- `observations_verified/manifest.json`, `frames.jsonl`: complete observations.v1 cache with exact model revision and model/processor/weights hashes.
- `observations_verified/inference_metrics.json`: every crop/person call, shapes, dataset index, timing and CUDA allocator peaks.
- `.local/c2/vitpose_plus_20261109/smoke_validation.complete.json`: assertion-based validation and repeatable replay results.
- `pose_overlays/frame_00000.jpg`, `frame_00012.jpg`: inspected original-video overlays; 0.6 is a display threshold only.
- `docs/c2_vitpose_plus_validation.json`: portable current evidence; `docs/c2_validation.json` remains prior structural evidence.

The cache is reusable without neural inference. For example, with a new output path:

```powershell
& $py -B analyze_subgroups.py replay `
  --cache '.local/c2/vitpose_plus_20261109/observations_verified' `
  --geometry-mode image-space-diagnostic --eps 50 --close-distance 10 `
  --output '.local/c2/replay_NEW_RUN'
```

These pixel settings are engineering diagnostics, not tuned research thresholds. Two actual cache replay runs produced identical predictions and left raw observations unchanged. Predicted group IDs are pre-annotations, never GT. `unreviewed` references remain unknown; excluded persons have null predictions and eligible clustering noise has -1.

## Remaining requirements and scope

1. **Metric calibration evidence:** existing homographies were preserved. Before metric replay, verify their actual-video correspondence, physical measurements/units and camera provenance, then validate a calibration.v1 manifest. Raw CUDA caching is already independent of this verification.
2. **Repository layout:** untracked `draw_grid.py`, present before this clarification, causes one existing root-layout check to fail. It was not edited, staged or deleted. Core C1/C2 tests pass; do not describe all repository checks as green.
3. **Future scientific evaluation:** incomplete G1–G9 labels are not complete subgroup GT. No pose/group accuracy or benchmark gain follows from finite arrays or visual spot checks. Many distant/occluded joints have low scores.

No environment repair is needed for the executed case: paper-research's installed CUDA PyTorch/Transformers ran ViTPose+ successfully. No packages were changed and Vast.ai was not contacted. The failed early attempts remain distinguishable from the successful cache. Do not run the full 600-sample export without separate authorization.

Other preserved limits: cache export starts at source frame 0; other video timing/order assumptions require explicit validation. Existing legacy renderer/report scripts consume their original JSON format, not C2 replay JSONL. The B0 absolute-cosine predicate, DBSCAN chaining, heading heuristic and C1 association are unchanged.
