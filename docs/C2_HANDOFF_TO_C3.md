# C2 handoff and pending prerequisites

C2 structural paths are implemented and tested. **C3 has not begun.** Real pose-cache acceptance and trusted metric replay remain pending; this is not a full benchmarking approval.

## Required next inputs

1. Supply a local Hugging Face-format snapshot of `usyd-community/vitpose-base-simple`, with its revision/provenance. Required files are `config.json`, `preprocessor_config.json` and `model.safetensors` (or a complete supported local weight/shard set). An original MMPose `.pth` alone is not the active Transformers format. No YOLO/BoxMOT checkpoint is needed for XML-based caching.
2. Supply the actual-video homography, image/ground correspondences, measured dimensions/units and measurement/camera provenance. The existing edited H remains unverified for metric use. We can then assemble and check a `calibration.v1` manifest; the example in `config/` deliberately cannot pass as verified calibration.

The upstream model repository inspected at [revision a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45](https://huggingface.co/usyd-community/vitpose-base-simple/tree/a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45) lists approximately 344 MB of weights. Its [configuration](https://huggingface.co/usyd-community/vitpose-base-simple/blob/main/config.json) specifies the Transformers ViTPose class, simple decoder and COCO17 label order; the [processor configuration](https://huggingface.co/usyd-community/vitpose-base-simple/blob/main/preprocessor_config.json) uses a 256x192 pose input. Only metadata was inspected; no weights were downloaded. Local file hashes and supplied revision will be recorded, without falsely claiming an upstream hash match.

## Environment/dependency plan

Continue using `C:\Users\trant\miniconda3\envs\paper-research\python.exe` explicitly. Required C2 paths use existing Python, NumPy, OpenCV, sklearn, PyTorch, Transformers, Pillow and safetensors/HF dependencies. Tests/native imports succeeded; no package installation is currently proposed. `inference` is unnecessary with auxiliary attributes disabled; `openpyxl` is optional reporting functionality.

The latest torch build supports CUDA and a scalar allocation/readback succeeded. Given the approximately 344 MB checkpoint, FP32 and **one-person batches** are a plausible bounded test on the local 6 GB GPU, but actual model/activation memory remains unverified. Start with a few source samples and batch size 1. Do not provision Vast or change torch/packages without a separate concrete plan and approval.

## Bounded next execution, after checkpoint supply

The following is a reproduction template, **not an already executed inference command**. Replace local checkpoint/revision with the supplied snapshot. Use a new output directory.

```powershell
$py = 'C:\Users\trant\miniconda3\envs\paper-research\python.exe'
& $py -B analyze_subgroups.py cache `
  --video 'WIN_20260303_10_09_56_PRO_GT_First60s/WIN_20260303_10_09_56_Pro.mp4' `
  --xml 'WIN_20260303_10_09_56_PRO_GT_First60s/annotations.xml' `
  --alignment '.local/c2/alignment_20261109/alignment.reviewed.json' `
  --checkpoint 'LOCAL_HF_VITPOSE_DIRECTORY' `
  --checkpoint-revision 'SUPPLIED_SNAPSHOT_REVISION' `
  --start 0 --end 13 --stride 3 --max-pose-frames 5 --pose-batch-size 1 `
  --device cuda:0 --output '.local/c2/pose_smoke_NEW_RUN'
```

This schedules frames 0,3,6,9,12. It preserves source PTS and all explicit XML persons on those frames, including exclusions. No calibration is required for raw caching. Inspect pose layout, confidence values, person/keypoint association, missingness and bounded memory before authorizing a 600-sample export over `[0,1800)`.

For grouping-only replay of an actual validated cache:

```powershell
& $py -B analyze_subgroups.py replay `
  --cache '.local/c2/pose_smoke_NEW_RUN' `
  --geometry-mode image-space-diagnostic --eps 50 --close-distance 10 `
  --output '.local/c2/replay_NEW_RUN'
```

The values 50 and 10 are **illustrative explicit pixel diagnostic settings**, not calibrated/optimized research parameters. No metric substitution is allowed. `eps` must remain below the existing rejected-distance sentinel 1000. Metric replay instead requires `--geometry-mode metric`, a verified `--calibration` manifest and explicitly chosen meter thresholds.

Replay output contains source frame, PTS, person ID, bbox, predicted group ID, raw/smoothed/projected headings, heading freshness/age, positions/units and null group confidence. Predictions are pre-annotations, never GT. Excluded persons have null group IDs; eligible DBSCAN noise has -1. Raw `unreviewed` references remain unknown and are never used as negative membership or clustering input.

## Compatibility and remaining limits

- The old live path requires explicit `--compatibility-b0 --tracks PATH`; its hardcoded video/output/model behavior remains separate. Do not run it over historical outputs. B0 unit semantics are deliberately marked unverified.
- Existing visualization/report scripts consume their legacy JSON format, not the new replay JSONL. No conversion or final evaluation harness is claimed here.
- Initial cache input is oracle boxes/person identities, not detector/tracker accuracy. No predicted-track adapter/identity correspondence has been validated.
- Cache v1 export starts at source frame 0; nonzero seek/offset/resume is rejected. Replay resets at cache start rather than inventing warm-up history.
- PTS packet mapping currently supports verified, ordered no-B-frame streams. Other videos require a proper decoded presentation index; no silent fallback.
- Reader validation loads scheduled records before replay; this is appropriate for the present scope, not a claim of unlimited-scale streaming.
- Complete subgroup review masks, reviewed no-group labels and independent headings/physical positions remain future reference work. Do not infer these from predictions.
- B0 absolute-cosine tautology, DBSCAN chaining and heuristic heading/footpoint limitations remain baseline methodology. No H2/H3/H4, tracker retuning, model training, public benchmarking or final Setti F1 harness was implemented.

Stop here until the missing checkpoint and calibration evidence are supplied. Additional packages or external GPU resources require separate approval.
