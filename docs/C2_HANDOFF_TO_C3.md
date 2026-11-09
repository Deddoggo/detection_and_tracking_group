# C2 acceptance handoff — observation engineering complete

**C2 observation-cache engineering is complete.** Curated tracked observation/replay commits are ready for integration into master with explicit geometry/workspace limitations. No merge is authorized or performed here. Trusted metric grouping is not complete. C3/H2/H3/H4 remain unstarted.

## Artifacts to retain

Root: `.local/c2/full_acceptance_20261109/` (ignored local outputs; source data/model weights are not committed).

| Artifact | Identity/purpose |
|---|---|
| observations/manifest.json | observations.v1 complete, 600 source samples; SHA256 fda85542696711cd50567a20be216ef4d903416423cf4c0b8cec3eb4bcf67366 |
| observations/frames.jsonl | 23,734 retained records, 23,426 actual poses; SHA256 7a8efcbc164f6980409e8b3d57a250365152ca44188a3bf463366601a66975b6 |
| observations/inference_metrics.json | All crop/ID/expert/shape/timing records and CUDA allocator peaks; manifest-hashed |
| replay_a/, replay_b/ | Complete neural-free image diagnostics; prediction SHA256 49235da70d967a40b2f18ddb7ad8256e0de307eacf6ae7c16cea1d85bcabda6f |
| calibration.existing-H.v1.json | Additive unverified manifest referencing unchanged existing H; not permission for metric use |
| cache_validation.json, replay_validation.json, regression_validation.json | Full acceptance receipts |
| export_execution.json, export.log, C1.log, C2.log, workspace_layout.log | Execution/exit/time evidence |
| docs/c2_full_acceptance.json | Consolidated portable evidence, acceptance A-G and preservation/source/model provenance |

The old smoke cache `.local/c2/vitpose_plus_20261109/observations_verified/` remains immutable evidence. Its arrays match the five-frame prefix of the full cache exactly; optional extra metadata in the new full cache preserves z_order and track provenance without invalidating old v1 records.

## Exact full-export command (already executed)

Use the existing paper-research interpreter from the repository root. The output below already exists and is intentionally refused if reused; choose a new output path for an explicitly authorized repetition.

```powershell
$py = 'C:/Users/trant/miniconda3/envs/paper-research/python.exe'
& $py -u -B analyze_subgroups.py cache `
  --video 'WIN_20260303_10_09_56_PRO_GT_First60s/WIN_20260303_10_09_56_Pro.mp4' `
  --xml 'WIN_20260303_10_09_56_PRO_GT_First60s/annotations.xml' `
  --alignment '.local/c2/alignment_20261109/alignment.reviewed.json' `
  --revision 92be54d7a29e42fad47b6e2ca01dd9e685a61e0d `
  --start 0 --end 1800 --stride 3 --max-pose-frames 600 --pose-batch-size 1 `
  --device cuda:0 --output '.local/c2/full_acceptance_20261109/observations'
```

Only usyd-community/vitpose-plus-base is supported. Hugging Face from_pretrained loads model/processor directly by ID at the same SHA and reuses standard cached weights. No manual checkpoint loader or environment installation is required. This is an oracle-box/oracle-identity path, not end-to-end detection/tracking performance.

## Reproduce artifact validation and replay

The export wall-time receipt was produced by the isolated execute_export.py wrapper. It records argv, exit status, start/end UTC and whole-process elapsed time. Validation reads source/cache artifacts without changing them:

```powershell
& $py -B tests/validate_c2_full.py `
  --cache '.local/c2/full_acceptance_20261109/observations' `
  --video 'WIN_20260303_10_09_56_PRO_GT_First60s/WIN_20260303_10_09_56_Pro.mp4' `
  --xml 'WIN_20260303_10_09_56_PRO_GT_First60s/annotations.xml' `
  --alignment '.local/c2/alignment_20261109/alignment.reviewed.json' `
  --execution '.local/c2/full_acceptance_20261109/export_execution.json' `
  --output '.local/c2/validation_NEW.json'
& $py -B analyze_subgroups.py replay `
  --cache '.local/c2/full_acceptance_20261109/observations' `
  --geometry-mode image-space-diagnostic --eps 50 --close-distance 10 `
  --output '.local/c2/replay_NEW_RUN'
& $py -B tests/run_c1.py
& $py -B tests/run_c2.py
& $py -B tests/check_active_layout.py
```

The executed two-run wrapper execute_replays.py adds import guards, times each CPU process and compares complete predictions and raw-cache hashes. Its record is replay_validation.json. The pixel thresholds above are diagnostics, not physical or optimized subgroup parameters. Predictions are pre-annotations; incomplete unreviewed references remain unknown. Do not derive ground truth or Setti F1 from these predictions/partial labels.

## Remaining geometry input

H generation is already done. The actual available export is calibration_matrix_playground.json, numerically valid and source-compatible. If a distinct homography_matrix.json was intended, provide its actual path; it was not found in the searched locations. Do not regenerate or replace H merely to change its filename.

For trusted metric replay, provide the existing measurement record or confirmation identifying the four actual coplanar ground landmarks at the coded image coordinates, their 10.42x11.54 dimensions/units, measurement verifier and accepted physical-error tolerance. The paper's measurement statement is retained as a claim. Once that evidence is reviewed, the additive calibration manifest can be completed and a bounded metric replay run separately. Do not set verified=true solely because the matrix is finite or its source image matches the camera.

## Integration and local utility treatment

The existing workspace layout guard remains red because draw_grid.py is a root-level untracked, user-owned visualization tool. No active script imports it. It was neither executed, edited, deleted nor force-added, and the allowlist was unchanged. A clean tracked implementation snapshot passed all three layout tests. With separate owner authorization, a non-active tools/calibration location is an appropriate future home; no relocation happened in C2.

Integrate only curated C2 commits, excluding the pre-existing calibration/image changes and grid utility. No automatic merge or further work follows this handoff. The full cache and repeatable CPU replay are accepted engineering artifacts; physical metric validity and scientific grouping accuracy remain separate. Existing legacy renderer/report JSON interfaces are unchanged; public adapters/benchmarks and new research methods belong to subsequent authorized work.
