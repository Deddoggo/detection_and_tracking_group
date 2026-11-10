# C2 closed — handoff for separately authorized C3/W3/W4

**C2 is closed and integrated into local master.** Implementation: `9469fbb03fcd8947830e114aa633af8a366a2552`; final closure documentation is recorded by its containing commit. No push was performed. C2 feature branch is retained; original archived master and all accepted artifacts/user changes are preserved.

## Current artifacts and inputs

- Full immutable observations: `.local/c2/full_acceptance_20261109/observations/`, observations.v1, 600 source samples, 23,426 successful poses, zero missing eligible poses. Manifest SHA256 `fda85542696711cd50567a20be216ef4d903416423cf4c0b8cec3eb4bcf67366`.
- Original deterministic image runs: `.local/c2/full_acceptance_20261109/replay_a/` and replay_b/. Fresh closure image runs match their full prediction hash.
- Current owner-confirmed calibration manifest: `.local/c2/closure_20261110/calibration.user-confirmed.v1.json`. Original H file remains calibration_matrix_playground.json, unchanged. Width 10.42 m, height 11.54 m and stored landmark relationships are user-confirmed; independent physical-error accuracy remains unknown.
- Bounded metric result on integrated master: `.local/c2/closure_20261110/metric_replay_master/` (five records, source frames 0,3,6,9,12), eps=1.75 m and close_distance=0.5 m. Finite coordinate/heading checks passed; this is not subgroup or metric-accuracy ground truth.
- Current closure/integration/test/preservation evidence: `closure` in docs/c2_full_acceptance.json and `.local/c2/closure_20261110/`. Original full-run receipts and smoke cache remain unchanged.
- Source-frame/PTS alignment: `.local/c2/alignment_20261109/alignment.reviewed.json`; original video/XML hashes remain recorded. Input is explicitly oracle boxes/identities; detector/tracker performance is not measured by this cache.

## Reproduce checks, not raw inference

```powershell
$py = 'C:/Users/trant/miniconda3/envs/paper-research/python.exe'
& $py -B tests/run_c1.py
& $py -B tests/run_c2.py
& $py -B tests/check_active_layout.py
& $py -B tests/validate_c2_full.py `
  --cache '.local/c2/full_acceptance_20261109/observations' `
  --video 'WIN_20260303_10_09_56_PRO_GT_First60s/WIN_20260303_10_09_56_Pro.mp4' `
  --xml 'WIN_20260303_10_09_56_PRO_GT_First60s/annotations.xml' `
  --alignment '.local/c2/alignment_20261109/alignment.reviewed.json' `
  --execution '.local/c2/full_acceptance_20261109/export_execution.json' `
  --output '.local/c2/closure_integrity_NEW.json'
```

For bounded metric replay from the existing cache, use analyze_subgroups.py replay with --geometry-mode metric, the owner-confirmed --calibration file, --eps 1.75 --close-distance 0.5 --max-frames 5 and a **new** output directory. Omit max-frames only when a separately authorized full replay is intended. Existing cache/runs must not be overwritten. No ViTPose inference/model download is needed for replay.

## Inputs needed by subsequent work (not begun)

| Work | Required inputs/protocol decisions |
|---|---|
| C3 module benchmarking | Frozen source/model versions and adapters, target dataset/annotation formats and permitted data, independent reference labels, evaluation splits/metrics, source timing and units. Keep oracle-input and end-to-end detection/tracking conditions separate. COCO17 image keypoints and planar H positions are not full metric 3D pose GT. |
| W3 temporal research | Independently labeled event-rich membership intervals, visibility/disappearance rules, causal/latency requirements, group/person identity and lineage semantics, and relevant temporal comparators. Existing heading EMA and C1 ID association are not a validated H4 model. |
| W4 final research selection/integration | Agreed final target, permitted references/data, independent test units, submission/resource constraints, and evidence from isolated candidate experiments before selecting a contribution or scientific architecture. Local Git integration of C2 is not W4 research selection. |

These W3/W4 inputs follow the already available W2 handoff at `C:/Users/trant/Downloads/SCIA Paper Publication/W2_research_2026-11-09/W2_CANDIDATES_AND_HANDOFF.md:252`; no new W3/W4 model was designed or implemented here.

For independent metric-accuracy claims later, obtain real landmark localization/projection-error evidence. C2's floating-point fit limit is not that evidence. Preserve raw scores as uncalibrated model outputs. G1-G9/unreviewed remain unfinished reference attributes; do not interpret unreviewed as negative membership, manufacture subgroup GT or compute final Setti F1 from it.

## User work and publication

The eight existing calibration/image changes remain unstaged, with their contents/absence unchanged. draw_grid.py was already absent at the beginning of closure; no deletion, relocation or allowlist weakening occurred. Current workspace layout tests pass. No calibration regeneration or personal-file commit was needed.

No further input is required to close C2. Review the local commits and remote before publishing. Exact push command for user review only:

```powershell
git push origin master
```

This command was **not executed**. Do not force-push. Any C3/W3/W4/H2/H3/H4/public benchmark work requires a separate instruction. No independent physical or social-group accuracy, training gain or benchmark improvement is claimed.
