# Rebuilt active C1 validation

Clean-master parent: `3b3f75ae5b7b0b69e81c1d9a49f1eb7febf5711a`.
Original B0: `2db7aac2fca6f03e1a974b1c46a921ab54330160`.
Current implementation is `fix/c1-active-pipeline-only`; its commit is the one
containing this report. The original master and both earlier C1 tips remain
reachable through the archive refs described in `REPOSITORY_CLEANUP.md`.

## Results and execution limits

| Check | PASS | FAIL | SKIP |
|---|---:|---:|---:|
| Focused C1 regression suite | 54 | 0 | 3 |
| Source compilation/layout/resources | 3 | 0 | 0 |

Commands from repository root:

```powershell
python -B tests/check_active_layout.py
python -B tests/run_c1.py
```

Executed with Anaconda Python 3.13.5, NumPy 2.5.4 and OpenCV 5.0.0.93 on Windows.
No packages, models or datasets were installed. Two native sklearn integration
tests (chain and split/noise) skip because sklearn is absent. Native Excel export
skips because pandas/openpyxl are absent. These are NOT VERIFIED native runtime
behaviors, not successful mocked integrations. No model inference or benchmark.

`c1_validation.json` contains every test identifier, input, expected result,
actual result, PASS/FAIL/SKIP status, environment, dependency mode and source hashes.
Long orientation sweeps record their tested counts rather than every angle.
Fresh suite runs write uniquely named, exclusive-created JSON under ignored
`.local/c1_validation/`, preserving previous evidence. Tests create scratch files
there, never in historical output directories.

## Coverage

- Native pure association tests: continuation, split with no lost component,
  deterministic merge, unrelated groups, ties, raw-label/member permutations,
  asymmetric old coverage, empty state, repeated empty intervals, all noise,
  disappearance, explicit new-run initialization and split/gap/reappearance.
- Actual active grouping entry point: split/singleton/empty lifecycle, invalid
  inputs/projection, B0 matrix preservation and diagnostic-mode dispatch. These
  use real NumPy/OpenCV plus a clearly labeled `GraphDouble` DBSCAN substitute.
  They are not evidence of native sklearn execution.
- Orientation helpers: original .7/.3/.3 abs-cosine OR, controlled mutually facing,
  back-to-back, parallel, side-by-side and perpendicular dyads; missing/zero/
  malformed/nonfinite vectors; thresholds/bypass/radius boundaries; 12,960,000
  angular pairs and 1,000,000 seeded random pairs confirm the original tautology.
  Signed mutual-facing AND is diagnostic only, never the scientific default.
- Active input contracts: lists/NumPy bboxes/IDs, empty inputs, mismatched lengths,
  duplicate IDs, invalid widths/shapes/nonfinite values, keypoint arrays, supplied
  singular/malformed calibration and horizon projection. Missing H retains B0's
  pixel-space fallback, which still needs an explicit C2 units/failure policy.
- Reporting: source FPS 30/27.848645/7, stride3, verified timestamps with priority,
  missing/nonuniform samples, gaps, session continuation, membership transitions,
  split/merge snapshots, single-frame/empty cases, half-open intervals, clipping,
  triangle/collinear/moving geometry, missing positions and Unknown attributes.
- Preservation: all nine B0 Git blob hashes match the cleanup manifest; helper
  imports and call sites are checked in active source; retired source is absent
  and no legacy path changes relative to clean master. Layout checks compile
  active source without importing side-effectful model stages.

## Deliberate compatibility changes

Only three existing active files change relative to clean master:
`identify_group_social.py`, `orientation_utils.py`, `generate_report.py`. The four
new helpers are called by those files. The selected seven production files match
their counterparts in preserved broad C1; only the tests' baseline/evidence paths
were adapted for this cleaned tree. No obsolete implementation was migrated.

One-to-one globally ordered greedy matching retains the old asymmetric coverage
eligibility (not IoU); it prevents collisions, allocates fresh IDs and clears only
active observed state at gaps. It does not infer group dissolution or event lineage.
The allocator is initialized to -1 only for an independent run/video namespace.
Current component membership is not temporally regularized.

The B0 abs-cosine predicate and DBSCAN/default thresholds remain intact. For unit
planar headings, if both directional absolute dots are <=.3, the mutual absolute
dot is >=.82>.7. Thus C1 bookkeeping does not make this a validated social model.
Missing heading still uses its declared B0 proximity fallback; invalid arrays now
raise explicit errors instead of acting as positive social evidence.

Reporting is explicitly `C1-report-v1`: timestamps or explicit nominal source FPS,
separate source indices/observation order/stride and declared coordinate units.
Counts distinguish IDs, framewise observations and constant-membership observed
sessions. Formation events are undefined without event annotation. Unknown
auxiliary labels never remove group members. Missing geometry is not an origin.
Instantaneous untrimmed hull area is null for fewer than three/missing positions
and zero for valid collinear triples. Time averages use supported observed spans,
not pooled positions; gaps/transitions/terminal samples receive no invented time.
Single-frame observed span is zero. The old demographic travel/distance tables
are not silently redefined; their implementation remains in original history.

New report CLI requires new output, timing, stride and units, for example:

```powershell
python generate_report.py --input INPUT.json --output NEW_REPORT.json --fps 27.848645 --stride 3 --coordinate-units m
```

This is a usage example, not verification of a particular video's FPS or meters.
For VFR use verified source-frame timestamps via `--timestamps` instead. Output
overwrite is refused; historical outputs were not migrated or reinterpreted.

## Stop boundary and remaining work

C2 observation caches/exports and GT/frame/units validation, C3 canonical model
integration, C4/public benchmarks and H2/H3/H4 research methods were not started.
Data/checkpoint/version/calibration provenance and full inference remain
unresolved. Passing software regressions is not whole-pipeline benchmark approval.
