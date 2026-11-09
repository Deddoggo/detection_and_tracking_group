# Repository Instructions — Active Research Pipeline

## Active pipeline

The authoritative active files are:

- `detect_yolo.py`
- `track.py`
- `analyze_subgroups.py`
- `identify_group_social.py`
- `orientation_utils.py`
- `calibration_headless.py`
- `render_visuals.py`
- `generate_report.py`

`stitch_tracks.py` is optional and should only be modified for explicitly relevant tasks.

## Modification scope

- Treat all other historical Python scripts as read-only unless explicitly authorized.
- Legacy files may be inspected for context but must not be refactored or repaired automatically.
- Make minimal, testable changes to the active pipeline.
- Tests, configurations, and small helper modules may be added when directly required.
- Do not refactor unrelated scripts or reorganize the repository.
- Do not modify original datasets, annotations, checkpoints, or historical experimental outputs.
- Preserve baseline behavior for scientific comparison.
- Use Git branches and focused commits for changes.
- Report every modified file and its purpose.

## Research requirements

- Keep engineering correctness repairs separate from scientific methodology changes.
- Do not automatically replace the existing subgrouping algorithm.
- Preserve the historical B0 baseline.
- Use explicit source-frame indices, timestamps, and coordinate units.
- Do not claim benchmark improvements without controlled evaluation.
- Do not infer group ground truth from model predictions.

## Documentation

- Avoid redundant Markdown files.
- Prefer a concise changelog, test report, and milestone handoff.
- Keep machine-readable experimental evidence and version manifests separate.
- Do not create unnecessary documentation for obsolete branches.

## Before completing a task

Check the Git diff and verify that only authorized files were modified.

If an essential fix requires changing a file outside this boundary, explain the dependency and request authorization instead of expanding the scope automatically.