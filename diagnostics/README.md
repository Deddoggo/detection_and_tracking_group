# Evidence-first playground pilot

This folder is separate from the advisor manuscript and archived thesis results.
No new inference or playground GT metrics were run during its initial preparation.
Read `CHECKPOINT_VI.md`, `ANNOTATION.md` and `AUDIT.md` first.

For manual GT creation before inference, start with
`MANUAL_ANNOTATION_WORKFLOW_VI.md`. The verified15s clip and frame mapping are in
`manual_annotation_pilot01/`; `cvat_annotations.py` converts manual CVAT XML/ZIP,
keeps uncertainty and human review pending, checks labels and renders review frames.
No annotation is considered complete merely because conversion succeeded.

**Video update, 2026-10-05:** one raw file is now available. See
`video_review_2026-10-05/video_inventory.json` and `pilot_manifest.json` for measured
metadata and three candidate intervals. Timing is variable: use the generated
decoded timestamp table and zero-based frame ranges, not 60*seconds. Candidate
configs and annotation templates are ready; checkpoints/tracker YAML are still
missing. The selected intervals require continuous review before annotation.
The runner preserves per-frame source timestamps when `frame_timestamp_file` is
provided. ByteTrack remains frame-driven; this change does not implement a
time-aware motion model. No inference or playground GT evaluation has been run.

## A. Engineering correctness and raw baseline

`baselines/source_2db7aac.zip` is an exact tracked-source archive of the clean initial
commit. It preserves both original grouping implementations and legacy runners.
Ignored videos/checkpoints were not part of the Git snapshot. Extract to a **new**
directory for legacy reproduction; do not overwrite this worktree. Local ignored
output videos are unchanged. Historical weights/configs/environment remain unknown.

`group_identity.py` fixes group-ID overwrite through deterministic one-to-one
continuation. Greedy matching prioritizes intersection count, old-member retention,
then old-ID/cluster-label ties. One split child retains an old ID, other children
receive fresh monotonically increasing IDs. One merge parent retains its ID; other
parents end. Empty observations end current observed clusters. There is no new
missing-frame persistence or social membership model. Overlap lineage is inspection
metadata, not reference social split/merge events. Duplicate person IDs are errors.

The two existing modules import this fix without changing their existing tuple
return interfaces. `identify_group_social` defaults to the old gate for compatibility.
Use `orientation_mode="proximity"` for the orientation-free corrected comparator.

For a **same-input grouping correctness comparison**, copy the pilot config and set
`orientation_mode` to `legacy` in `pilot.legacy.json`. Then:

```powershell
python -m diagnostics.cli group --raw-baseline --config diagnostics/configs/pilot.legacy.json --input diagnostics/runs/run01_tracks.json --out diagnostics/runs/groups_raw_baseline.json
python -m diagnostics.cli group --config diagnostics/configs/pilot.legacy.json --input diagnostics/runs/run01_tracks.json --out diagnostics/runs/groups_corrected_legacy.json
python -m diagnostics.cli group --config diagnostics/configs/pilot.local.json --input diagnostics/runs/run01_tracks.json --out diagnostics/runs/groups_corrected_proximity.json
```

The raw option loads the original grouping code directly from the preserved source
copies in `baselines/`; they are byte-identical to this commit and intentionally
retain the known overwrite bug. It does not reproduce
archived detector/tracker results. The new detector/tracker runner is a **fresh
explicit pilot baseline**, not an exact reconstruction of legacy or thesis runs.
For original full-chain scripts, extract the source archive into a separate folder
and restore its hardcoded input paths and original environment/checkpoints first;
the commands there are `python detect_yolo.py`, `python track.py`,
`python stitch_tracks.py`, `python analyze_subgroups.py`. Those require missing
inputs and include external gender inference; this session does not execute them.

## B. Experimental infrastructure

Run from repository root using your actual inference environment's Python.
The bundled desktop Python can run cache/evaluator/correctness tools but has no
OpenCV, scikit-learn, PyTorch, Ultralytics, BoxMOT or Transformers installed.
No inference packages were installed or upgraded in this session.

The large full-source archive is intentionally ignored by Git. The two small
original grouping source files, baseline manifest, current config lock and protocol
are reviewable files. Recreate the full archive if needed with
`git archive --format=zip --output=diagnostics/baselines/source_2db7aac.zip 2db7aac`.

```powershell
python -m diagnostics.cli inventory --root . --out diagnostics/runs/inventory.json
python -m unittest discover -s tests -v
```

Copy `configs/pilot.template.json` to `configs/pilot.local.json`. Fill **confirmed**
raw video, absolute start/end frames, scene/session, explicit existing local
detector checkpoint and the installed-version ByteTrack YAML. The template contains
an unverified calibration and a proposed low-confidence cache floor; it is NOT the
thesis run config. Do not assign physical meters until ground calibration is checked.
Set `calibration_source_video` to the exact verified source descriptor, or explicitly
enable `allow_unverified_calibration` for inspection only. Without a homography,
set `calibration_file: null`, `coordinate_unit: "pixels"` and choose a pixel eps;
do not reuse 1.75 calibration units as 1.75 pixels. A matching filename does not
verify ground-plane landmarks or physical scale.
Device and precision are explicit. Copy the exact YAML used into the run archive;
configure buffer/thresholds through that YAML, not post-construction attributes.
Inspect the tracker implementation and saved effective attributes for your version.

```powershell
python -m diagnostics.cli freeze --config diagnostics/configs/pilot.local.json --out diagnostics/runs/run01_config_lock.json
python -m diagnostics.run_pilot --config diagnostics/configs/pilot.local.json --stage detect --out diagnostics/runs/run01_detections.json
python -m diagnostics.run_pilot --config diagnostics/configs/pilot.local.json --stage track --detections diagnostics/runs/run01_detections.json --out diagnostics/runs/run01_tracks.json
python -m diagnostics.cli group --config diagnostics/configs/pilot.local.json --input diagnostics/runs/run01_tracks.json --out diagnostics/runs/run01_groups.json
```

Each pilot is bounded to <=20s, contiguous frames, starting with a fresh tracker.
Document startup effects when interpreting short clips; future warm-up must keep
scored frame ranges identical between conditions. Frames keep absolute zero-based
source indices. Output files refuse overwrite. JSON exports bbox/confidence, tracks
and group members/IDs/positions/cached direction proxies. Hashes, package versions,
command and effective tracker scalars are recorded. Exact local checkpoint is
mandatory: no silent model replacement, downloads or implicit tracker config.

The detector cache contains **post-NMS person detections**, float xywh boxes and
scores. It is not a cache of pre-NMS proposals. Keep NMS, imgsz, max_det, checkpoint,
precision/device fixed in a confidence sweep. Cache once at a low floor, then filter:

```powershell
python -m diagnostics.cli sweep --input diagnostics/runs/run01_detections.json --thresholds 0.10 0.15 0.20 0.25 --out-dir diagnostics/runs/sweep01
```

Run `--stage track` separately on each filtered cache using the SAME tracker YAML
and input cutoff no higher than the sweep's smallest threshold. Each call starts a
fresh tracker. Vary detector input cutoff only; do not silently also change
ByteTrack high/low association thresholds. A sweep below the cache floor is refused.
This is a post-NMS cache experiment; it is not guaranteed identical to rerunning
YOLO at each threshold (NMS/max_det implementation can couple predictions).

Existing JSON is reusable with `convert`: supply a JSON source descriptor with
video/frame range/FPS. Historical caches have unknown confidence floor unless
verified; the sweep refuses them. Neither missing frames nor missing scores are
invented. Keypoints/direction/confidence fields, if present, are retained for
downstream replay. Old final_sociometrics_data has no raw keypoints/directions:
orientation cannot be reconstructed reliably from it. PETS pickle caches are
preserved; do not load untrusted pickle automatically. Pose inference is not
implemented by the new runner: use the existing pose implementation in your
confirmed environment, cache actual keypoints/directions, and then replay grouping.
No gender cloud call or stitching is invoked by this initial pilot runner.

```powershell
python -m diagnostics.cli convert --input playground_results/detections.json --kind detections --source diagnostics/source.local.json --out diagnostics/runs/imported_detections.json
python -m diagnostics.cli annotation-template --source diagnostics/source.local.json --out diagnostics/runs/annotations_pending.json
python -m diagnostics.cli validate --input diagnostics/runs/annotations_reviewed.json
python -m diagnostics.cli evaluate-detection --input diagnostics/runs/run01_detections.json --annotations diagnostics/runs/annotations_reviewed.json --out diagnostics/runs/detection_metrics.json
python -m diagnostics.cli prepare-trackeval --annotations diagnostics/runs/annotations_reviewed.json --tracks diagnostics/runs/run01_tracks.json --sequence pilot01 --out-dir diagnostics/runs/eval01
```

The TrackEval export prepares gt/seqinfo, tracker data, seqmap and a manifest; it
does not calculate official metrics itself. After restoring the official TrackEval
installation, inspect its `scripts/run_mot_challenge.py --help` and use its
MotChallenge2DBox adapter with this custom pilot. Intended options (not executed or
version-verified in this environment):

```powershell
python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --GT_FOLDER diagnostics/runs/eval01/gt --TRACKERS_FOLDER diagnostics/runs/eval01/trackers --TRACKERS_TO_EVAL estimated --SEQMAP_FILE diagnostics/runs/eval01/seqmap.txt --SKIP_SPLIT_FOL True --DO_PREPROC False --USE_PARALLEL False --METRICS HOTA CLEAR Identity
```

Here MOT17 selects the adapter schema; **the data are a custom diagnostic pilot,
not MOT17 data**. Preserve evaluator version/config/output. Do not publish tracking
metrics until the installed adapter and its known-outcome sanity fixtures are
checked. Current tests verify export/matching, not the absent official evaluator.

## Oracles, when reference is sufficient

```powershell
python -m diagnostics.cli oracle --annotations diagnostics/runs/annotations_reviewed.json --mode gt_boxes --out diagnostics/runs/oracle_boxes.json
python -m diagnostics.run_pilot --config diagnostics/configs/pilot.local.json --stage track --detections diagnostics/runs/oracle_boxes.json --out diagnostics/runs/oracle_boxes_tracked.json
python -m diagnostics.cli oracle --annotations diagnostics/runs/annotations_reviewed.json --mode gt_boxes_ids --out diagnostics/runs/oracle_boxes_ids.json
python -m diagnostics.cli oracle --annotations diagnostics/runs/annotations_reviewed.json --mode matched_detection_ids --detections diagnostics/runs/run01_detections.json --out diagnostics/runs/oracle_detected_ids.json
```

Compare (1) full estimated tracks, (2) GT boxes freshly run through tracker,
(3) GT boxes+GT IDs, (4) detected boxes assigned GT IDs by one-to-one IoU matching.
GT boxes confidence=1.0 is an explicit artificial oracle condition. No tracked IDs
are borrowed from another run. In (4), misses remain absent; unmatched FPs and
duplicate detections are **dropped and counted**, making this an optimistic
matched-only diagnostic, not a pure association isolation. Preserve this distinction.
Run the same grouping config on each output. Pose-dependent comparisons require
pose on each condition's actual boxes; do not transfer pose cues from incompatible
boxes/IDs. Begin proximity-only if these caches do not exist.

Without reliable group reference these outputs permit inspection only. Group
metrics, stability and event evaluation remain conditional on suitable labels.
Use the same group GT, frame mask, definition and missing-person/FP policy in every
condition. Module errors can interact; do not add deltas as error percentages.

## C. Research candidates

`orientation_gate.py` retains `legacy`, `proximity`, and optional
`facing_experimental`. The old absolute-cosine OR gate is non-selective for valid
unit 2D directions. The optional candidate requires BOTH signed directions toward
the other person, with explicitly supplied reliability >=0.6. Missing/unreliable
cues fall back to proximity; very-close pairs retain the existing <0.5 bypass.
This candidate assumes signed directions express ground headings. The current
image pose heuristic does not establish that assumption, and keypoint confidence
is not automatically heading reliability. Keep it disabled until diagnostic
inspection/reference can support it. It is not claimed to improve grouping.
# Latest annotation range / GPU baseline preparation

See `gt_first900_2026-10-06/CHECKPOINT_VI.md` and `GPU_BASELINE_VI.md`.
The user reviewed a full-video CVAT prefix0–900; the distant-candidate inclusion
policy remains to clarify. Two questioned near-camera regions were confirmed as
objects by the user. Conversion is structurally valid but pending; do not score
the provisional MOT rows. Resolve `ANNOTATION_REVIEW_VI.md` and export a new CVAT
version before locking GT. No new inference or metrics have been run locally.
