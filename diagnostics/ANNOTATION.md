# Minimal diagnostic annotation pilot

Start with 2–3 reviewed segments of 10–20 seconds from the five raw videos. Record
absolute source timestamps/frame ranges, FPS, scene, session and camera. The same
source/session is not an independent sample. Select an easy interval, a missed/
occluded interval, and (if available) a crossing/FP/duplicate interval. Do not
choose the final operating point using these clips and claim held-out performance.

## Person reference

Use a tool you already have (for example a local annotation tool), or the JSON
interface below. Draw pixel **xywh** boxes around the visible person extent,
consistently; do not invent full-body extents behind an occluder. Record truncation
separately. Assign a persistent integer `person_id` within the pilot. Reappearance
retains ID only if identity can be established visually. Do not infer identity
from tracker output. Predictions may be annotation suggestions, never automatic GT.

For each frame include every discernible person, including small/partly occluded
people, or explicitly mark unresolved cases. Visibility is a rough visible fraction
in [0,1], with `occlusion` = none/partial/heavy and `identity_uncertain` for review.
Fully invisible people get no invented bbox. Keep the reference identity across a
gap only if confident. For an unresolvable visible person use `person_id: null`;
its bbox can still be scored for detection. A genuinely unlocalizable region may
be marked `ignore: true`, with a written reason; never exclude difficult people
just to improve the scores. Review interpolated annotations on every frame before
marking full coverage.

Absolute frames are zero-based, end is exclusive. Pixel boxes are float-valued and
unrounded. The video string and frame range must agree exactly with prediction
metadata. Keep a video hash when possible.

```json
{
  "schema_version": 1,
  "kind": "annotations",
  "source": {"video": "C:/raw/session1.mp4", "fps": 30,
             "start_frame": 300, "end_frame_exclusive": 750},
  "annotation_status": "human_reviewed",
  "annotator": "actual annotator name",
  "frames": [
    {"frame": 300, "coverage": "full", "items": [
      {"bbox": [50, 80, 20, 45], "person_id": 1, "visibility": 0.7,
       "occlusion": "partial", "identity_uncertain": false, "ignore": false}
    ]}
  ]
}
```

The abbreviated example is **sparse**. Add every frame, including empty frames,
before tracking export. `annotation-template` produces empty **pending**, not GT,
records from a source descriptor. Human review is required to change the status.

## Scoring

Detection: full annotation of each scored frame is mandatory. Sparse fully labeled
frames may measure detection **on those frames only**. IoU >=0.5, maximum-cardinality
bipartite matching with descending-IoU edge visitation, deterministic index ties.
An unmatched detection overlapping a scored GT bbox at IoU >=0.5 is a duplicate,
counted as FP. A remaining unmatched prediction overlapping an ignored bbox at the
same threshold is excluded and reported. Other unmatched predictions are FP;
unmatched scored people are FN. Missing prediction records are an error, not zero
detections. This protocol is a diagnostic convention, not an official MOT score.

Tracking: use contiguous fully reviewed frames, with trustworthy identities and no
ignored/uncertain instances for the initial strict export. Split evaluation ranges
at unresolved identity intervals; report exclusions and duration. The exporter
refuses sparse or unresolved annotation. Use **official TrackEval**, not sparse
frame matching, for HOTA/DetA/AssA/IDF1/IDSW/Frag. Check any installed-version dataset
preprocessing (including distractors/ignore handling) before reporting numbers.
All versions, evaluator command, source/frame mappings and counts must be saved.

## Optional group reference

Only label clear intervals if needed to distinguish a downstream hypothesis.
Agree an observable group definition with LAPSCO: spatial proximity, joint activity
and long-term social relationship are different targets. Mark ambiguity as unknown.
Use GT person IDs, explicit member sets and intervals, and keep singletons explicit.
Do not derive labels from DBSCAN/proximity or model arrows. Review a small subset
with a second annotator if available. Orientation annotation is not required now.

Suggested separate interface (not consumed as person GT):

```json
{"reference_type": "human_group_membership", "definition": "agreed observable activity",
 "source_video": "C:/raw/session1.mp4", "intervals": [
   {"start_frame": 300, "end_frame_exclusive": 330,
    "groups": [[1, 2], [3]], "unknown_person_ids": [4], "reviewed": true}
 ]}
```

Membership precision/recall/F1 needs reliable groups **and** a fixed GT-to-estimated
identity matching protocol. Missing persons, FPs and duplicates must not silently
vanish from evaluation. Stability/split/merge needs continuous reference events;
numeric group-ID uniqueness alone is not membership stability. No group metric is
implemented/reported in this first session: reference definition is still open.
