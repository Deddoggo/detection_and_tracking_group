import cv2
import json
from pathlib import Path

import boxmot
import numpy as np
import torch
from boxmot.trackers.tracker_zoo import create_tracker
import boxmot

# --- 1) Input / output configuration ---
INPUT_VIDEO_PATH = "videos/WIN_20260326_10_00_11_Pro.mp4"
INPUT_JSON_PATH = "playground_results/detections.json"
OUTPUT_VIDEO_PATH = "playground_results/output_bytetrack_only.mp4"
OUTPUT_JSON_PATH = "playground_results/tracks_bytetrack_only.json"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Resolve BoxMOT config path robustly to avoid FileNotFoundError.
CONFIG_DIR = Path(boxmot.__file__).parent / "configs"

# --- 2) Initialize ByteTrack ---
print(f"Initializing ByteTrack on {DEVICE}...")
tracker = create_tracker(
    tracker_type="bytetrack",
    tracker_config=CONFIG_DIR / "trackers/bytetrack.yaml",
    reid_weights=None,
    device=DEVICE,
    half=True,
    per_class=False,
)

# Runtime overrides for this scene.
tracker.track_thresh = 0.25
tracker.track_buffer = 90
print(f"ByteTrack overrides: track_thresh={tracker.track_thresh}, buffer={tracker.track_buffer}")

with open(INPUT_JSON_PATH, "r") as f:
    all_detections = json.load(f)

# --- 3) Process video ---
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

all_tracks = {}
frame_idx = 0

print(f"Starting tracking on {total_frames} frames with ByteTrack...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_key = f"frame_{frame_idx:05d}"
    frame_dets = all_detections.get(frame_key, [])

    dets_array = []
    for det in frame_dets:
        x, y, w, h = det["bbox"]
        conf = det["conf"]

        # BoxMOT expects [x1, y1, x2, y2, conf, class_id].
        if conf > 0.15:
            dets_array.append([x, y, x + w, y + h, conf, 0])

    dets_array = np.array(dets_array) if dets_array else np.empty((0, 6))

    # ByteTrack update from geometry-only detections.
    tracks = tracker.update(dets_array, frame)

    # Save and draw tracks.
    frame_tracks = []
    if len(tracks) > 0:
        valid_tracks = tracks[tracks[:, 4] != -1]
        for trk in valid_tracks:
            x1, y1, x2, y2 = map(int, trk[:4])
            track_id = int(trk[4])

            # Keep output format consistent with detection step.
            frame_tracks.append({"id": track_id, "bbox": [x1, y1, x2 - x1, y2 - y1]})

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    all_tracks[frame_key] = frame_tracks
    out.write(frame)
    frame_idx += 1

    print(f"Tracked: {frame_idx}/{total_frames}", end="\r")

cap.release()
out.release()

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(all_tracks, f, indent=4)

print("\n\nTracking completed successfully.")
print(f"Video saved to: {OUTPUT_VIDEO_PATH}")
print(f"Track JSON saved to: {OUTPUT_JSON_PATH}")
