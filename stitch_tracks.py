import cv2
import json
import math

import numpy as np
from tqdm import tqdm

# --- 1) Stitching configuration (relaxed for playground scenes) ---
INPUT_VIDEO = "videos/WIN_20260326_10_00_11_Pro.mp4"
INPUT_JSON = "playground_results/tracks_bytetrack_only.json"
OUTPUT_JSON = "playground_results/tracks_stitched.json"
OUTPUT_VIDEO = "playground_results/output_stitched.mp4"

MAX_TIME_LOST = 150      # 5 seconds at 30 FPS
MAX_SPATIAL_DIST = 350.0 # Maximum center-to-center distance in pixels
MIN_SIMILARITY = 0.60    # Histogram similarity threshold

DEBUG_LOGS = True


def get_center(bbox):
    """Return bbox center from [x, y, w, h]."""
    return bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0


def extract_color_histogram(image, bbox):
    """Extract center-focused HSV histogram to reduce background noise."""
    x, y, w, h = map(int, bbox)
    x, y = max(0, x), max(0, y)

    crop = image[y:y + h, x:x + w]
    if crop.size == 0 or w < 10 or h < 10:
        return None

    # Keep the central body region: less background, more appearance signal.
    cx_min, cx_max = int(w * 0.25), int(w * 0.75)
    cy_min, cy_max = int(h * 0.15), int(h * 0.85)
    center_crop = crop[cy_min:cy_max, cx_min:cx_max]
    if center_crop.size == 0:
        return None

    hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def compute_similarity(hist1, hist2):
    """Convert Bhattacharyya distance to similarity in [0, 1]."""
    if hist1 is None or hist2 is None:
        return 0.0
    dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - dist)


print("Loading tracking data...")
with open(INPUT_JSON, "r") as f:
    tracks_data = json.load(f)

# --- 2) Pass through video to extract per-track metadata + appearance ---
cap = cv2.VideoCapture(INPUT_VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

track_meta = {}
track_hists = {}

print(f"Step 1/4: extracting color histograms from {total_frames} frames...")
frame_idx = 0
pbar = tqdm(total=total_frames)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_key = f"frame_{frame_idx:05d}"
    frame_tracks = tracks_data.get(frame_key, [])

    for trk in frame_tracks:
        tid = trk["id"]
        bbox = trk["bbox"]

        if tid not in track_meta:
            track_meta[tid] = {
                "start": frame_idx,
                "first_box": bbox,
                "end": frame_idx,
                "last_box": bbox,
            }
            track_hists[tid] = []
        else:
            track_meta[tid]["end"] = frame_idx
            track_meta[tid]["last_box"] = bbox

        # Sample every few frames for speed.
        if frame_idx % 5 == 0:
            hist = extract_color_histogram(frame, bbox)
            if hist is not None:
                track_hists[tid].append(hist)

    frame_idx += 1
    pbar.update(1)

cap.release()
pbar.close()

avg_hists = {}
for tid, hists in track_hists.items():
    if hists:
        avg_hist = np.mean(hists, axis=0)
        cv2.normalize(avg_hist, avg_hist, 0, 1, cv2.NORM_MINMAX)
        avg_hists[tid] = avg_hist
    else:
        avg_hists[tid] = None

# --- 3) Tracklet stitching ---
print("\nStep 2/4: analyzing and stitching tracklets...")
sorted_tids = sorted(track_meta.keys(), key=lambda x: track_meta[x]["start"])

id_mapping = {}
active_tracks = []

for new_tid in sorted_tids:
    meta = track_meta[new_tid]
    start_f = meta["start"]
    first_center = get_center(meta["first_box"])
    new_hist = avg_hists.get(new_tid)

    best_match_id = None
    best_similarity = MIN_SIMILARITY

    for old_tid in list(active_tracks):
        old_meta = track_meta[old_tid]
        end_f = old_meta["end"]

        if start_f - end_f > MAX_TIME_LOST:
            active_tracks.remove(old_tid)
            continue

        if start_f <= end_f:
            continue

        last_center = get_center(old_meta["last_box"])
        dist = math.hypot(first_center[0] - last_center[0], first_center[1] - last_center[1])

        if dist > MAX_SPATIAL_DIST:
            continue

        old_hist = avg_hists.get(old_tid)
        sim = compute_similarity(old_hist, new_hist)

        if DEBUG_LOGS:
            print(f"[DEBUG] ID_{old_tid} -> ID_{new_tid} | dist={dist:.1f}px | sim={sim:.2f}")

        if sim > best_similarity:
            best_similarity = sim
            best_match_id = old_tid

    if best_match_id is not None:
        root_id = best_match_id
        while root_id in id_mapping:
            root_id = id_mapping[root_id]

        id_mapping[new_tid] = root_id
        track_meta[root_id]["end"] = meta["end"]
        track_meta[root_id]["last_box"] = meta["last_box"]

        if best_match_id in active_tracks:
            active_tracks.remove(best_match_id)
        active_tracks.append(root_id)
        print(f"Linked: ID_{new_tid} -> ID_{root_id} (confidence={best_similarity:.2f})")
    else:
        active_tracks.append(new_tid)

print(f"\nSummary: stitched {len(id_mapping)} track-ID pairs.")

# --- 4) Save stitched JSON and render video ---
print("Step 3/4: writing stitched JSON...")
stitched_data = {}

for frame_key, tracks in tracks_data.items():
    new_frame_tracks = []
    for trk in tracks:
        tid = trk["id"]
        bbox = trk["bbox"]

        final_id = tid
        while final_id in id_mapping:
            final_id = id_mapping[final_id]

        new_frame_tracks.append({"id": final_id, "bbox": bbox})

    stitched_data[frame_key] = new_frame_tracks

with open(OUTPUT_JSON, "w") as f:
    json.dump(stitched_data, f, indent=4)
print(f"Stitched JSON saved to: {OUTPUT_JSON}")

print("Step 4/4: rendering stitched video...")
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

frame_idx = 0
pbar_render = tqdm(total=total_frames)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_key = f"frame_{frame_idx:05d}"
    frame_tracks = stitched_data.get(frame_key, [])

    for trk in frame_tracks:
        tid = trk["id"]
        x, y, w, h = trk["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    out.write(frame)
    frame_idx += 1
    pbar_render.update(1)

cap.release()
out.release()
pbar_render.close()

print(f"\nAll done. Final stitched video saved to: {OUTPUT_VIDEO}")
