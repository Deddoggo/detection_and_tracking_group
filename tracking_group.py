import numpy as np
import cv2
import os
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.application_util import preprocessing


from identify_group import cluster_bboxes_with_ids
import argparse

# ===== CONFIGURATION =====
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
parser.add_argument('--frames_dir', type=str, required=True, help='Directory containing extracted frames')
parser.add_argument('--epsilon', type=float, default=50.0, help='DBSCAN epsilon value')
parser.add_argument('--threshold_overlap', type=float, default=0.7, help='Overlap threshold for old/new group determination')
args = parser.parse_args()

input_video = args.input_video
frames_dir = args.frames_dir
epsilon = args.epsilon
threshold_overlap = args.threshold_overlap

os.makedirs(frames_dir, exist_ok=True)

detection_file = os.path.join(frames_dir, "det.npy")
reid_model_path = "./deep_sort/resources/networks/mars-small128.pb"
video_name = os.path.splitext(os.path.basename(input_video))[0]  # "v2"

output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_video = f"{output_dir}/output_{video_name}_x101.mp4"

# ===== LOAD REID MODEL =====
max_cosine_distance = 0.4
nn_budget = 100
model = gdet.create_box_encoder(reid_model_path, batch_size=32)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# ===== LOAD DETECTIONS =====
detections = np.load(detection_file)

# ===== LOAD FRAME LIST =====
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
total_frames = len(frame_files)

# ===== VIDEO WRITER (for saving tracking video) =====
sample_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
h, w = sample_frame.shape[:2]


# Open video
cap = cv2.VideoCapture(input_video)

# Check if video can be opened
if not cap.isOpened():
    print("Cannot open video.")
else:
    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

# Release video
cap.release()
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# ===== TRACKING WITH DEEPSORT =====
groups_status = {}  # Store group status
max_group_id = -1   # Variable to track the highest group ID

import time
start_time = time.time()  # Bắt đầu đo thời gian xử lý frame

for frame_idx, frame_name in enumerate(frame_files):
    frame = cv2.imread(os.path.join(frames_dir, frame_name))
    current_frame_id = frame_idx + 1

    # Get all detections of the current frame
    frame_dets = detections[detections[:, 0] == current_frame_id]
    bboxes = frame_dets[:, 2:6]  # x, y, w, h
    scores = frame_dets[:, 6]

    features = model(frame, bboxes)
    det_objects = [Detection(bbox, score, feat) for bbox, score, feat in zip(bboxes, scores, features)]

    # Apply NMS
    boxes = np.array([d.tlwh for d in det_objects])
    scores = np.array([d.confidence for d in det_objects])
    indices = preprocessing.non_max_suppression(boxes, 0.8, scores)
    det_objects = [det_objects[i] for i in indices]

    # UPDATE TRACKER
    tracker.predict()
    tracker.update(det_objects)

    # Initialize list to store bboxes and track_ids
    track_ids = []
    bboxes = []

    # DRAW TRACKS
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlwh()
        bboxes.append(bbox)
        track_ids.append(track.track_id)

    if len(bboxes) >= 2:
        cluster_results, groups_status, max_group_id = cluster_bboxes_with_ids(
            groups_status, bboxes, track_ids, max_group_id, eps=epsilon, min_samples=2, threshold_overlap=threshold_overlap)

        # Draw bbox and info for each person
        for person in cluster_results:
            id_p = person['id_p']
            color = person['color']
            id_g = person['id_g']
            bbox = person['bbox']
            center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

            # Draw bbox
            cv2.rectangle(frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                        color, 1)

            # Draw center point
            center_x = int(center[0])
            center_y = int(center[1])
            cv2.circle(frame, (center_x, center_y), 3, color, -1)

            text = ""
            if id_g == -1:
                text = f"ID:{id_p} No group"
            else:
                text = f"ID:{id_p} G:{id_g}"

            cv2.putText(frame, text,
                        (center_x - 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, color, 1)
    # Display / save video
    out.write(frame)
    print(f"Tracking frame {current_frame_id}/{total_frames}", end='\r')

end_time = time.time()  # Kết thúc đo thời gian xử lý frame

out.release()
print("\n🎉 Tracking completed. Video saved at:", output_video)
print(f"Processed {total_frames} frames in {end_time - start_time:.2f} seconds.")
print(f"Average processing speed: {total_frames / (end_time - start_time):.2f} FPS.")
