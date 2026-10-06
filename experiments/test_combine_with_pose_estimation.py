# Allow both python -m PACKAGE.NAME and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import repo_path, checkpoint_path, external_path, output_path, load_paths
import sys as _dependency_sys
_dependency_sys.path.insert(0, external_path())
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import torch
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import sys
import time
from collections import deque


# Adjust the path to the sibling deep_sort_pytorch directory.
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory path.
parent_dir = external_path()  # Get the parent directory path.
sys.path.append(parent_dir)  # Add the parent directory to sys.path.
hrnet_w32 = os.path.join(parent_dir, "HRNet-Human-Pose-Estimation")
sys.path.append(hrnet_w32)


# from hrnet_td import get_pose_from_bbox_fixed, 
from hrnet_test import HRNetPoseDetector

hrnet_detector = HRNetPoseDetector()
# Import DeepSORT
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# Update the configuration-file path.
cfg_deep_sort = get_config()
cfg_deep_sort.merge_from_file(os.path.join(parent_dir, "deep_sort_pytorch/configs/deep_sort.yaml"))

# Update the checkpoint-file path.
reid_model_path = os.path.join(parent_dir, "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

# ...existing code...

# Check whether the checkpoint file exists.
if not os.path.exists(reid_model_path):
    print(f"CẢNH BÁO: Không tìm thấy file checkpoint tại {reid_model_path}")
    print("Vui lòng tải xuống file checkpoint và đặt nó vào đúng đường dẫn")
    print("Bạn có thể tải từ: https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6")
    sys.exit(1)

# Initialize the DeepSORT tracker.
deepsort = DeepSort(
    reid_model_path,  # Use a direct path rather than a path from the configuration file.
    max_dist=cfg_deep_sort.DEEPSORT.MAX_DIST,
    min_confidence=cfg_deep_sort.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg_deep_sort.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg_deep_sort.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg_deep_sort.DEEPSORT.MAX_AGE,
    n_init=cfg_deep_sort.DEEPSORT.N_INIT,
    nn_budget=cfg_deep_sort.DEEPSORT.NN_BUDGET,
    use_cuda=True
)

# Configure Detectron2 (Mask R-CNN).
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # confidence threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Function converting Detectron2 bounding boxes to DeepSORT format.
def xyxy_to_xywh(bbox):
    """Convert bbox from [x1, y1, x2, y2] to [x, y, w, h].
    x, y are the top-left corner coordinates.
    w, h are width and height."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return [cx, cy, w, h]

# Input image and result directories.
input_folder = "./input/v3"  # Adapt to your own path.
output_folder = "./output/pose_output_frames_v3"
os.makedirs(output_folder, exist_ok=True)

# Save the output video.
video_output_path = "./output/pose_tracked_video_v3.mp4"
fps = 20  # Frames per second (adjust as needed).
frame_size = None
out = None

# Dictionary storing object trajectories for drawing paths.
trajectories = {}
max_trajectory_length = 30  # Maximum trajectory length.

# Process frames in order.
sorted_files = sorted(os.listdir(input_folder))
image_files = [f for f in sorted_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for frame_idx, filename in enumerate(image_files):
    start_time = time.time()
    
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    
    if frame_size is None:
        frame_size = (image.shape[1], image.shape[0])
        out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    # 1. Detect people using Detectron2.
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    # Filter person bounding boxes (COCO class ID 0).
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    # Keep person instances only (COCO class_id=0).
    person_indices = np.where(classes == 0)[0]
    person_boxes = boxes[person_indices]
    person_scores = scores[person_indices]
    
    # If the frame contains no person, save only the original frame.
    if len(person_boxes) == 0:
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
        out.write(image)
        print(f"Frame {frame_idx+1}/{len(image_files)}: Không phát hiện người")
        continue
    
    # 2. Convert bounding-box format for DeepSORT.
    xywh_boxes = [xyxy_to_xywh(box) for box in person_boxes]
    
    # 3. Update the DeepSORT tracker.
    # Create class IDs for all bounding boxes (all class 0 - person).
    person_classes = np.zeros(len(xywh_boxes), dtype=int)
    
    # Call the new update method with the additional classes argument.
    tracks, mask_outputs = deepsort.update(np.array(xywh_boxes), person_scores, person_classes, image)
    
    # 4. Draw tracking results.
    vis_image = image.copy()
    
    # Check whether any tracks were returned.
    if len(tracks) == 0:
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
        out.write(image)
        print(f"Frame {frame_idx+1}/{len(image_files)}: Không có track nào được cập nhật")
        continue
        
    for track in tracks:
        # New version: x1, y1, x2, y2, class_id, track_id.
        track_cls = int(track[4])
        track_id = int(track[5])
        bbox = track[:4]
        x1, y1, x2, y2 = bbox
        #Get p2d confidence theo hm36
        p2d, confidence = hrnet_detector.detect_pose_from_bbox(image, bbox)
        hrnet_detector.visualize_pose(vis_image, p2d, confidence)
        # Draw the bounding box.
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Display object ID and class.
        text = f"ID: {track_id} (Person)"
        cv2.putText(vis_image, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # # Compute the trajectory center.
        # width = x2 - x1
        # height = y2 - y1
        # center = (int(x1 + width/2), int(y1 + height/2))
        
        # # Update the trajectory for this ID.
        # if track_id not in trajectories:
        #     trajectories[track_id] = deque(maxlen=max_trajectory_length)
        # trajectories[track_id].append(center)
        
        # # Draw the trajectory.
        # color = (0, 0, 255)  # Red for the trajectory.
        # pts = list(trajectories[track_id])
        # for i in range(1, len(pts)):
        #     if pts[i-1] is None or pts[i] is None:
        #         continue
        #     cv2.line(vis_image, pts[i-1], pts[i], color, thickness=2)
    
    # Display information.
    fps_text = f"FPS: {1/(time.time() - start_time):.2f}"
    cv2.putText(vis_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    count_text = f"Persons: {len(tracks)}"
    cv2.putText(vis_image, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Save the result.
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, vis_image)
    out.write(vis_image)
    print(f"Frame {frame_idx+1}/{len(image_files)}: Đã phát hiện {len(tracks)} người, FPS: {1/(time.time() - start_time):.2f}")

# Release resources.
if out is not None:
    out.release()

print(f"✅ Đã hoàn tất xử lý tất cả ảnh trong thư mục và lưu video tại {video_output_path}")