import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import os
import argparse
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

import time

# Cấu hình Detectron2 với Mask R-CNN
cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # confidence threshold
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

# Automatically detect if GPU is available
if torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cuda"
    print("Using GPU for inference.")
else:
    cfg.MODEL.DEVICE = "cpu"
    print("No GPU found, using CPU for inference.")
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Create parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder')
args = parser.parse_args()

input_video = args.input_video
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

# Read input video
cap = cv2.VideoCapture(input_video)
frame_idx = 0
total_infer_time = 0
num_frames = 0

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()
# Create a list to store all detections
all_detections = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    try:
        start_time = time.time()  # Bắt đầu đo thời gian

        # Run Detectron2 on the frame
        outputs = predictor(frame)

        infer_time = time.time() - start_time  # Kết thúc đo thời gian
        total_infer_time += infer_time
        num_frames += 1

        # Get instances from outputs and move to CPU
        instances = outputs["instances"].to("cpu")

        # Filter only person (class_id = 0)
        class_person = instances.pred_classes == 0
        instances = instances[class_person]

        # Get bounding boxes and scores of persons
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        # Save detections for the current frame
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(float, box)  # Convert to float
            w = x2 - x1  # calculate width
            h = y2 - y1  # calculate height
            # Format for DeepSORT: [frame_id, -1, bb_left, bb_top, width, height, conf]
            # frame_id starts from 1
            detection = np.array([frame_idx + 1, -1, x1, y1, w, h, score])
            all_detections.append(detection)

        # Save the processed frame to the output folder
        output_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(output_path, frame)  # Save the frame with bbox drawn
        frame_idx += 1
        
        print(f"Processed frame {frame_idx}", end='\r')
        
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {str(e)}")
        continue

cap.release()

# Convert list to numpy array and save
all_detections = np.array(all_detections)
det_path = os.path.join(output_folder, 'det.npy')  # rename file to match DeepSORT
np.save(det_path, all_detections)
print(f"\nSaved detections to {det_path}")
print(f"Shape of detections: {all_detections.shape}")

if num_frames > 0:
    avg_infer_time = total_infer_time / num_frames
    fps = 1.0 / avg_infer_time
    print(f"Average inference time per frame: {avg_infer_time:.4f} seconds")
    print(f"Average FPS: {fps:.2f}")
# Format of each line: [frame_id, track_id, bb_left, bb_top, width, height, confidence]