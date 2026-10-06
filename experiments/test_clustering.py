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
from detectron2.data import MetadataCatalog
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# Adjust the path to the sibling deep_sort_pytorch directory.
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory path.
parent_dir = external_path()  # Get the parent directory path.
sys.path.append(parent_dir)  # Add the parent directory to sys.path.
# Deep SORT
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

# Configure Detectron2 with Mask R-CNN.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold.
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


# Initialize DeepSORT.
cfg_deep_sort = get_config()
cfg_deep_sort.merge_from_file(external_path("deep_sort_pytorch/configs/deep_sort.yaml"))
reid_model_path = external_path('deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7')


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

# Input image and result directories.
input_folder = "./input/v3"
output_folder = "./output/output_frames_cluster_v3"
os.makedirs(output_folder, exist_ok=True)

# DBSCAN parameters.
eps = 100  # Maximum distance between two samples considered to be in the same cluster.
min_samples = 2  # Minimum number of neighboring samples needed to form a cluster.

# Initialize the dictionary storing group information for each track_id.
track_to_group = {}  # {track_id: group_id}

# Create a set to record IDs already used.
used_group_ids = set()
# Variable storing the highest ID used so far.
max_group_id = -1

# Iterate over all images in the directory in order.
for idx, filename in enumerate(sorted(os.listdir(input_folder))):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Create a copy for drawing clustering results.
        image_with_clusters = image.copy()

        # Detect people using Detectron2.
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        class_ids = instances.pred_classes.numpy()

        # Keep person instances only (COCO class_id = 0).
        person_instances = instances[class_ids == 0]
        if len(person_instances) == 0:
            continue  # Skip if no people are detected.

        bbox_xyxy = person_instances.pred_boxes.tensor.cpu().numpy()  # (x1, y1, x2, y2)
        scores = person_instances.scores.cpu().numpy()
        classes = [0] * len(scores)  # All instances are people.

        # Convert to (cx, cy, w, h).
        bbox_xywh = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            bbox_xywh.append([cx, cy, w, h])
        bbox_xywh = np.array(bbox_xywh)

        # Update Deep SORT.
        tracking_outputs, _ = deepsort.update(bbox_xywh, scores, classes, image)
        
        # Retrieve center coordinates and track_id from tracking results.
        centers = []  # List of center points.
        track_ids = []  # List of corresponding track_ids.
        
        for out in tracking_outputs:
            x1, y1, x2, y2, cls_id, track_id = out
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])
            track_ids.append(track_id)
        
        # Cluster only if at least 2 people are detected.
        if len(centers) >= 2:
            # Perform DBSCAN clustering.
            centers_array = np.array(centers)
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_array)
            labels = clustering.labels_  # -1 denotes noise (not assigned to any group).
            
            # Number of detected groups, excluding noise.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Create a color palette for groups.
            colors = plt.cm.rainbow(np.linspace(0, 1, max(n_clusters, 1)))
            colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
            
            # Update group information for each track_id.
            current_groups = {}  # {group_id: [track_id1, track_id2, ...]}
            
            for i, (track_id, label) in enumerate(zip(track_ids, labels)):
                if label != -1:  # Skip noise points.
                    if label not in current_groups:
                        current_groups[label] = []
                    current_groups[label].append(track_id)
            
            # Build the reverse mapping: {group_id: set(track_ids)}.
            old_groups = defaultdict(set)
            for t_id, g_id in track_to_group.items():
                old_groups[g_id].add(t_id)
            
            # Record old group IDs in used_group_ids to prevent reuse.
            for g_id in old_groups.keys():
                used_group_ids.add(g_id)
                max_group_id = max(max_group_id, g_id)
            
            # Initialize a new track_to_group dictionary.
            new_track_to_group = {}
            
            # Process each current group.
            for current_group_id, members in current_groups.items():
                # Check whether this group continues an old group.
                best_match = None
                best_overlap = 0
                
                for old_group_id, old_members in old_groups.items():
                    # Compute overlap between the current and old groups.
                    overlap = len(set(members) & old_members)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = old_group_id
                
                # Reuse the old group_id if the overlap is substantial.
                # Criterion: at least 50% of the old members remain in the new group.
                if best_match is not None:
                    old_members = old_groups[best_match]
                    overlap_ratio = best_overlap / len(old_members) if old_members else 0
                    
                    if overlap_ratio >= 0.5:  # Retain the group if at least 50% of its old members remain.
                        for member in members:
                            new_track_to_group[member] = best_match
                    else:
                        # Create a new group_id for the new group.
                        max_group_id += 1
                        for member in members:
                            new_track_to_group[member] = max_group_id
                            used_group_ids.add(max_group_id)
                else:
                    # Create a new group_id for the new group.
                    max_group_id += 1
                    for member in members:
                        new_track_to_group[member] = max_group_id
                        used_group_ids.add(max_group_id)
            
            # Update the track_to_group dictionary.
            track_to_group = new_track_to_group
            
            # Draw clustering results.
            for i, (out, label) in enumerate(zip(tracking_outputs, labels)):
                x1, y1, x2, y2, cls_id, track_id = out
                cx, cy = centers[i]
                
                # Draw the bounding box.
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw track_id.
                cv2.putText(image, f'ID: {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw the center point.
                cv2.circle(image_with_clusters, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                
                # Draw the group if it is not a noise point.
                if track_id in track_to_group:
                    group_id = track_to_group[track_id]
                    color = colors[group_id % len(colors)]
                    
                    # Draw the bounding box in the group color.
                    cv2.rectangle(image_with_clusters, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw track_id and group_id information.
                    cv2.putText(image_with_clusters, f'ID: {track_id}, Group: {group_id}', 
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    # Draw a bounding box for a noise point.
                    cv2.rectangle(image_with_clusters, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(image_with_clusters, f'ID: {track_id}, No Group', 
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
            
           
            for group_id, members in current_groups.items():
                if len(members) >= 2:
                    # Get the actual group_id from track_to_group.
                    actual_group_id = track_to_group.get(members[0], -1)
                    color = colors[actual_group_id % len(colors)]
                    
                    # Get the indices of track_ids in the track_ids list.
                    member_indices = [track_ids.index(m) for m in members if m in track_ids]
                    
                    # Compute the group center.
                    center_points = [centers[idx] for idx in member_indices]
                    center_point = np.mean(center_points, axis=0)
                    center_point = (int(center_point[0]), int(center_point[1]))
                    
                    # Draw the center point.
                    cv2.circle(image_with_clusters, center_point, 8, color, -1)
                    
                    # Draw connections from the center to group members.
                    for idx in member_indices:
                        pt = (int(centers[idx][0]), int(centers[idx][1]))
                        cv2.line(image_with_clusters, center_point, pt, color, 2)
        else:
            # If only one person is present, skip clustering.
            for out in tracking_outputs:
                x1, y1, x2, y2, cls_id, track_id = out
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'ID: {track_id}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
                
                # Copy onto the image containing clusters.
                cv2.rectangle(image_with_clusters, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_clusters, f'ID: {track_id}, No Group', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 2)

        # Save the result.
        output_path = os.path.join(output_folder, filename)
        cluster_output_path = os.path.join(output_folder, f"cluster_{filename}")
        # cv2.imwrite(output_path, image)
        cv2.imwrite(cluster_output_path, image_with_clusters)
        print(f"✅ Đã xử lý {filename}: Phát hiện {len(tracking_outputs)} người, {len(set(track_to_group.values()))} nhóm")

print("Hoàn thành xử lý tất cả các frame!")