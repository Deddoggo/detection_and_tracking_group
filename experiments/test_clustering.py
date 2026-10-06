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

# Sửa đường dẫn để trỏ đến thư mục deep_sort_pytorch ở cùng cấp
current_dir = os.path.dirname(os.path.abspath(__file__))  # Lấy đường dẫn thư mục hiện tại
parent_dir = external_path()  # Lấy đường dẫn thư mục cha
sys.path.append(parent_dir)  # Thêm thư mục cha vào sys.path
# Deep SORT
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

# Cấu hình Detectron2 với Mask R-CNN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # ngưỡng confidence
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


# Khởi tạo DeepSORT
cfg_deep_sort = get_config()
cfg_deep_sort.merge_from_file(external_path("deep_sort_pytorch/configs/deep_sort.yaml"))
reid_model_path = external_path('deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7')


# Khởi tạo tracker DeepSORT
deepsort = DeepSort(
    reid_model_path,  # Sử dụng đường dẫn trực tiếp thay vì từ file cấu hình
    max_dist=cfg_deep_sort.DEEPSORT.MAX_DIST,
    min_confidence=cfg_deep_sort.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg_deep_sort.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg_deep_sort.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg_deep_sort.DEEPSORT.MAX_AGE,
    n_init=cfg_deep_sort.DEEPSORT.N_INIT,
    nn_budget=cfg_deep_sort.DEEPSORT.NN_BUDGET,
    use_cuda=True
)

# Thư mục chứa ảnh đầu vào và thư mục kết quả
input_folder = "./input/v3"
output_folder = "./output/output_frames_cluster_v3"
os.makedirs(output_folder, exist_ok=True)

# Tham số cho DBSCAN
eps = 100  # Khoảng cách tối đa giữa hai mẫu để được coi là trong cùng một cụm
min_samples = 2  # Số mẫu tối thiểu trong vùng lân cận để hình thành một cụm

# Khởi tạo từ điển lưu trữ thông tin nhóm cho từng track_id
track_to_group = {}  # {track_id: group_id}

# Tạo tập hợp (set) để lưu trữ các ID đã được sử dụng
used_group_ids = set()
# Biến lưu giá trị ID lớn nhất đã được sử dụng
max_group_id = -1

# Duyệt qua tất cả các ảnh trong thư mục (theo thứ tự)
for idx, filename in enumerate(sorted(os.listdir(input_folder))):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Tạo bản sao để vẽ kết quả clustering
        image_with_clusters = image.copy()

        # Phát hiện người bằng Detectron2
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        class_ids = instances.pred_classes.numpy()

        # Lọc ra các instance là người (class_id = 0 trong COCO)
        person_instances = instances[class_ids == 0]
        if len(person_instances) == 0:
            continue  # bỏ qua nếu không có người

        bbox_xyxy = person_instances.pred_boxes.tensor.cpu().numpy()  # (x1, y1, x2, y2)
        scores = person_instances.scores.cpu().numpy()
        classes = [0] * len(scores)  # tất cả đều là người

        # Chuyển sang (cx, cy, w, h)
        bbox_xywh = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            bbox_xywh.append([cx, cy, w, h])
        bbox_xywh = np.array(bbox_xywh)

        # Cập nhật Deep SORT
        tracking_outputs, _ = deepsort.update(bbox_xywh, scores, classes, image)
        
        # Lấy ra tọa độ trung tâm và track_id từ kết quả tracking
        centers = []  # Danh sách các điểm trung tâm
        track_ids = []  # Danh sách các track_id tương ứng
        
        for out in tracking_outputs:
            x1, y1, x2, y2, cls_id, track_id = out
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])
            track_ids.append(track_id)
        
        # Chỉ thực hiện clustering nếu có ít nhất 2 người được phát hiện
        if len(centers) >= 2:
            # Thực hiện clustering với DBSCAN
            centers_array = np.array(centers)
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_array)
            labels = clustering.labels_  # -1 là nhiễu (không thuộc nhóm nào)
            
            # Số nhóm được phát hiện (không tính nhiễu)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Tạo bảng màu cho các nhóm
            colors = plt.cm.rainbow(np.linspace(0, 1, max(n_clusters, 1)))
            colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
            
            # Cập nhật thông tin nhóm cho mỗi track_id
            current_groups = {}  # {group_id: [track_id1, track_id2, ...]}
            
            for i, (track_id, label) in enumerate(zip(track_ids, labels)):
                if label != -1:  # Bỏ qua các điểm nhiễu
                    if label not in current_groups:
                        current_groups[label] = []
                    current_groups[label].append(track_id)
            
            # Tạo dict ngược để kiểm tra: {group_id: set(track_ids)}
            old_groups = defaultdict(set)
            for t_id, g_id in track_to_group.items():
                old_groups[g_id].add(t_id)
            
            # Lưu ID nhóm cũ vào tập hợp used_group_ids để không tái sử dụng
            for g_id in old_groups.keys():
                used_group_ids.add(g_id)
                max_group_id = max(max_group_id, g_id)
            
            # Khởi tạo từ điển mới cho track_to_group
            new_track_to_group = {}
            
            # Xử lý từng nhóm hiện tại
            for current_group_id, members in current_groups.items():
                # Kiểm tra xem nhóm này có phải là sự kế thừa của nhóm cũ không
                best_match = None
                best_overlap = 0
                
                for old_group_id, old_members in old_groups.items():
                    # Tính độ chồng lấn giữa nhóm hiện tại và nhóm cũ
                    overlap = len(set(members) & old_members)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = old_group_id
                
                # Nếu có sự chồng lấn đáng kể, sử dụng group_id cũ
                # Tiêu chí: có ít nhất 50% số thành viên của nhóm cũ vẫn còn trong nhóm mới
                if best_match is not None:
                    old_members = old_groups[best_match]
                    overlap_ratio = best_overlap / len(old_members) if old_members else 0
                    
                    if overlap_ratio >= 0.5:  # Giữ nguyên group nếu có ít nhất 50% thành viên cũ
                        for member in members:
                            new_track_to_group[member] = best_match
                    else:
                        # Tạo group_id mới cho nhóm mới
                        max_group_id += 1
                        for member in members:
                            new_track_to_group[member] = max_group_id
                            used_group_ids.add(max_group_id)
                else:
                    # Tạo group_id mới cho nhóm mới
                    max_group_id += 1
                    for member in members:
                        new_track_to_group[member] = max_group_id
                        used_group_ids.add(max_group_id)
            
            # Cập nhật từ điển track_to_group
            track_to_group = new_track_to_group
            
            # Vẽ kết quả clustering
            for i, (out, label) in enumerate(zip(tracking_outputs, labels)):
                x1, y1, x2, y2, cls_id, track_id = out
                cx, cy = centers[i]
                
                # Vẽ bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ track_id
                cv2.putText(image, f'ID: {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Vẽ điểm trung tâm
                cv2.circle(image_with_clusters, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                
                # Vẽ nhóm nếu không phải điểm nhiễu
                if track_id in track_to_group:
                    group_id = track_to_group[track_id]
                    color = colors[group_id % len(colors)]
                    
                    # Vẽ bounding box với màu của nhóm
                    cv2.rectangle(image_with_clusters, (x1, y1), (x2, y2), color, 2)
                    
                    # Vẽ thông tin track_id và group_id
                    cv2.putText(image_with_clusters, f'ID: {track_id}, Group: {group_id}', 
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    # Vẽ bounding box cho điểm nhiễu
                    cv2.rectangle(image_with_clusters, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(image_with_clusters, f'ID: {track_id}, No Group', 
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
            
           
            for group_id, members in current_groups.items():
                if len(members) >= 2:
                    # Lấy group_id thật từ track_to_group
                    actual_group_id = track_to_group.get(members[0], -1)
                    color = colors[actual_group_id % len(colors)]
                    
                    # Lấy chỉ số của các track_id trong danh sách track_ids
                    member_indices = [track_ids.index(m) for m in members if m in track_ids]
                    
                    # Tính điểm trung tâm của nhóm
                    center_points = [centers[idx] for idx in member_indices]
                    center_point = np.mean(center_points, axis=0)
                    center_point = (int(center_point[0]), int(center_point[1]))
                    
                    # Vẽ điểm trung tâm
                    cv2.circle(image_with_clusters, center_point, 8, color, -1)
                    
                    # Vẽ đường nối từ điểm trung tâm đến các thành viên
                    for idx in member_indices:
                        pt = (int(centers[idx][0]), int(centers[idx][1]))
                        cv2.line(image_with_clusters, center_point, pt, color, 2)
        else:
            # Nếu chỉ có một người, không thực hiện clustering
            for out in tracking_outputs:
                x1, y1, x2, y2, cls_id, track_id = out
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'ID: {track_id}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
                
                # Sao chép lên ảnh với clusters
                cv2.rectangle(image_with_clusters, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_clusters, f'ID: {track_id}, No Group', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 2)

        # Lưu kết quả
        output_path = os.path.join(output_folder, filename)
        cluster_output_path = os.path.join(output_folder, f"cluster_{filename}")
        # cv2.imwrite(output_path, image)
        cv2.imwrite(cluster_output_path, image_with_clusters)
        print(f"✅ Đã xử lý {filename}: Phát hiện {len(tracking_outputs)} người, {len(set(track_to_group.values()))} nhóm")

print("Hoàn thành xử lý tất cả các frame!")