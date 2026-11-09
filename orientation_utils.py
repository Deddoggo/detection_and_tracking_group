import numpy as np
import math

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g); yi1 = max(y1, y1g)
    xi2 = min(x2, x2g); yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_pose_vector(keypoints, conf_thresh=0.5):
    """
    Tính vector hướng nhìn kết hợp (Ensemble):
    1. Hướng Vai (Body Orientation) - Ổn định nhất
    2. Hướng Mặt (Face Orientation) - Chi tiết nhất
    """
    if keypoints is None: return None

    if hasattr(keypoints, 'cpu'): kpts = keypoints.cpu().numpy()
    else: kpts = keypoints
    kpts = np.asarray(kpts)
    if not np.issubdtype(kpts.dtype, np.floating):
        kpts = kpts.astype(float)
    if kpts.shape == (0,): return None
    if kpts.ndim != 2 or kpts.shape[0] < 7 or kpts.shape[1] != 3:
        raise ValueError("keypoints must have shape (K>=7, 3): x, y, confidence")
    if not np.isfinite(kpts).all():
        raise ValueError("keypoints must be finite; use None for an unavailable pose")
    if not np.isfinite(conf_thresh) or not 0 <= conf_thresh <= 1:
        raise ValueError("conf_thresh must be finite and in [0, 1]")

    # Indices: 0:Nose, 1:L-Eye, 2:R-Eye, 3:L-Ear, 4:R-Ear, 5:L-Sh, 6:R-Sh
    nose = kpts[0]
    l_ear = kpts[3]; r_ear = kpts[4]
    l_sh = kpts[5];  r_sh = kpts[6]

    vectors = []
    weights = []

    # --- NGUỒN 1: VECTOR PHÁP TUYẾN VAI (BODY) ---
    # Đường nối 2 vai
    if l_sh[2] > conf_thresh and r_sh[2] > conf_thresh:
        # Vector vai: Trái -> Phải
        shoulder_vec = r_sh[:2] - l_sh[:2]
        # Vector pháp tuyến (vuông góc): (-y, x) -> Chỉ về phía trước ngực
        # Lưu ý: Hệ toạ độ ảnh y tăng xuống dưới.
        # Vai trái ở bên phải ảnh (nếu đối diện). Logic này cần kiểm chứng thực tế.
        # Thông thường: Pháp tuyến = (dy, -dx)
        normal_vec = np.array([shoulder_vec[1], -shoulder_vec[0]])

        # Chuẩn hóa
        norm = np.linalg.norm(normal_vec)
        if norm > 0:
            vectors.append(normal_vec / norm)
            weights.append(1.0) # Trọng số Body: 1.0 (Rất tin cậy)

    # --- NGUỒN 2: VECTOR TAI -> MŨI (HEAD) ---
    if nose[2] > conf_thresh and l_ear[2] > conf_thresh and r_ear[2] > conf_thresh:
        mid_ear = (l_ear[:2] + r_ear[:2]) / 2
        head_vec = nose[:2] - mid_ear

        norm = np.linalg.norm(head_vec)
        if norm > 0:
            vectors.append(head_vec / norm)
            weights.append(1.5) # Trọng số Đầu: 1.5 (Ưu tiên hướng quay đầu hơn người)

    # --- TỔNG HỢP (WEIGHTED AVERAGE) ---
    if not vectors:
        # Fallback: Nếu không có cả vai lẫn tai, thử dùng Vai -> Mũi (như cũ)
        if nose[2] > conf_thresh and l_sh[2] > conf_thresh and r_sh[2] > conf_thresh:
             mid_sh = (l_sh[:2] + r_sh[:2]) / 2
             fallback_vec = nose[:2] - mid_sh
             return _post_process_vector(fallback_vec)
        return None

    # Tính trung bình có trọng số
    avg_vec = np.average(vectors, axis=0, weights=weights)

    return _post_process_vector(avg_vec)

def _post_process_vector(vec):
    """Xử lý hậu kỳ để chống lỗi 'chỉ lên trời'"""
    vec = np.array(vec, copy=True)
    if not np.issubdtype(vec.dtype, np.floating):
        vec = vec.astype(float)
    if vec.shape != (2,) or not np.isfinite(vec).all():
        raise ValueError("orientation vector must be finite and have shape (2,)")
    # 1. Ép trục Y (Perspective Correction)
    # Giảm thành phần Y đi 50% để vector có xu hướng chỉ ngang nhiều hơn
    # Khắc phục lỗi camera từ trên cao
    vec[1] *= 0.5

    # 2. Chuẩn hóa lại
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm):
        raise ValueError("orientation vector norm must be finite")
    if norm < 0.1: return None # Vector quá ngắn (nhìn thẳng cam) -> Bỏ

    return vec / norm

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None: return 0
    return np.dot(v1, v2)
