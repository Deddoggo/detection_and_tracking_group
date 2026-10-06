# Allow both python -m PACKAGE.NAME and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import repo_path, checkpoint_path, external_path, output_path, load_paths
import sys as _dependency_sys
_dependency_sys.path.insert(0, external_path())
import sys
import os
import subprocess

# --- CONFIGURE YOUR PATHS HERE ---
MOT15_TRAIN_DIR = repo_path('./MOT15/train')               # Path to the training directory containing 11 folders.
OUTPUT_TRACKER_DIR = repo_path('./output_detectron2/data') # Destination directory for .txt files.
TEMP_FRAMES_DIR = repo_path('./temp_frames')               # Temporary directory for images and det.npy from step 1.

os.makedirs(OUTPUT_TRACKER_DIR, exist_ok=True)

# Get sequence names (ADL-Rundle-6, ETH-Bahnhof, ...).
sequences = [d for d in os.listdir(MOT15_TRAIN_DIR) if os.path.isdir(os.path.join(MOT15_TRAIN_DIR, d))]

print(f"🔥 Tìm thấy {len(sequences)} sequences. Bắt đầu chạy hàng loạt...")

for seq in sequences:
    print(f"\n{'='*50}")
    print(f"🚀 ĐANG XỬ LÝ SEQUENCE: {seq}")
    print(f"{'='*50}")
    
    img_dir_pattern = os.path.join(MOT15_TRAIN_DIR, seq, "img1", "%06d.jpg")
    seq_temp_dir = os.path.join(TEMP_FRAMES_DIR, seq)
    os.makedirs(seq_temp_dir, exist_ok=True)
    
    # 1. RUN DETECTION
    print(f"⏳ Bước 1/2: Chạy Detectron2 cho {seq}...")
    cmd_detect = [
        sys.executable, "-m", "legacy.detection",
        "--input_video", img_dir_pattern,
        "--output_folder", seq_temp_dir
    ]
    subprocess.run(cmd_detect)
    
    # 2. RUN TRACKING
    print(f"⏳ Bước 2/2: Chạy DeepSORT cho {seq}...")
    cmd_track = [
        sys.executable, "-m", "evaluation.eval_tracking_group",
        "--input_video", img_dir_pattern,
        "--frames_dir", seq_temp_dir,
        "--seq_name", seq,
        "--output_folder", OUTPUT_TRACKER_DIR
    ]
    subprocess.run(cmd_track)

print("\n✅ HOÀN THÀNH TẤT CẢ! Kiểm tra kết quả trong folder:", OUTPUT_TRACKER_DIR)