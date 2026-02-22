import os
import subprocess

# --- CẤU HÌNH ĐƯỜNG DẪN CỦA BẠN TẠI ĐÂY ---
MOT15_TRAIN_DIR = "./MOT15/train"               # Đường dẫn tới thư mục train chứa 11 folder
OUTPUT_TRACKER_DIR = "./output_detectron2/data" # Thư mục đích chứa các file .txt
TEMP_FRAMES_DIR = "./temp_frames"               # Thư mục tạm để lưu ảnh và det.npy của bước 1

os.makedirs(OUTPUT_TRACKER_DIR, exist_ok=True)

# Lấy danh sách tên các sequence (ADL-Rundle-6, ETH-Bahnhof,...)
sequences = [d for d in os.listdir(MOT15_TRAIN_DIR) if os.path.isdir(os.path.join(MOT15_TRAIN_DIR, d))]

print(f"🔥 Tìm thấy {len(sequences)} sequences. Bắt đầu chạy hàng loạt...")

for seq in sequences:
    print(f"\n{'='*50}")
    print(f"🚀 ĐANG XỬ LÝ SEQUENCE: {seq}")
    print(f"{'='*50}")
    
    img_dir_pattern = os.path.join(MOT15_TRAIN_DIR, seq, "img1", "%06d.jpg")
    seq_temp_dir = os.path.join(TEMP_FRAMES_DIR, seq)
    os.makedirs(seq_temp_dir, exist_ok=True)
    
    # 1. CHẠY DETECTION
    print(f"⏳ Bước 1/2: Chạy Detectron2 cho {seq}...")
    cmd_detect = [
        "python", "detection.py",
        "--input_video", img_dir_pattern,
        "--output_folder", seq_temp_dir
    ]
    subprocess.run(cmd_detect)
    
    # 2. CHẠY TRACKING
    print(f"⏳ Bước 2/2: Chạy DeepSORT cho {seq}...")
    cmd_track = [
        "python", "eval_tracking_group.py",
        "--input_video", img_dir_pattern,
        "--frames_dir", seq_temp_dir,
        "--seq_name", seq,
        "--output_folder", OUTPUT_TRACKER_DIR
    ]
    subprocess.run(cmd_track)

print("\n✅ HOÀN THÀNH TẤT CẢ! Kiểm tra kết quả trong folder:", OUTPUT_TRACKER_DIR)