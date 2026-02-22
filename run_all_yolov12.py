import os
import subprocess

# --- CẤU HÌNH ĐƯỜNG DẪN TẠI ĐÂY ---
MOT15_TRAIN_DIR = "./MOT15/train"               
OUTPUT_TRACKER_DIR = "./output_yolo/data" # Tạo một ngăn kéo riêng biệt

os.makedirs(OUTPUT_TRACKER_DIR, exist_ok=True)
sequences = [d for d in os.listdir(MOT15_TRAIN_DIR) if os.path.isdir(os.path.join(MOT15_TRAIN_DIR, d))]

print(f"🔥 Bắt đầu chạy YOLOv12 + DeepOCSORT cho {len(sequences)} sequences...")

for seq in sequences:
    print(f"\n{'='*50}")
    print(f"🚀 ĐANG XỬ LÝ: {seq}")
    print(f"{'='*50}")
    
    # Định dạng %06d.jpg cho OpenCV đọc
    img_dir_pattern = os.path.join(MOT15_TRAIN_DIR, seq, "img1", "%06d.jpg")
    
    cmd = [
        "python", "tracking_yolov12.py", # Tên file rút gọn ở trên
        "--input_video", img_dir_pattern,
        "--output_folder", OUTPUT_TRACKER_DIR,
        "--seq_name", seq
    ]
    subprocess.run(cmd)

print("\n✅ HOÀN THÀNH TẤT CẢ! Thư mục kết quả nằm ở:", OUTPUT_TRACKER_DIR)