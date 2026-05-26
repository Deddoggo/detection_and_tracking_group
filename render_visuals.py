import cv2
import json
import numpy as np
import math

# --- CONFIG ---
INPUT_VIDEO = "videos/WIN_20260326_10_00_11_Pro.mp4"
INPUT_JSON = "playground_results/final_sociometrics_data.json"
OUTPUT_VIDEO = "playground_results/output_visualized_clean.mp4"
FRAME_STRIDE = 3

MAX_DRAW_DIST = 4.0 # Ngưỡng vẽ đường nối khoảng cách (mét)
CONE_LENGTH = 50
CONE_ANGLE = 40

# --- THEME TỐI GIẢN (MINIMALIST) ---
COLOR_MAIN = (200, 200, 200)      # Xám nhạt cho Box
COLOR_GROUP_LINE = (255, 200, 100) # Xanh Cyan nhạt cho đường nối nhóm
COLOR_TEXT_F = (220, 150, 220)    # Hồng nhạt cho Nữ
COLOR_TEXT_M = (150, 200, 220)    # Xanh lơ nhạt cho Nam
COLOR_CONE = (255, 255, 255)      # Trắng mờ cho nón nhìn

def interpolate_bbox(bbox1, bbox2, alpha):
    return [b1 + (b2 - b1) * alpha for b1, b2 in zip(bbox1, bbox2)]

print("Đang tải dữ liệu...")
with open(INPUT_JSON, 'r') as f:
    frames_data = json.load(f)

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"🚀 Đang render đồ họa tối giản {total_frames} frames...")

for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret: break

    f_start = (frame_idx // FRAME_STRIDE) * FRAME_STRIDE
    f_end = f_start + FRAME_STRIDE
    key_start, key_end = f"frame_{f_start:05d}", f"frame_{f_end:05d}"
    
    persons_to_draw = []
    if key_start in frames_data and key_end in frames_data:
        alpha = (frame_idx - f_start) / FRAME_STRIDE
        end_data = {p['id_p']: p for p in frames_data[key_end]}
        for p_start in frames_data[key_start]:
            tid = p_start['id_p']
            p_new = p_start.copy()
            if tid in end_data:
                p_new['bbox'] = interpolate_bbox(p_start['bbox'], end_data[tid]['bbox'], alpha)
            persons_to_draw.append(p_new)
    elif key_start in frames_data:
        persons_to_draw = frames_data[key_start]

    # ==========================================
    # LỚP 1: NÓN NHÌN MỜ (VIEW CONE)
    # ==========================================
    overlay = frame.copy()
    for p in persons_to_draw:
        if 'pose_vec' in p and p['pose_vec'] is not None:
            x, y, w, h = map(int, p['bbox'])
            head_x, head_y = int(x + w/2), int(y + h*0.2)
            vx, vy = p['pose_vec']
            
            angle = math.atan2(vy, vx)
            half_fov = math.radians(CONE_ANGLE / 2)
            
            pt1 = (head_x, head_y)
            pt2 = (int(head_x + CONE_LENGTH * math.cos(angle - half_fov)), 
                   int(head_y + CONE_LENGTH * math.sin(angle - half_fov)))
            pt3 = (int(head_x + CONE_LENGTH * math.cos(angle + half_fov)), 
                   int(head_y + CONE_LENGTH * math.sin(angle + half_fov)))
            
            pts = np.array([pt1, pt2, pt3], np.int32)
            # Vẽ nón bằng màu trắng mờ
            cv2.fillPoly(overlay, [pts], COLOR_CONE)

    # Trộn lớp overlay với frame gốc, chỉ giữ 20% độ sáng (rất mờ và tinh tế)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # ==========================================
    # LỚP 2: ĐƯỜNG NỐI & BOUNDING BOX
    # ==========================================
    num_people = len(persons_to_draw)
    
    # 2.1 Vẽ đường nối khoảng cách trước (để nó chìm dưới text)
    for i in range(num_people):
        for j in range(i + 1, num_people):
            p1, p2 = persons_to_draw[i], persons_to_draw[j]
            pos1, pos2 = p1.get('real_pos'), p2.get('real_pos')
            
            if pos1 and pos2:
                dist = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
                # Chỉ vẽ nét đứt/mờ cho người CÙNG NHÓM
                if p1['id_g'] != -1 and p1['id_g'] == p2['id_g']:
                    c1 = (int(p1['bbox'][0] + p1['bbox'][2]/2), int(p1['bbox'][1] + p1['bbox'][3]/2))
                    c2 = (int(p2['bbox'][0] + p2['bbox'][2]/2), int(p2['bbox'][1] + p2['bbox'][3]/2))
                    mid = ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
                    
                    cv2.line(frame, c1, c2, COLOR_GROUP_LINE, 1)
                    
                    dist_text = f"{dist:.1f}m"
                    cv2.putText(frame, dist_text, (mid[0]-10, mid[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 2)
                    cv2.putText(frame, dist_text, (mid[0]-10, mid[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

    # 2.2 Vẽ Bounding Box và Thông tin Cực Gọn
    for p in persons_to_draw:
        x, y, w, h = map(int, p['bbox'])
        tid, id_g = p['id_p'], p['id_g']
        gender = str(p.get('gender', '?')).capitalize() 
        gender_char = "F" if gender == 'Female' else "M" if gender == 'Male' else "?"
        
        # Hộp đơn sắc mỏng
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_MAIN, 1)
        
        # Text định danh
        text = f"{tid}-G{id_g} {gender_char}" if id_g != -1 else f"{tid} {gender_char}"
        text_color = COLOR_TEXT_F if gender == 'Female' else COLOR_TEXT_M if gender == 'Male' else COLOR_MAIN
        
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)

    out.write(frame)
    if frame_idx % 100 == 0: print(f"Tiến độ: {frame_idx}/{total_frames}", end='\r')

cap.release()
out.release()
print(f"\n✅ Xong! Video tối giản đã lưu: {OUTPUT_VIDEO}")