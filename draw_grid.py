import cv2
import numpy as np
import json
import os

def draw_virtual_grid(image_path, matrix_path):
    # 1. Load image and matrix
    if not os.path.exists(matrix_path):
        print("Matrix file not found!")
        return
    
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]
    
    with open(matrix_path, "r") as f:
        data = json.load(f)
        H = np.array(data["homography_matrix"])

    # 2. Calculate inverse matrix to convert from Meters -> Pixels
    # H: Pixel -> World  ==>  H_inv: World -> Pixel
    try:
        H_inv = np.linalg.inv(H)
    except:
        print("Matrix is not invertible!")
        return

    def world_to_pixel(x_world, y_world):
        """Function to convert from Meters to Pixels for drawing"""
        vec_world = np.array([x_world, y_world, 1]).reshape(3, 1)
        vec_pixel = np.dot(H_inv, vec_world)
        u = vec_pixel[0] / vec_pixel[2]
        v = vec_pixel[1] / vec_pixel[2]
        return int(u), int(v)

    # 3. Configure grid (Draw grid within which range?)
    # Assume drawing from -10m to 30m (depending on how wide your field is)
    START_X, END_X = -20, 40 
    START_Y, END_Y = -10, 50
    STEP = 2 # Draw one line every 2 meters (to avoid clutter) 
    overlay = img.copy()

    # 4. Draw vertical lines (along Y axis)
    for x in range(START_X, END_X, STEP):
        # Get 2 endpoints in real world coordinates
        pt1_real = (x, START_Y)
        pt2_real = (x, END_Y)
        
        # Convert to pixels
        pt1_pixel = world_to_pixel(*pt1_real)
        pt2_pixel = world_to_pixel(*pt2_real)
        
        # Draw (Blue color)
        cv2.line(overlay, pt1_pixel, pt2_pixel, (255, 100, 0), 1)
        # Write meter number
        if -1000 < pt1_pixel[0] < 3000 and -1000 < pt1_pixel[1] < 3000:
             cv2.putText(overlay, f"{x}m", pt1_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # 5. Draw horizontal lines (along X axis)
    for y in range(START_Y, END_Y, STEP):
        pt1_real = (START_X, y)
        pt2_real = (END_X, y)
        
        pt1_pixel = world_to_pixel(*pt1_real)
        pt2_pixel = world_to_pixel(*pt2_real)
        
        # Draw (Blue color)
        cv2.line(overlay, pt1_pixel, pt2_pixel, (255, 100, 0), 1)

    # Blend original image and grid (make grid slightly transparent)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Lưu kết quả
    output_file = "grid_visualization.jpg"
    cv2.imwrite(output_file, img)
    print(f"Grid drawing completed! Open the file {output_file} to view it.")

if __name__ == "__main__":
    IMG = "playground.png"
    JSON = "calibration_matrix.json"
    draw_virtual_grid(IMG, JSON)