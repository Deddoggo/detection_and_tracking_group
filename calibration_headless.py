import cv2
import numpy as np
import json
import os

class CalibrationTool:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Can not find image at: {image_path}")
        
        self.clone = self.image.copy()
        self.src_points = [] 
        self.matrix = None   

    def set_hardcoded_points(self, points):
        """Enter 4 points directly instead of clicking"""
        if len(points) != 4:
            print("Error: You must provide exactly 4 points!")
            return False
        self.src_points = points
        print(f"Entered 4 input points: {self.src_points}")
        return True

    def compute_homography(self, real_width, real_height):
        pts_src = np.array(self.src_points, dtype=np.float32)
        
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        pts_dst = np.array([
            [0, 0],                     
            [real_width, 0],            
            [real_width, real_height],  
            [0, real_height]            
        ], dtype=np.float32)

        self.matrix, status = cv2.findHomography(pts_src, pts_dst)
        print("\n[SUCCESS] Computed Homography Matrix.")
        return self.matrix

    def validate_warp_headless(self, real_width, real_height, output_filename="validation_result.jpg"):
        """Instead of imshow, we imwrite (save image to file) for headless validation"""
        if self.matrix is None: return

        scale_factor = 100 
        warp_w = int(real_width * scale_factor)
        warp_h = int(real_height * scale_factor)

        scale_matrix = np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, 1]
        ])
        
        visual_matrix = np.dot(scale_matrix, self.matrix)
        warped_img = cv2.warpPerspective(self.clone, visual_matrix, (warp_w, warp_h))
        
        # LƯU ẢNH RA FILE
        cv2.imwrite(output_filename, warped_img)
        print(f"\n[SAVED] Validation image saved to: {output_filename}")
        print("Please transfer this file to your personal computer to check if it is rectangular.")

    def save_matrix(self, filename="calibration_matrix.json"):
        if self.matrix is not None:
            data = {"homography_matrix": self.matrix.tolist()}
            with open(filename, "w") as f:
                json.dump(data, f)
            print(f"\n[SAVED] Matrix saved to: {filename}")

if __name__ == "__main__":
    IMG_PATH = "playground.png"
    
    if not os.path.exists(IMG_PATH):
        print(f"Error: Cannot find {IMG_PATH}")
    else:
        tool = CalibrationTool(IMG_PATH)

        # --- IMPORTANT PART: ENTER THE COORDINATES YOU OBSERVED FROM PAINT HERE ---
        # For example: These are the assumed coordinates of the 4 legs of the red-roofed gazebo (You must edit these numbers yourself)
        # Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left
        MY_POINTS = [
            (533, 170),  # Point 1 (x, y)
            (598, 153),  # Point 2
            (665, 171),  # Point 3
            (600, 188)   # Point 4
        ]
        
        # 1. Enter points
        if tool.set_hardcoded_points(MY_POINTS):
            
            # 2. Enter real dimensions (meters) - For example, the gazebo is 4x4m
            REAL_W = 4.0
            REAL_H = 4.0
            
            # 3. Compute
            tool.compute_homography(REAL_W, REAL_H)
            
            # 4. Save validation image (DO NOT show on screen)
            tool.validate_warp_headless(REAL_W, REAL_H)
            
            # 5. Save matrix
            tool.save_matrix()