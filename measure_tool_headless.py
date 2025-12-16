import cv2
import numpy as np
import json
import os

class VirtualRulerHeadless:
    def __init__(self, matrix_path):
        # 1. Load matrix from JSON file
        if not os.path.exists(matrix_path):
            print(f"[ERROR] File {matrix_path} not found.")
            exit()
            
        with open(matrix_path, "r") as f:
            data = json.load(f)
            self.H = np.array(data["homography_matrix"])
            print("[INFO] Loaded homography matrix from JSON.")

    def pixel_to_world(self, u, v):
        """Convert Pixel -> Meters"""
        vec_pixel = np.array([u, v, 1]).reshape(3, 1)
        vec_world = np.dot(self.H, vec_pixel)
        x = vec_world[0] / vec_world[2]
        y = vec_world[1] / vec_world[2]
        return float(x), float(y)

    def measure(self, point1, point2):
        """Calculate distance between two pixel points"""
        # Convert to meters
        x1, y1 = self.pixel_to_world(point1[0], point1[1])
        x2, y2 = self.pixel_to_world(point2[0], point2[1])
        
        # Calculate distance
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        print(f"\n--- MEASUREMENT RESULT ---")
        print(f"Point A (Pixel {point1}) --> Real coordinates: ({x1:.2f}m, {y1:.2f}m)")
        print(f"Point B (Pixel {point2}) --> Real coordinates: ({x2:.2f}m, {y2:.2f}m)")
        print(f"==> ACTUAL DISTANCE: {dist:.2f} METERS")
        return dist

if __name__ == "__main__":
    JSON_PATH = "calibration_matrix.json"
    
    # Initialize the tool
    ruler = VirtualRulerHeadless(JSON_PATH)
    
    # --- ENTER THE COORDINATES YOU WANT TO MEASURE HERE ---    
    POINT_A = (782, 368)  
    POINT_B = (767, 360)  
    
    # Run measurement
    ruler.measure(POINT_A, POINT_B)