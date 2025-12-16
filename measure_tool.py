import cv2
import numpy as np
import json
import os

class VirtualRuler:
    def __init__(self, image_path, matrix_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.points = [] # Save clicked points
        
        # 1. LOAD MATRIX FROM JSON FILE
        if not os.path.exists(matrix_path):
            print(f"[ERROR] File {matrix_path} not found. Have you run calibration.py?")
            exit()
            
        with open(matrix_path, "r") as f:
            data = json.load(f)
            self.H = np.array(data["homography_matrix"])
            print("[OK] Matrix loaded successfully!")

    def pixel_to_world(self, u, v):
        """Magic function: Convert Pixel -> Meters"""
        # Pixel point vector [u, v, 1]
        vec_pixel = np.array([u, v, 1]).reshape(3, 1)
        
        # Multiply matrix: World = H * Pixel
        vec_world = np.dot(self.H, vec_pixel)
        
        # Normalize coordinates (Divide by parameter w)
        x = vec_world[0] / vec_world[2]
        y = vec_world[1] / vec_world[2]
        
        return float(x), float(y)

    def click_event(self, event, x, y, flags, params):
        # When left mouse button is clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Only allow selecting up to 2 points at a time
            if len(self.points) >= 2:
                self.points = [] # Reset if already clicked 2 times before
                self.image = cv2.imread(self.image_path) # Reload clean image
            
            self.points.append((x, y))
            
            # Draw red dot at clicked point
            cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Virtual Ruler (Press 'q' to quit)", self.image)
            
            # IF 2 POINTS ARE SELECTED --> CALCULATE DISTANCE
            if len(self.points) == 2:
                p1 = self.points[0]
                p2 = self.points[1]
                
                # --- MOST IMPORTANT PART ---
                # Convert pixel coordinates to meters
                x1_real, y1_real = self.pixel_to_world(p1[0], p1[1])
                x2_real, y2_real = self.pixel_to_world(p2[0], p2[1])
                
                # Calculate Euclidean distance: d = sqrt((x2-x1)^2 + (y2-y1)^2)
                distance_meters = np.sqrt((x2_real - x1_real)**2 + (y2_real - y1_real)**2)
                # -----------------------------
                
                # Draw line and display distance on image
                cv2.line(self.image, p1, p2, (0, 255, 0), 2)
                
                # Print results on image
                text = f"{distance_meters:.2f} meters"
                mid_point = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                
                # Draw black background for better text readability
                cv2.rectangle(self.image, (mid_point[0], mid_point[1]-20), (mid_point[0]+200, mid_point[1]+10), (0,0,0), -1)
                cv2.putText(self.image, text, (mid_point[0], mid_point[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                print(f"\n--- MEASUREMENT RESULT ---")
                print(f"Point 1 (Pixel): {p1} -> Real: ({x1_real:.2f}m, {y1_real:.2f}m)")
                print(f"Point 2 (Pixel): {p2} -> Real: ({x2_real:.2f}m, {y2_real:.2f}m)")
                print(f"==> ACTUAL DISTANCE: {distance_meters:.2f} METERS")
                
                cv2.imshow("Virtual Ruler (Press 'q' to quit)", self.image)

    def run(self):
        print("\n--- INSTRUCTIONS FOR USING THE RULER ---")
        print("1. Left-click on the FIRST POINT (e.g., Person A's foot).")
        print("2. Left-click on the SECOND POINT (e.g., Person B's foot).")
        print("3. Look at the image to see the distance in METERS.")
        print("4. Click again to measure another pair.")
        print("5. Press 'q' to quit.")
        
        cv2.imshow("Virtual Ruler (Press 'q' to quit)", self.image)
        cv2.setMouseCallback("Virtual Ruler (Press 'q' to quit)", self.click_event)
        
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    IMG = "playground.png" 
    JSON = "calibration_matrix.json"
    
    if os.path.exists(IMG):
        app = VirtualRuler(IMG, JSON)
        app.run()
    else:
        print(f"Error: Image {IMG} not found. Please check the path.")