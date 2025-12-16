import cv2
import numpy as np
import json
import os

class CalibrationTool:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot find image at: {image_path}")
        
        self.clone = self.image.copy()
        self.src_points = [] # Store pixel coordinates of mouse clicks
        self.matrix = None   # Homography matrix to be calculated
    def click_event(self, event, x, y, flags, params):
        """Mouse event handler to select points on the image"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.src_points) < 4:
                self.src_points.append((x, y))
                
                # Draw point and order number on the image for better visualization
                cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.image, str(len(self.src_points)), (x+10, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Calibration: Select 4 points", self.image)

    def select_reference_points(self):
        """Open window for user to click 4 points"""
        print("\n--- CALIBRATION INSTRUCTIONS ---")
        print("1. Please click on 4 points forming a rectangle on the ground.")
        print("2. The order of clicks is MANDATORY: \n   Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
        print("3. Press any key after selecting 4 points to continue.")
        print("--------------------------------")
        
        cv2.imshow("Calibration: Select 4 points", self.image)
        cv2.setMouseCallback("Calibration: Select 4 points", self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(self.src_points) != 4:
            print("Error: You have not selected 4 points!")
            return False
        return True

    def compute_homography(self, real_width, real_height):
        """Calculate homography matrix based on real dimensions (meters)"""
        # 1. Source points (image coordinates) - Convert to numpy float32
        pts_src = np.array(self.src_points, dtype=np.float32)

        # 2. Destination points (real-world coordinates) - Assume origin (0,0) is the first point
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        pts_dst = np.array([
            [0, 0],                     # Top-Left
            [real_width, 0],            # Top-Right
            [real_width, real_height],  # Bottom-Right
            [0, real_height]            # Bottom-Left
        ], dtype=np.float32)

        # 3. Calculate Homography matrix
        self.matrix, status = cv2.findHomography(pts_src, pts_dst)
        
        print("\n[SUCCESS] Computed Homography Matrix:")
        print(self.matrix)
        return self.matrix

    def validate_warp(self, real_width, real_height):
        """Display Top-Down view image for accuracy check"""
        if self.matrix is None: return

        # Scale up to display the warped image nicely (e.g., 1 meter = 100 pixels)
        scale_factor = 100 
        warp_w = int(real_width * scale_factor)
        warp_h = int(real_height * scale_factor)

        # Create scale matrix to warp image for better visualization (otherwise it would be tiny 4x4 pixels)
        scale_matrix = np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, 1]
        ])
        
        # Combine original Homography matrix with scale matrix for visualization
        visual_matrix = np.dot(scale_matrix, self.matrix)

        # Warp image to top-down view
        warped_img = cv2.warpPerspective(self.clone, visual_matrix, (warp_w, warp_h))
        
        print(f"\nDisplaying warped image (Top-down view)...")
        print("If the sidewalk/tiles lines become PARALLEL and PERPENDICULAR --> Calibration is accurate.")
        cv2.imshow("Validation: Top-Down View", warped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_matrix(self, filename="calibration_matrix.json"):
        """Save matrix to file for use in other modules"""
        if self.matrix is not None:
            data = {"homography_matrix": self.matrix.tolist()}
            with open(filename, "w") as f:
                json.dump(data, f)
            print(f"\n[SAVED] Matrix saved to file: {filename}")

    def pixel_to_metric(self, u, v):
        """Utility function: Convert a pixel point to meters (for testing)"""
        if self.matrix is None: return None
        
        # Transform vector [u, v, 1]
        point_vector = np.array([u, v, 1]).reshape(3, 1)
        metric_vector = np.dot(self.matrix, point_vector)
        
        # Normalize (divide by parameter w' to get Cartesian coordinates)
        x = metric_vector[0] / metric_vector[2]
        y = metric_vector[1] / metric_vector[2]
        return (float(x), float(y))

# --- MAIN PROGRAM ---
if __name__ == "__main__":
    # 1. Image path
    IMG_PATH = "playground.png" 
    
    # Check if image exists
    if not os.path.exists(IMG_PATH):
        print(f"Error: Please rename the image file in the code to the correct filename (currently {IMG_PATH})")
    else:
        tool = CalibrationTool(IMG_PATH)

        # STEP 1: Select points
        if tool.select_reference_points():
            
            # STEP 2: Enter real dimensions (Assumed or Measured in meters)
            try:
                print("\n--- ENTER REAL DIMENSIONS (METERS) ---")
                rw = float(input("Enter the real width of the rectangle you just drew (m): "))
                rh = float(input("Enter the real height of the rectangle you just drew (m): "))
                
                # STEP 3: Compute Homography matrix
                tool.compute_homography(rw, rh)
                
                # STEP 4: Visual validation
                tool.validate_warp(rw, rh)
                
                # STEP 5: Save matrix
                tool.save_matrix()
                
                # STEP 6: Test a random point conversion
                print("\n--- TEST ---")
                print("Assuming AI detected a child at pixel coordinates (500, 500)")
                x_real, y_real = tool.pixel_to_metric(500, 500)
                print(f"Actual position on the meter map: X={x_real:.2f}m, Y={y_real:.2f}m")
                
            except ValueError:
                print("Error: Please enter a valid number.")