import cv2
import numpy as np
import argparse

def calibrate_camera(checker_image_path, checker_size=(9, 6), square_size=2.5):
    """
    Calibrate camera using a checkerboard pattern.
    checker_size: Number of inner corners (width, height)
    square_size: Size of each square in cm
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((checker_size[0] * checker_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_size[0], 0:checker_size[1]].T.reshape(-1, 2) * square_size
    
    # Read calibration image
    img = cv2.imread(checker_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checker_size, None)
    
    if ret:
        # Get camera matrix and distortion coefficients
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            [objp], [corners2], gray.shape[::-1], None, None
        )
        return mtx, dist
    else:
        print("Couldn't find checkerboard pattern. Camera calibration failed.")
        return None, None

def measure_stair_height(image_path, reference_object_height=None, camera_height=0, camera_matrix=None, dist_coeffs=None):
    """
    Measure the height of the first stair step.
    
    Parameters:
    - image_path: Path to the stair image
    - reference_object_height: Height of reference object in cm (if available)
    - camera_height: Height of camera from ground in cm
    - camera_matrix, dist_coeffs: Camera calibration parameters
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Undistort image if calibration parameters are available
    if camera_matrix is not None and dist_coeffs is not None:
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Create a copy of the original image for visualization
    result_img = img.copy()
    
    # Filter horizontal lines (potential stair edges)
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope (near-horizontal lines will have a small slope)
            if abs(x2 - x1) > 0:  # Avoid division by zero
                slope = abs((y2 - y1) / (x2 - x1))
                if slope < 0.1:  # Threshold for horizontal lines
                    cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    horizontal_lines.append((x1, y1, x2, y2))
    
    # If reference object is used
    if reference_object_height is not None:
        print("Using reference object method for stair height measurement")
        # Here you would implement code to detect the reference object and calculate the pixel-to-cm ratio
        # For simplicity, this part is not fully implemented
        # In a real implementation, you would need to detect the reference object and use its known height
        # to calculate the stair height
        
        # Placeholder:
        pixel_to_cm_ratio = 0.1  # This would be calculated based on the reference object
        
        # For demonstration purposes, assuming the lowest horizontal line is the stair edge
        if horizontal_lines:
            stair_edge_y = max([y1 for x1, y1, x2, y2 in horizontal_lines])
            image_height = img.shape[0]
            stair_height_pixels = image_height - stair_edge_y
            stair_height_cm = stair_height_pixels * pixel_to_cm_ratio
            print(f"Estimated stair height: {stair_height_cm:.2f} cm")
            
            # Draw the measurement
            cv2.line(result_img, (img.shape[1]//2, image_height), (img.shape[1]//2, stair_edge_y), (0, 0, 255), 2)
            cv2.putText(result_img, f"{stair_height_cm:.2f} cm", (img.shape[1]//2 + 10, (image_height + stair_edge_y)//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print("No reference object provided. Only detecting potential stair edges.")
    
    # Display results
    cv2.imshow("Edges", edges)
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    cv2.imwrite("stair_measurement_result.jpg", result_img)
    print("Result saved as 'stair_measurement_result.jpg'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure stair height from an image')
    parser.add_argument('image_path', type=str, help='Path to the stair image')
    parser.add_argument('--calibration', type=str, help='Path to checkerboard image for calibration')
    parser.add_argument('--reference_height', type=float, help='Height of reference object in cm')
    parser.add_argument('--camera_height', type=float, default=0, help='Height of camera from ground in cm')
    
    args = parser.parse_args()
    
    camera_matrix, dist_coeffs = None, None
    if args.calibration:
        camera_matrix, dist_coeffs = calibrate_camera(args.calibration)
    
    measure_stair_height(
        args.image_path, 
        reference_object_height=args.reference_height,
        camera_height=args.camera_height,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )