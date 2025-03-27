import cv2
import numpy as np
import os

def height_measurement():
    # Try multiple camera indices
    camera_found = False
    
    for camera_index in range(3):  # Try indices 0, 1, 2
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            camera_found = True
            print(f"Camera found at index {camera_index}")
            break
        cap.release()
    
    if not camera_found:
        print("No webcam found. Using static image mode.")
        # Use static image mode
        return static_image_measurement()
    
    # Variables for measurement
    points = []
    reference_height_mm = 100.0  # Default reference height (100mm)
    scale = None
    
    # Instructions
    print("INSTRUCTIONS:")
    print("1. Click to mark top and bottom of reference object")
    print("2. Next 2 clicks will measure a new object")
    print("3. Press 'r' to reset, 'q' to quit")
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)} marked")
    
    cv2.namedWindow("Height Measurement")
    cv2.setMouseCallback("Height Measurement", mouse_callback)
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw existing points
        for i, point in enumerate(points):
            color = (0, 255, 0) if i < 2 else (0, 0, 255)
            cv2.circle(frame, point, 5, color, -1)
        
        # Draw reference line and calculate scale
        if len(points) >= 2:
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            ref_height_px = np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
            scale = reference_height_mm / ref_height_px
            cv2.putText(frame, f"Reference: {reference_height_mm}mm", 
                      (points[0][0]+10, points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Measure objects
        if len(points) >= 4 and len(points) % 2 == 0:
            for i in range(2, len(points), 2):
                if i+1 < len(points):
                    # Draw line between object points
                    cv2.line(frame, points[i], points[i+1], (0, 0, 255), 2)
                    
                    # Calculate height
                    obj_height_px = np.sqrt((points[i][0] - points[i+1][0])**2 + 
                                           (points[i][1] - points[i+1][1])**2)
                    obj_height_mm = obj_height_px * scale
                    
                    # Display measurement
                    cv2.putText(frame, f"{obj_height_mm:.1f}mm", 
                              (points[i][0]+10, points[i][1]-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show instructions on frame
        cv2.putText(frame, "Mark reference: 2 points", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Then mark objects to measure", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'r': reset, 'q': quit", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Height Measurement", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points = []
            print("Points reset")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def static_image_measurement():
    # Create a blank image as placeholder
    image_path = None
    
    # Try to find an image file in the current directory
    for file in os.listdir('.'):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = file
            print(f"Found image: {image_path}")
            break
    
    if image_path:
        # Use found image
        image = cv2.imread(image_path)
    else:
        # Create a blank canvas
        print("No image found. Creating blank canvas.")
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.putText(image, "No camera or image found!", (100, 100), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "Place a .jpg or .png file in this directory", (100, 150), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Variables for measurement
    points = []
    reference_height_mm = 100.0  # Default reference height (100mm)
    scale = None
    
    # Instructions
    print("STATIC IMAGE MODE:")
    print("1. Click to mark top and bottom of reference object")
    print("2. Next 2 clicks will measure a new object")
    print("3. Press 'r' to reset, 'q' to quit")
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)} marked")
            update_image()
    
    def update_image():
        # Create a copy of the original image
        display_img = image.copy()
        
        # Draw existing points
        for i, point in enumerate(points):
            color = (0, 255, 0) if i < 2 else (0, 0, 255)
            cv2.circle(display_img, point, 5, color, -1)
        
        # Draw reference line and calculate scale
        if len(points) >= 2:
            cv2.line(display_img, points[0], points[1], (0, 255, 0), 2)
            ref_height_px = np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
            scale = reference_height_mm / ref_height_px
            cv2.putText(display_img, f"Reference: {reference_height_mm}mm", 
                      (points[0][0]+10, points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Measure objects
        if len(points) >= 4 and len(points) % 2 == 0:
            for i in range(2, len(points), 2):
                if i+1 < len(points):
                    # Draw line between object points
                    cv2.line(display_img, points[i], points[i+1], (0, 0, 255), 2)
                    
                    # Calculate height
                    obj_height_px = np.sqrt((points[i][0] - points[i+1][0])**2 + 
                                           (points[i][1] - points[i+1][1])**2)
                    obj_height_mm = obj_height_px * scale
                    
                    # Display measurement
                    cv2.putText(display_img, f"{obj_height_mm:.1f}mm", 
                              (points[i][0]+10, points[i][1]-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show instructions on frame
        cv2.putText(display_img, "Mark reference: 2 points", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_img, "Then mark objects to measure", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_img, "Press 'r': reset, 'q': quit", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the image
        cv2.imshow("Height Measurement (Static Image)", display_img)
    
    cv2.namedWindow("Height Measurement (Static Image)")
    cv2.setMouseCallback("Height Measurement (Static Image)", mouse_callback)
    
    # Initial display
    update_image()
    
    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points = []
            print("Points reset")
            update_image()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    height_measurement()