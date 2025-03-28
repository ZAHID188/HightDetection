import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(1)

# Define a reference object width in centimeters (for scale)
REFERENCE_WIDTH_CM = 10.0

# Add these variables at the top
REFERENCE_OBJECT_SIZE_CM = 50  # Size of your reference object in cm
reference_object_pixels = None   # Will be updated based on detection




# Colors for visualization
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)

# For adjusting edge detection parameters interactively
cv2.namedWindow('Height Measurement (cm)')
cv2.createTrackbar('Canny Threshold 1', 'Height Measurement (cm)', 30, 255, lambda x: None)
cv2.createTrackbar('Canny Threshold 2', 'Height Measurement (cm)', 150, 255, lambda x: None)
cv2.createTrackbar('Min Line Length', 'Height Measurement (cm)', 100, 500, lambda x: None)
cv2.createTrackbar('Line Gap', 'Height Measurement (cm)', 20, 100, lambda x: None)
cv2.createTrackbar('Hough Threshold', 'Height Measurement (cm)', 30, 100, lambda x: None)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
        
    # Get frame dimensions
    height, width = frame.shape[:2]

    # Get current trackbar values
    canny_threshold1 = cv2.getTrackbarPos('Canny Threshold 1', 'Height Measurement (cm)')
    canny_threshold2 = cv2.getTrackbarPos('Canny Threshold 2', 'Height Measurement (cm)')
    min_line_length = cv2.getTrackbarPos('Min Line Length', 'Height Measurement (cm)')
    max_line_gap = cv2.getTrackbarPos('Line Gap', 'Height Measurement (cm)')
    hough_threshold = cv2.getTrackbarPos('Hough Threshold', 'Height Measurement (cm)')
        
    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple preprocessing steps to improve edge detection
    # 1. Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Apply adaptive thresholding to handle varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 3. Apply morphological operations to enhance horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    enhanced_horizontal = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # 4. Apply Canny edge detection with trackbar parameters
    edges = cv2.Canny(enhanced_horizontal, canny_threshold1, canny_threshold2)
    
    # Define scale parameters
    scale_x = width - 70  # Position from right edge
    scale_y_top = 20  # Top position
    scale_y_bottom = height - 20  # Bottom position
    scale_height = scale_y_bottom - scale_y_top

    
    
    # Use a reference object to calibrate the scale
    if reference_object_pixels is not None:
        pixels_per_cm = reference_object_pixels / REFERENCE_OBJECT_SIZE_CM
    else:
        # Fallback to the fixed scale if no reference object is detected
        pixels_per_cm = scale_height / 25
    
    # Use Hough Line Transform to detect lines with trackbar parameters
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    # Draw original frame
    result = frame.copy()
    
    # Draw the main scale line - extend from top to bottom
    cv2.line(result, (scale_x, 0), (scale_x, height), WHITE, 2)
    
    # Draw tick marks and labels for centimeters
    for i in range(26):  # 0 to 25 cm
        # Calculate position for this centimeter mark
        y_pos = int(scale_y_bottom - i * pixels_per_cm)
        
        if i % 5 == 0:  # Every 5 cm - longer ticks with labels
            # Draw longer tick marks for each 5 cm
            cv2.line(result, (scale_x - 12, y_pos), (scale_x, y_pos), WHITE, 2)
            
            # Add text labels
            cv2.putText(result, f"{i} cm", (scale_x - 50, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        
        elif i % 1 == 0:  # Every 1 cm - medium ticks
            cv2.line(result, (scale_x - 8, y_pos), (scale_x, y_pos), LIGHT_GRAY, 1)
            
            # Add small labels for each cm
            cv2.putText(result, f"{i}", (scale_x - 20, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, LIGHT_GRAY, 1)
    
    # Add scale title
    cv2.putText(result, "Scale (cm)", (scale_x - 70, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    
    # Process the detected lines
    if lines is not None:
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope to identify horizontal lines
            if x2 != x1:  # Avoid division by zero
                slope = abs((y2 - y1) / (x2 - x1))
                if slope < 0.1:  # Threshold for horizontal lines
                    # Calculate line length
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    # Add line if it's long enough
                    if line_length > min_line_length:
                        horizontal_lines.append((x1, y1, x2, y2, y1))  # Using y1 for sorting
        
        # If we found multiple horizontal lines
        if len(horizontal_lines) >= 2:
            # Sort horizontal lines by y-position (top to bottom)
            horizontal_lines.sort(key=lambda line: line[4])  # Sort by y1
            
            # Take the top line and bottom line
            top_line = horizontal_lines[0]
            bottom_line = horizontal_lines[-1]
            
            # Extract coordinates
            top_x1, top_y1, top_x2, top_y2, _ = top_line
            bot_x1, bot_y1, bot_x2, bot_y2, _ = bottom_line
            
            # Use the average y-coordinate for each line
            top_y_avg = (top_y1 + top_y2) // 2
            bot_y_avg = (bot_y1 + bot_y2) // 2
            
            # Draw the top horizontal line in GREEN
            cv2.line(result, (top_x1, top_y_avg), (top_x2, top_y_avg), GREEN, 2)
            
            # Draw the bottom horizontal line in RED
            cv2.line(result, (bot_x1, bot_y_avg), (bot_x2, bot_y_avg), RED, 2)
            
            # Calculate the scale reading for top line
            top_cm = (scale_y_bottom - top_y_avg) / pixels_per_cm
            
            # Calculate the scale reading for bottom line
            bot_cm = (scale_y_bottom - bot_y_avg) / pixels_per_cm
            
            # Calculate the height difference
            height_diff_cm = top_cm - bot_cm
            
            # Format the measurement texts
            top_measurement = f"{top_cm:.1f} cm"
            bot_measurement = f"{bot_cm:.1f} cm"
            height_measurement = f"{height_diff_cm:.1f} cm"
            
            # Display the measurements
            # Top line label
            cv2.putText(result, f"Top: {top_measurement}", 
                        (top_x1 + 10, top_y_avg - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
            
            # Bottom line label
            cv2.putText(result, f"Bottom: {bot_measurement}", 
                        (bot_x1 + 10, bot_y_avg + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
            
            # Height difference label
            cv2.putText(result, f"HEIGHT: {height_measurement}", 
                        (width//2 - 100, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, MAGENTA, 2)
            
            # Draw a vertical measure line between the two horizontal lines
            mid_x = min(top_x1, bot_x1) - 20  # 20 pixels to the left of the leftmost line
            cv2.line(result, (mid_x, top_y_avg), (mid_x, bot_y_avg), MAGENTA, 2)
            
            # Add arrows at the ends of the vertical measure line
            arrow_size = 5
            # Top arrow
            cv2.line(result, (mid_x, top_y_avg), (mid_x - arrow_size, top_y_avg + arrow_size), MAGENTA, 2)
            cv2.line(result, (mid_x, top_y_avg), (mid_x + arrow_size, top_y_avg + arrow_size), MAGENTA, 2)
            # Bottom arrow
            cv2.line(result, (mid_x, bot_y_avg), (mid_x - arrow_size, bot_y_avg - arrow_size), MAGENTA, 2)
            cv2.line(result, (mid_x, bot_y_avg), (mid_x + arrow_size, bot_y_avg - arrow_size), MAGENTA, 2)
            
            # Extend dotted lines to the scale
            for x in range(top_x2, scale_x, 5):
                cv2.line(result, (x, top_y_avg), (x + 3, top_y_avg), GREEN, 1)
            
            for x in range(bot_x2, scale_x, 5):
                cv2.line(result, (x, bot_y_avg), (x + 3, bot_y_avg), RED, 1)
    
    # Show the edge detection result for debugging
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Edge Detection", edges_color)
    
    # Show the result
    cv2.imshow("Height Measurement (cm)", result)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()