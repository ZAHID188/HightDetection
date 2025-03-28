import cv2
import numpy as np

# Initialize webcam and check if connected
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Define constants
RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)
WHITE, LIGHT_GRAY, MAGENTA = (255, 255, 255), (200, 200, 200), (255, 0, 255)
CANNY_THRESHOLD1, CANNY_THRESHOLD2 = 30, 150
MIN_LINE_LENGTH, MAX_LINE_GAP, HOUGH_THRESHOLD = 100, 20, 30
scale_range = 15  # Default scale range

def draw_scale(frame, scale_x, scale_y_bottom, scale_range):
    """Draw measurement scale on the frame with given range"""
    height, width = frame.shape[:2]
    scale_y_top = 20
    scale_height = scale_y_bottom - scale_y_top
    
    # Draw main scale line
    cv2.line(frame, (scale_x, 0), (scale_x, height), RED, 2)
    
    # Set tick intervals based on scale range
    major_tick = 5 if scale_range <= 25 else (10 if scale_range <= 50 else (20 if scale_range <= 100 else 50))
    pixels_per_cm = scale_height / scale_range
    
    # Draw tick marks and labels
    for i in range(scale_range + 1):
        y_pos = int(scale_y_bottom - i * pixels_per_cm)
        
        if i % major_tick == 0:  # Major ticks
            cv2.line(frame, (scale_x - 12, y_pos), (scale_x, y_pos), RED, 2)
            cv2.putText(frame, f"{i} cm", (scale_x - 50, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        elif i % (major_tick // 5) == 0 and scale_range <= 50:  # Medium ticks
            cv2.line(frame, (scale_x - 8, y_pos), (scale_x, y_pos), RED, 1)
            cv2.putText(frame, f"{i}", (scale_x - 20, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, RED, 1)
        elif scale_range <= 25:  # Minor ticks
            cv2.line(frame, (scale_x - 4, y_pos), (scale_x, y_pos), RED, 1)
    
    cv2.putText(frame, f"Scale (0-{scale_range}cm)", (scale_x - 100, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    
    return pixels_per_cm

# Create window with scale range trackbar
cv2.namedWindow('Height Measurement (cm)')
cv2.createTrackbar('Scale Range (cm)', 'Height Measurement (cm)', scale_range, 200, lambda x: None)

while True:
    # Capture and process frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    height, width = frame.shape[:2]
    scale_range = max(1, cv2.getTrackbarPos('Scale Range (cm)', 'Height Measurement (cm)'))
    
    # Process image for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    adaptive_thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    enhanced_horizontal = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel)
    edges = cv2.Canny(enhanced_horizontal, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    
    # Setup scale and draw it
    scale_x = width - 70
    scale_y_bottom = height - 20
    result = frame.copy()
    pixels_per_cm = draw_scale(result, scale_x, scale_y_bottom, scale_range)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESHOLD, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    
    # Process detected lines
    if lines is not None:
        horizontal_lines = []
        
        # Find horizontal lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1 and abs((y2 - y1) / (x2 - x1)) < 0.1:
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if line_length > MIN_LINE_LENGTH:
                    horizontal_lines.append((x1, y1, x2, y2, y1))
        
        # If we have at least 2 horizontal lines
        if len(horizontal_lines) >= 2:
            horizontal_lines.sort(key=lambda line: line[4])
            top_line = horizontal_lines[0]
            bottom_line = horizontal_lines[-1]
            
            # Calculate positions
            top_x1, top_y1, top_x2, top_y2, _ = top_line
            bot_x1, bot_y1, bot_x2, bot_y2, _ = bottom_line
            top_y_avg = (top_y1 + top_y2) // 2
            bot_y_avg = (bot_y1 + bot_y2) // 2
            
            # Draw lines
            cv2.line(result, (top_x1, top_y_avg), (top_x2, top_y_avg), GREEN, 2)
            cv2.line(result, (bot_x1, bot_y_avg), (bot_x2, bot_y_avg), RED, 2)
            
            # Calculate measurements
            top_cm = (scale_y_bottom - top_y_avg) / pixels_per_cm
            bot_cm = (scale_y_bottom - bot_y_avg) / pixels_per_cm
            height_diff_cm = top_cm - bot_cm
            
            # Add measurement labels
            cv2.putText(result, f"Top: {top_cm:.1f} cm", (top_x1 + 10, top_y_avg - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
            cv2.putText(result, f"Bottom: {bot_cm:.1f} cm", (bot_x1 + 10, bot_y_avg + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
            cv2.putText(result, f"HEIGHT: {height_diff_cm:.1f} cm", (width//2 - 100, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, MAGENTA, 2)
            
            # Draw measurement line with arrows
            mid_x = min(top_x1, bot_x1) - 20
            cv2.line(result, (mid_x, top_y_avg), (mid_x, bot_y_avg), MAGENTA, 2)
            
            # Draw arrows
            arrow_size = 5
            for x_offset, y_offset in [(-arrow_size, arrow_size), (arrow_size, arrow_size)]:
                cv2.line(result, (mid_x, top_y_avg), (mid_x + x_offset, top_y_avg + y_offset), MAGENTA, 2)
                cv2.line(result, (mid_x, bot_y_avg), (mid_x + x_offset, bot_y_avg - y_offset), MAGENTA, 2)
            
            # Draw dotted lines to scale
            for x in range(top_x2, scale_x, 5):
                cv2.line(result, (x, top_y_avg), (x + 3, top_y_avg), GREEN, 1)
            for x in range(bot_x2, scale_x, 5):
                cv2.line(result, (x, bot_y_avg), (x + 3, bot_y_avg), RED, 1)
    
    # Display results
    cv2.imshow("Edge Detection", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    cv2.putText(result, f"Scale: {scale_range}cm | r:reset | +/-:adjust | q:quit", 
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.imshow("Height Measurement (cm)", result)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        scale_range = min(200, scale_range + 1)
        cv2.setTrackbarPos('Scale Range (cm)', 'Height Measurement (cm)', scale_range)
    elif key == ord('-') or key == ord('_'):
        scale_range = max(5, scale_range - 1)
        cv2.setTrackbarPos('Scale Range (cm)', 'Height Measurement (cm)', scale_range)
    elif key == ord('r'):
        scale_range = 15
        cv2.setTrackbarPos('Scale Range (cm)', 'Height Measurement (cm)', scale_range)

# Clean up
cap.release()
cv2.destroyAllWindows()