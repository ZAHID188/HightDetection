import cv2
import numpy as np
import time

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
scale_range = 150  # Default scale range in millimeters (15cm = 150mm)

# Variables for stabilizing the height measurement
stable_height = None
height_history = []
HISTORY_LENGTH = 10  # Number of frames to consider for stabilization
STABILITY_THRESHOLD = 2.0  # Maximum allowed deviation for stable reading (mm)
last_stable_time = 0
STABLE_DISPLAY_TIME = 3  # Seconds to show stable height before allowing new measurement

def draw_scale(frame, scale_x, scale_y_bottom, scale_range):
    """Draw measurement scale on the frame with given range in millimeters"""
    height, width = frame.shape[:2]
    scale_y_top = 20
    scale_height = scale_y_bottom - scale_y_top
    
    # Draw main scale line
    cv2.line(frame, (scale_x, 0), (scale_x, height), RED, 2)
    
    # Set tick intervals based on scale range
    if scale_range <= 250:  # 25cm
        major_tick = 50   # 5cm
    elif scale_range <= 500:  # 50cm
        major_tick = 100  # 10cm
    elif scale_range <= 1000:  # 100cm
        major_tick = 200  # 20cm
    else:
        major_tick = 500  # 50cm
        
    pixels_per_mm = scale_height / scale_range
    
    # Draw tick marks and labels
    for i in range(0, scale_range + 1, 10):  # Step by 10mm increments
        y_pos = int(scale_y_bottom - i * pixels_per_mm)
        
        if i % major_tick == 0:  # Major ticks
            # Draw longer tick marks for major intervals
            cv2.line(frame, (scale_x - 12, y_pos), (scale_x, y_pos), RED, 2)
            
            # Add text labels
            cv2.putText(frame, f"{i} mm", (scale_x - 60, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        
        elif i % (major_tick // 5) == 0 and scale_range <= 500:  # Medium ticks
            cv2.line(frame, (scale_x - 8, y_pos), (scale_x, y_pos), RED, 1)
            
            # Add small labels for medium ticks
            cv2.putText(frame, f"{i}", (scale_x - 25, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, RED, 1)
        
        elif scale_range <= 250 and i % 10 == 0:  # Minor ticks for smaller ranges
            cv2.line(frame, (scale_x - 4, y_pos), (scale_x, y_pos), RED, 1)
    
    # Add scale title
    cv2.putText(frame, f"Scale (0-{scale_range}mm)", (scale_x - 120, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    
    return pixels_per_mm

def is_stable_measurement(history, new_value, threshold):
    """Check if measurement is stable within threshold"""
    if len(history) < HISTORY_LENGTH:
        return False
    
    avg = sum(history) / len(history)
    return all(abs(v - avg) < threshold for v in history) and abs(new_value - avg) < threshold

# Create window with scale range trackbar (maximum 2000mm = 2m)
cv2.namedWindow('Height Measurement (mm)')
cv2.createTrackbar('Scale Range (mm)', 'Height Measurement (mm)', scale_range, 2000, lambda x: None)

while True:
    # Capture and process frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    height, width = frame.shape[:2]
    scale_range = max(10, cv2.getTrackbarPos('Scale Range (mm)', 'Height Measurement (mm)'))
    
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
    pixels_per_mm = draw_scale(result, scale_x, scale_y_bottom, scale_range)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESHOLD, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    
    # Current time for stability logic
    current_time = time.time()
    found_new_measurement = False
    current_height = None
    
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
            
            # Calculate measurements in millimeters
            top_mm = (scale_y_bottom - top_y_avg) / pixels_per_mm
            bot_mm = (scale_y_bottom - bot_y_avg) / pixels_per_mm
            height_diff_mm = top_mm - bot_mm
            current_height = height_diff_mm
            
            # Add measurement labels
            cv2.putText(result, f"Top: {top_mm:.1f} mm", (top_x1 + 10, top_y_avg - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
            cv2.putText(result, f"Bottom: {bot_mm:.1f} mm", (bot_x1 + 10, bot_y_avg + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
            
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
            
            found_new_measurement = True
    
    # Handle stable height measurement
    if found_new_measurement and current_height is not None:
        # Update history
        height_history.append(current_height)
        if len(height_history) > HISTORY_LENGTH:
            height_history.pop(0)
        
        # Check for stability
        if current_time - last_stable_time > STABLE_DISPLAY_TIME and is_stable_measurement(height_history, current_height, STABILITY_THRESHOLD):
            stable_height = sum(height_history) / len(height_history)
            last_stable_time = current_time
            print(f"New stable height measurement: {stable_height:.1f} mm")
    
    # Display the final stable height measurement
    if stable_height is not None:
        # Draw a prominent box for the final measurement
        box_width, box_height = 300, 60
        box_x = (width - box_width) // 2
        box_y = 30
        
        # Draw semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # Add the measurement text with a bold, prominent display
        final_text = f"FINAL HEIGHT: {stable_height:.1f} mm"
        text_size = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = box_x + (box_width - text_size[0]) // 2
        text_y = box_y + (box_height + text_size[1]) // 2
        
        cv2.putText(result, final_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        
        # If currently measuring, also display the real-time measurement
        if found_new_measurement and current_height is not None:
            cv2.putText(result, f"Current: {current_height:.1f} mm", 
                        (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, MAGENTA, 2)
    elif found_new_measurement and current_height is not None:
        # If no stable height yet, show current measurement
        cv2.putText(result, f"HEIGHT: {current_height:.1f} mm", 
                    (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, MAGENTA, 2)
        
        # Show stability progress
        stability_msg = f"Stabilizing: {len(height_history)}/{HISTORY_LENGTH} frames"
        cv2.putText(result, stability_msg, (width//2 - 120, height//2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    
    # Display results
    cv2.imshow("Edge Detection", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    
    # Add key controls info
    controls = f"Scale: {scale_range}mm | r:reset | c:clear stable | +/-:adjust | q:quit"
    cv2.putText(result, controls, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.imshow("Height Measurement (mm)", result)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        scale_range = min(2000, scale_range + 10)  # Increment by 10mm
        cv2.setTrackbarPos('Scale Range (mm)', 'Height Measurement (mm)', scale_range)
    elif key == ord('-') or key == ord('_'):
        scale_range = max(50, scale_range - 10)  # Decrement by 10mm
        cv2.setTrackbarPos('Scale Range (mm)', 'Height Measurement (mm)', scale_range)
    elif key == ord('r'):
        scale_range = 150  # Reset to 150mm (15cm)
        cv2.setTrackbarPos('Scale Range (mm)', 'Height Measurement (mm)', scale_range)
    elif key == ord('c'):
        # Clear stable height to start new measurement
        stable_height = None
        height_history = []
        last_stable_time = 0
        print("Cleared stable height measurement")

# Clean up
cap.release()
cv2.destroyAllWindows()