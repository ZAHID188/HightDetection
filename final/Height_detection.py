import cv2
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Define constants
COLORS = {'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'BLUE': (255, 0, 0), 
          'WHITE': (255, 255, 255), 'LIGHT_GRAY': (200, 200, 200), 'MAGENTA': (255, 0, 255)}
EDGE_PARAMS = {'CANNY_THRESHOLDS': (30, 150), 'MIN_LINE_LENGTH': 100, 'MAX_LINE_GAP': 20, 'HOUGH_THRESHOLD': 30}
STABILITY_PARAMS = {'HISTORY_LENGTH': 10, 'THRESHOLD': 0.2, 'DISPLAY_TIME': 3}

# Variables for stabilizing height measurement
stable_height, height_history, last_stable_time = None, [], 0
scale_range = 15  # Default scale range

def draw_scale(frame, scale_x, scale_y_bottom, scale_range):
    """Draw measurement scale on the frame with given range"""
    height, width = frame.shape[:2]
    scale_y_top, scale_height = 20, scale_y_bottom - 20
    
    # Draw main scale line
    cv2.line(frame, (scale_x, 0), (scale_x, height), COLORS['RED'], 2)
    
    # Set tick intervals based on scale range
    major_tick = 5 if scale_range <= 25 else (10 if scale_range <= 50 else (20 if scale_range <= 100 else 50))
    pixels_per_cm = scale_height / scale_range
    
    # Draw tick marks and labels
    for i in range(scale_range + 1):
        y_pos = int(scale_y_bottom - i * pixels_per_cm)
        
        if i % major_tick == 0:  # Major ticks
            cv2.line(frame, (scale_x - 12, y_pos), (scale_x, y_pos), COLORS['RED'], 2)
            cv2.putText(frame, f"{i} cm", (scale_x - 50, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['RED'], 2)
        elif i % (major_tick // 5) == 0 and scale_range <= 50:  # Medium ticks
            cv2.line(frame, (scale_x - 8, y_pos), (scale_x, y_pos), COLORS['RED'], 1)
            cv2.putText(frame, f"{i}", (scale_x - 20, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS['RED'], 1)
        elif scale_range <= 25:  # Minor ticks
            cv2.line(frame, (scale_x - 4, y_pos), (scale_x, y_pos), COLORS['RED'], 1)
    
    cv2.putText(frame, f"Scale (0-{scale_range}cm)", (scale_x - 100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['RED'], 2)
    return pixels_per_cm

def is_stable_measurement(history, new_value, threshold):
    """Check if measurement is stable within threshold"""
    if len(history) < STABILITY_PARAMS['HISTORY_LENGTH']:
        return False
    avg = sum(history) / len(history)
    return all(abs(v - avg) < threshold for v in history) and abs(new_value - avg) < threshold

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
    
    # Process image for edge detection (simplified pipeline)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    adaptive_thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, 
                     cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))), *EDGE_PARAMS['CANNY_THRESHOLDS'])
    
    # Setup scale and draw it
    scale_x, scale_y_bottom = width - 70, height - 20
    result = frame.copy()
    pixels_per_cm = draw_scale(result, scale_x, scale_y_bottom, scale_range)
    
    # Detect lines and process measurements
    found_new_measurement, current_height = False, None
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, EDGE_PARAMS['HOUGH_THRESHOLD'], 
                           minLineLength=EDGE_PARAMS['MIN_LINE_LENGTH'], maxLineGap=EDGE_PARAMS['MAX_LINE_GAP'])
    
    if lines is not None:
        # Find and sort horizontal lines
        horizontal_lines = [(x1, y1, x2, y2, (y1+y2)//2) for line in lines for x1, y1, x2, y2 in [line[0]] 
                           if x2 != x1 and abs((y2 - y1) / (x2 - x1)) < 0.1 
                           and np.sqrt((x2 - x1)**2 + (y2 - y1)**2) > EDGE_PARAMS['MIN_LINE_LENGTH']]
        
        if len(horizontal_lines) >= 2:
            horizontal_lines.sort(key=lambda line: line[4])
            top_line, bottom_line = horizontal_lines[0], horizontal_lines[-1]
            
            # Calculate positions
            top_x1, top_y1, top_x2, top_y2, top_y_avg = top_line
            bot_x1, bot_y1, bot_x2, bot_y2, bot_y_avg = bottom_line
            
            # Draw lines
            cv2.line(result, (top_x1, top_y_avg), (top_x2, top_y_avg), COLORS['GREEN'], 2)
            cv2.line(result, (bot_x1, bot_y_avg), (bot_x2, bot_y_avg), COLORS['RED'], 2)
            
            # Calculate measurements
            top_cm = (scale_y_bottom - top_y_avg) / pixels_per_cm
            bot_cm = (scale_y_bottom - bot_y_avg) / pixels_per_cm
            current_height = top_cm - bot_cm
            
            # Add measurement labels
            cv2.putText(result, f"Top: {top_cm:.1f} cm", (top_x1 + 10, top_y_avg - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['GREEN'], 2)
            cv2.putText(result, f"Bottom: {bot_cm:.1f} cm", (bot_x1 + 10, bot_y_avg + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['RED'], 2)
            
            # Draw measurement line with arrows
            mid_x = min(top_x1, bot_x1) - 20
            cv2.line(result, (mid_x, top_y_avg), (mid_x, bot_y_avg), COLORS['MAGENTA'], 2)
            
            # Draw arrows and dotted lines
            arrow_size = 5
            for x_offset, y_offset in [(-arrow_size, arrow_size), (arrow_size, arrow_size)]:
                cv2.line(result, (mid_x, top_y_avg), (mid_x + x_offset, top_y_avg + y_offset), COLORS['MAGENTA'], 2)
                cv2.line(result, (mid_x, bot_y_avg), (mid_x + x_offset, bot_y_avg - y_offset), COLORS['MAGENTA'], 2)
            
            for x in range(top_x2, scale_x, 5):
                cv2.line(result, (x, top_y_avg), (x + 3, top_y_avg), COLORS['GREEN'], 1)
            for x in range(bot_x2, scale_x, 5):
                cv2.line(result, (x, bot_y_avg), (x + 3, bot_y_avg), COLORS['RED'], 1)
            
            found_new_measurement = True
    
    # Handle stable height measurement
    if found_new_measurement and current_height is not None:
        height_history.append(current_height)
        if len(height_history) > STABILITY_PARAMS['HISTORY_LENGTH']:
            height_history.pop(0)
        
        # Check for stability
        if time.time() - last_stable_time > STABILITY_PARAMS['DISPLAY_TIME'] and is_stable_measurement(
                height_history, current_height, STABILITY_PARAMS['THRESHOLD']):
            stable_height = sum(height_history) / len(height_history)
            last_stable_time = time.time()
            print(f"New stable height measurement: {stable_height:.1f} cm")
    
    # Display the measurements
    if stable_height is not None:
        # Draw a prominent box for the final measurement
        box_width, box_height = 300, 60
        box_x, box_y = (width - box_width) // 2, 30
        
        # Draw semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # Add the measurement text
        final_text = f"FINAL HEIGHT: {stable_height:.1f} cm"
        text_size = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = box_x + (box_width - text_size[0]) // 2
        text_y = box_y + (box_height + text_size[1]) // 2
        cv2.putText(result, final_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['WHITE'], 2)
        
        # If currently measuring, also display the real-time measurement
        if found_new_measurement and current_height is not None:
            cv2.putText(result, f"Current: {current_height:.1f} cm", 
                        (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['MAGENTA'], 2)
    elif found_new_measurement and current_height is not None:
        # If no stable height yet, show current measurement
        cv2.putText(result, f"HEIGHT: {current_height:.1f} cm", 
                    (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['MAGENTA'], 2)
        cv2.putText(result, f"Stabilizing: {len(height_history)}/{STABILITY_PARAMS['HISTORY_LENGTH']} frames", 
                    (width//2 - 120, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['WHITE'], 1)
    
    # Display results
    cv2.imshow("Edge Detection", cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    cv2.putText(result, f"Scale: {scale_range}cm | r:reset | c:clear stable | +/-:adjust | q:quit", 
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['WHITE'], 1)
    cv2.imshow("Height Measurement (cm)", result)
    
    # Handle keyboard input with simplified control structure
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('+'), ord('=')]:
        cv2.setTrackbarPos('Scale Range (cm)', 'Height Measurement (cm)', min(200, scale_range + 1))
    elif key in [ord('-'), ord('_')]:
        cv2.setTrackbarPos('Scale Range (cm)', 'Height Measurement (cm)', max(5, scale_range - 1))
    elif key == ord('r'):
        cv2.setTrackbarPos('Scale Range (cm)', 'Height Measurement (cm)', 15)
    elif key == ord('c'):
        stable_height, height_history, last_stable_time = None, [], 0
        print("Cleared stable height measurement")

# Clean up
cap.release()
cv2.destroyAllWindows()