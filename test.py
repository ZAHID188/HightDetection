import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define a reference object width in centimeters (for scale)
# In real applications, you would place an object of known size in the frame
REFERENCE_WIDTH_CM = 10.0

# Colors for visualization
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Simple thresholding to separate objects from background
    # Adjust the threshold value (128) as needed for your environment
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of objects in the frame
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw original frame
    result = frame.copy()
    
    # Process each contour (object) found
    for contour in contours:
        # Filter small contours (noise)
        if cv2.contourArea(contour) < 1000:  # Adjust this value for your needs
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw rectangle around the object
        cv2.rectangle(result, (x, y), (x + w, y + h), GREEN, 2)
        
        # Calculate height in pixels
        height_pixels = h
        
        # For a real measurement, you would need calibration
        # This is a simple approximation using the reference object
        # Assuming the reference object width (REFERENCE_WIDTH_CM) corresponds to w pixels
        height_cm = (height_pixels * REFERENCE_WIDTH_CM) / w
        
        # Display the height
        text = f"Height: {height_cm:.1f} cm"
        cv2.putText(result, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
        
    # Show the result
    cv2.imshow("Object Height Measurement", result)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()