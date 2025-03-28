import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(1)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection on the frame
    results = model(frame)
    
    # Visualize results on the frame
    annotated_frame = results[0].plot()
    
    # Display the result
    cv2.imshow("Object Detection", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()