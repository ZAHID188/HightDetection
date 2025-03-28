import cv2

# Open the default camera (usually webcam)
cap = cv2.VideoCapture(1)  # 0 is the default camera, use 1, 2, etc. for additional cameras

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture and display video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break
        
    # Display the frame
    cv2.imshow('Camera Feed', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()