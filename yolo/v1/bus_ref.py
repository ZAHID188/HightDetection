from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Specify the source image
source = 'https://ultralytics.com/images/bus.jpg'

# Make predictions
results = model.predict(source, save=True, imgsz=320, conf=0.5)

# Extract bounding box dimensions
boxes = results[0].boxes.xywh.cpu()
for box in boxes:
    x, y, w, h = box
    print("Width of Box: {}, Height of Box: {}".format(w, h))