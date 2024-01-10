from ultralytics import YOLO


# Load a model
model = YOLO("C:/Users/Admin/Desktop/GoaliePositioningGuide/yolov8n.pt")  # load a pretrained model (recommended for training)
# Set model to use GPU
model.to('cuda')

if __name__ == '__main__':
    # Train the model
    results = model.train(data='core_detector.yaml', epochs=500, imgsz=960, batch=32)
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category