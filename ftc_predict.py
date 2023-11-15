from ultralytics import YOLO
# image01     = "/home/mario/ftc/images/original/000001.jpg"
# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

# Predict with the model
results = model("/home/mario/ftc/images/original/000001.jpg")  # predict on an image