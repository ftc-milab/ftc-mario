from ultralytics import YOLO
# image01     = "/home/mario/ftc/images/original/000001.jpg"
# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('runs/detect/train7/weights/best.pt')  # load a custom model
model = YOLO('runs/detect/train7/weights/last.pt')  # load a custom model

# Predict with the model
# results = model("/home/mario/ftc/images/original/000001.jpg")  # predict on an image
# results = model("/home/mario/data/ftc-train20/images/train/000001.jpg",imgsz=(2058, 2456),show=True, conf=0.3)
results = model("/home/mario/data/ftc-train20/images/train/000019.jpg",save=True, imgsz=640,conf=0.2,hide_labels=True)