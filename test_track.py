import cv2
from sort import Sort

# Load pre-trained object detection model (e.g., YOLO)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Create SORT tracker
tracker = Sort()

cap = cv2.VideoCapture('development.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Object detection using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)
    
    # Process detections and pass them to the tracker
    detections = post_process(frame, outs)
    tracked_objects = tracker.update(detections)
    
    # Visualize tracked objects
    for obj in tracked_objects:
        x, y, w, h, track_id = obj
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow('Object Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

