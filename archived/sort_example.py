import cv2
from sort import Sort

tracker = Sort()

cap = cv2.VideoCapture('development.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Process your detections here
    # You need to format your detections as (x, y, w, h, confidence) and pass them to tracker.update()
    detections = [(x, y, w, h, confidence) for x, y, w, h in your_detections]

    # Pass detections to the tracker
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
