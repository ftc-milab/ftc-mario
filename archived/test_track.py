import cv2
import numpy as np
from sort import Sort

net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]



tracker = Sort()

cap = cv2.VideoCapture('development.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # cv2.imshow('window',  frame)
    # cv2.waitKey(5000)
    

    # Object detection using YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    # print(len(outputs))
    # for out in outputs:

    #     print(out.shape)
    r0 = blob[0, 0, :, :]
    r = r0.copy()
    # cv2.imshow('blob', r)
    # cv2.waitKey(5000)

    # print(outs)
    # Process detections and pass them to the tracker
    # detections = post_process(frame, outs)
# 
    # NEW
    confidence=0.01
    max_conf=0.0
    detected=0
    for output in np.vstack(outputs):
        if output[4]>max_conf:
            max_conf=output[4]

        if output[4] > confidence:
            detected+=1
            x, y, w, h = output[:4]
            p0 = int((x-w/2)*416), int((y-h/2)*416)
            p1 = int((x+w/2)*416), int((y+h/2)*416)
            cv2.rectangle(r, p0, p1, (255,0,255), 3)
            print(x,",",y,",",w,",",h)
        cv2.imshow('blob', r)
        text = f'Bbox confidence={confidence}'
        cv2.displayOverlay('blob', text)
    print("max_conf:",max_conf," detected:",detected)
    # NEW

    # tracked_objects = tracker.update(detections)
    
    # Visualize tracked objects
    # for obj in tracked_objects:
    #     x, y, w, h, track_id = obj
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(frame, str(track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # frameS = cv2.resize(frame, (960, 540))
    # cv2.imshow('Object Tracker', frameS)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

