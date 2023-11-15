import cv2
TrDict = {'dasiam':cv2.TrackerDaSiamRPN_create,
         'mil':cv2.TrackerMIL_create}

tracker=TrDict['mil']()
v = cv2.VideoCapture(r'development.mp4')
ret, frame = v.read()
frameS = cv2.resize(frame, (960, 540))

# cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv2.imshow('Frame',frameS)
bb = cv2.selectROI('Frame',frameS)
tracker.init(frameS,bb)
while True:
    ret, frame = v.read()
    frameS = cv2.resize(frame, (960, 540))
    if not ret:
        break
    (success,box) = tracker.update(frameS)
    if success:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frameS,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('Frame',frameS)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()
    
        