import numpy as np
import cv2
from math import floor
import os

from tqdm import tqdm

exp_id="sub-sortma2mh0it007mf10400"
max_frames=10400

global_thickness=2

video = "/work/marioeduardo-a/ftc/FTC-2024-data/Test/test.mp4"

TrackerName = f"Train{exp_id}"
trackers_folder = f"TrackEvalYulun/data/trackers/mot_challenge/FISH{exp_id}-train"
tracker_folder = os.path.join(trackers_folder, TrackerName)
result_folder = os.path.join(tracker_folder, "data")
tracker_config_file = os.path.join(tracker_folder, "custom-tracker.yaml")
raw_result_file = os.path.join(result_folder, "raw.txt")
result_file = os.path.join(result_folder, f"FISH{exp_id}.txt")

cap = cv2.VideoCapture(video)

if (cap.isOpened() == False):
  print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width,frame_height)


# out = cv2.VideoWriter(os.path.join(tracker_folder, f'outpy{max_frames}.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out = cv2.VideoWriter(os.path.join(tracker_folder, f'outpy_{exp_id}_{max_frames}_yolo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

# colors_gt = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 0, 128), (0, 255, 255), (255, 165, 0), (255, 192, 203)]
colors_gt = [(255, 255, 255),  (95,95,95), (0, 0, 0),#white,gray,black\
             (0, 0, 255), (0, 127, 0), (255, 0, 0), #red,green,blue
             (127, 127, 0), (255, 0, 255), (0, 255, 255), #cyan, magenta, yellow
             (64, 64, 159)]
# , #brown
#              (255, 255, 255),  (95,95,95), (0, 0, 0),#white,gray,black\
#              (0, 0, 127), (0, 127, 0), (127, 0, 0), ##dark red,green,blue
#              (127, 127, 0), (127, 0, 127), (0, 127, 127), # dark cyan, magenta, yellow light
#              (64, 64  , 92), #dark brown
#              (255, 255, 255),  (95,95,95), (0, 0, 0),#white,gray,black\
#              (127, 127, 255), (127, 255, 127), (255, 127, 127), ##light red,green,blue
#              (255, 255, 127), (255, 127, 255), (127, 255, 255), # light cyan, magenta, yellow light
#              (92, 92  , 165), #dark brown
#              ] 
colors_tr = [color for color in colors_gt]
# print(len(colors_gt))
# print(len(colors_tr))


with open(raw_result_file, 'r') as f:
    frames = 1
    ret, img = cap.read()
    # while ret == True:
    # for i in range(max_frames):
    for i in tqdm(range(max_frames)):
        # print('frame: ',i)
        
        cv2.putText(img, f"{i+1}  {exp_id}", (50, 100 ), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2)    

        # put ids at the top right                    
        x0, dx = 1900, 50
        y0, dy = 100, 50
        for ii in range(len(colors_gt)):
            x = x0 + ii%10*dx
            y = y0 + (ii//10)*dy
            cv2.putText(img, f"{(ii+1):2}", (x, y ), cv2.FONT_HERSHEY_SIMPLEX, 1,colors_gt[ii], global_thickness)    
        
        maxid=0
        while True:
            fine_num = f.tell()
            line = f.readline()
            line = line.strip()
            # print(f" fine_num:{fine_num} line:{line} len(line.split(" ")):{len(line.split(" "))}")
            if len(line.split(" ")) == 1:
                break
            frame, bid, left, top, width, height, _, _, _, _ = line.split(" ")
            frame, bid, left, top, width, height = int(frame), int(bid), float(left), float(top), float(width), float(height)
            # print(f" fine_num:{fine_num} frame:{frame} bid:{bid}")
            if bid>=maxid:
                maxid=bid
            
            if frame > frames:
                f.seek(fine_num)
                frames = frame
                break
            
            x1,y1=floor(left),floor(top)
            x2,y2=(floor(left+width),floor(top+height))
              
            if bid<=10:
                cv2.rectangle(img, (x1,y1), (x2,y2), colors_tr[(bid-1) %  len(colors_tr)], 2)
            elif bid<=20:
                delta=3
                cv2.rectangle(img, (x1-delta,y1-delta), (x2+delta,y2+delta), colors_tr[(bid-1) % len(colors_tr)], 1)
                cv2.rectangle(img, (x1+delta,y1+delta), (x2-delta,y2-delta), colors_tr[(bid-1) %  len(colors_tr)], 1)
            else: 
                delta=5
                cv2.rectangle(img, (x1-delta,y1-delta), (x2+delta,y2+delta), colors_tr[(bid-1) % len(colors_tr)], 1)
                cv2.rectangle(img, (x1,y1), (x2,y2), colors_tr[(bid-1) %  len(colors_tr)], 1)
                cv2.rectangle(img, (x1+delta,y1+delta), (x2-delta,y2-delta), colors_tr[(bid-1) %  len(colors_tr)], 1)
            
            
            # 标注文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(bid)
            # cv2.putText(img, text, (floor(left + 20), floor(top + 20)), font, 1, colors[(bid-1) %  len(colors)], 1)
            x1,y1=floor(left+width+5),floor(top+height/4)
            
            # cv2.putText(img, text, (floor(left + 20), floor(top + 20)), font, 1, colors[(bid-1) %  len(colors)], 1)
            cv2.putText(img, text, (x1, y1), font, 1, colors_tr[(bid-1) %  len(colors_tr)], 2)
        
        cv2.putText(img, f"maxid:{maxid}", (250, 150), font, 1, colors_tr[(maxid-1) %  len(colors_tr)], 2)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # text=f'Frame: {i:5}'
        # cv2.putText(img, text, (100,2000), font, 2, (0, 0, 0), 4)
        out.write(img)
        ret, img = cap.read()


# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()