import numpy as np
import cv2
from math import floor
import os

from tqdm import tqdm

# exp_id="sortma2mh0it007mf10400"
max_frames=10400

global_thickness=2

video = "/work/marioeduardo-a/ftc/FTC-2024-data/Test/test.mp4"
# raw_result_file = "TrackEvalYulun/data/trackers/mot_challenge/FISHsortma2mh0it007-train/Trainsortma2mh0it007/data/raw.txt"
raw_result_file = "TrackEvalYulun/data/trackers/mot_challenge/FISHsub-sortma2mh0it007mf10400-train/Trainsub-sortma2mh0it007mf10400/data/raw.txt"
# raw_result_file = "FISH_HOTA054.txt"

cap = cv2.VideoCapture(video)

if (cap.isOpened() == False):
  print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width,frame_height)
# max_frames=10
# max_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# out = cv2.VideoWriter(os.path.join(tracker_folder, f'outpy{max_frames}.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out = cv2.VideoWriter(f'outpy_{max_frames}_yolo_{raw_result_file.split(".")[0]}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

colors_tr = [(255, 255, 255),  (127,127,127), (0, 0, 0),#white,gray,black\
             (255, 0, 0), (0, 255, 0), (0, 0, 255), #red,green,blue
             (0, 255, 255), (255, 0, 255), (255, 255, 0), #cyan, magenta, yellow
             (64, 64, 159), #brown
             (255, 255, 255),  (127,127,127), (0, 0, 0),#white,gray,black\
             (127, 0, 0), (0, 127, 0), (0, 0, 127), ##dark red,green,blue
             (0, 127, 127), (127, 0, 127), (127, 127, 0), # dark cyan, magenta, yellow light
             (64, 64  , 92), #dark brown
             (255, 255, 255),  (127,127,127), (0, 0, 0),#white,gray,black\
             (255, 127, 127), (127, 255, 127), (127, 127, 255), ##light red,green,blue
             (127, 255, 255), (255, 127, 255), (255, 255, 127), # light cyan, magenta, yellow light
             (92, 92  , 165), #dark brown
             ] 


with open(raw_result_file, 'r') as f:
    frames = 1
    ret, img = cap.read()
    
    for i in tqdm(range(max_frames)):
                
        while True:
            fine_num = f.tell()
            line = f.readline()
            # print(line)
            line = line.strip()
            if len(line.split(" ")) == 1:
                break
            frame, bid, left, top, width, height, _, _, _, _ = line.split(" ")
            frame, bid, left, top, width, height = int(frame), int(bid), float(left), float(top), float(width), float(height)

            if frame > frames:
                f.seek(fine_num)
                frames = frame
                break
                

            if frame > frames:
                f.seek(fine_num)
                break
                #frames = frame
            if bid<=10:
                cv2.rectangle(img, (floor(left),floor(top)), (floor(left+width),floor(top+height)), colors_tr[(bid-1) %  len(colors_tr)], 2)
            elif bid<=20:
                cv2.rectangle(img, (floor(left)-3,floor(top)-3), (floor(left+width)+3,floor(top+height)+3), colors_tr[(bid-1) % len(colors_tr)], 1)
                cv2.rectangle(img, (floor(left)+3,floor(top)+3), (floor(left+width)-3,floor(top+height)-3), colors_tr[(bid-1) %  len(colors_tr)], 1)
            else: 
                cv2.rectangle(img, (floor(left)-5,floor(top)-5), (floor(left+width)+5,floor(top+height)+5), colors_tr[(bid-1) % len(colors_tr)], 1)
                cv2.rectangle(img, (floor(left),floor(top)), (floor(left+width),floor(top+height)), colors_tr[(bid-1) %  len(colors_tr)], 1)
                cv2.rectangle(img, (floor(left)+5,floor(top)+5), (floor(left+width)-5,floor(top+height)-5), colors_tr[(bid-1) %  len(colors_tr)], 1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(bid)
            # cv2.putText(img, text, (floor(left + 20), floor(top + 20)), font, 1, colors[bid % len(colors)], 1)
            cv2.putText(img, text, (floor(left), floor(top-5)), font, 1, colors_tr[(bid-1) % len(colors_tr)], global_thickness)
        
        out.write(img)
        ret, img = cap.read()


# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()