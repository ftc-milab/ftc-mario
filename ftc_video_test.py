import numpy as np
import cv2
from math import floor
import os

from tqdm import tqdm

global_thickness=2
# video = "/work/marioeduardo-a/ftc/FTC-2024-data/Test/test.mp4"
video = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4"
# raw_result_file = os.path.join(result_folder, "raw.txt")
raw_result_file = "FISH_HOTA054.txt"

cap = cv2.VideoCapture(video)

if (cap.isOpened() == False):
  print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width,frame_height)
# max_frames=10
max_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# out = cv2.VideoWriter(os.path.join(tracker_folder, f'outpy{max_frames}.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out = cv2.VideoWriter(f'outpy_{max_frames}_yolo_{raw_result_file.split(".")[0]}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

colors = [(127, 0, 0), (0, 127, 0), (0, 0, 127), (127, 127, 0), (127, 0, 127), (0, 127, 127), (127, 63, 0), (63, 127, 0), (127, 0, 63), (63, 0, 127), (0, 127, 63), (0, 63, 127)]

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
            frame, bid, left, top, width, height, _, _, _, _ = line.split(",")
            frame, bid, left, top, width, height = int(frame), int(bid), float(left), float(top), float(width), float(height)

            if frame > frames:
                f.seek(fine_num)
                frames = frame
                break
                

            cv2.rectangle(img, (floor(left),floor(top)), (floor(left+width),floor(top+height)), colors[bid % len(colors)], global_thickness)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(bid)
            # cv2.putText(img, text, (floor(left + 20), floor(top + 20)), font, 1, colors[bid % len(colors)], 1)
            cv2.putText(img, text, (floor(left), floor(top-5)), font, 1, colors[bid % len(colors)], global_thickness)
        
        out.write(img)
        ret, img = cap.read()


# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()