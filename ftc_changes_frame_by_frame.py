#use the output txt to generate the video for training video

import cv2
from math import floor
from tqdm import tqdm
import os

exp_id="ow_dp"
# video = "TrackEval/data/trackers/mot_challenge/FISHow_dp-train/Trainow_dp/outpy10000.mp4"
video = "TrackEval/data/trackers/mot_challenge/FISHow_dp-train/Trainow_dp/outpy10000_yolo.mp4"
# video = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4"
# label_file = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train_gt_mot.txt"

TrackerName = f"Train{exp_id}"
trackers_folder = f"TrackEval/data/trackers/mot_challenge/FISH{exp_id}-train"
tracker_folder = os.path.join(trackers_folder, TrackerName)
result_folder = os.path.join(tracker_folder, "data")
tracker_config_file = os.path.join(tracker_folder, "custom-tracker.yaml")
raw_result_file = os.path.join(result_folder, "raw.txt")
result_file = os.path.join(result_folder, f"FISH{exp_id}.txt")
match_file = os.path.join(tracker_folder, f"FISH{exp_id}-pedestrian-bestmatch.txt")
changes_file = os.path.join(tracker_folder, f"FISH{exp_id}-pedestrian-changes.txt")
changes_folder = os.path.join(tracker_folder, "changes5fr")
# changes_folder= "TrackEval/data/trackers/mot_challenge/FISHow_dp-train/Trainow_dp/changes"

if not os.path.exists(changes_folder):
    os.makedirs(changes_folder)


cap = cv2.VideoCapture(video)
with open(changes_file, 'r') as changes:
    lines=changes.readlines()
    for i in range(0,len(lines),2):
        # line1=lines[i]
        # line2=lines[i+1]
        frame=int(lines[i].strip().split(' ')[0])
        print('frame',frame)
        # path=os.path.join(changes_folder)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        
        # for j in [-1,0,1]:
        for j in [-2,-1,0,1,2]:
            if (frame+j)>=1 and (frame+j)<=10000:
                # print(frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,frame+j-1))
                ret, img = cap.read()
                cv2.imwrite(os.path.join(changes_folder,f"{frame+j:05d}.jpeg"), img)
        # print('i',i)
     

cap.release()
