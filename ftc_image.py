#use the output txt to generate the video for training video

import cv2
from math import floor
import os
from tqdm import tqdm

exp_id="ow_dp"

video = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4"
label_file = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train_gt_mot.txt"

TrackerName = f"Train{exp_id}"
trackers_folder = f"TrackEval/data/trackers/mot_challenge/FISH{exp_id}-train"
tracker_folder = os.path.join(trackers_folder, TrackerName)
result_folder = os.path.join(tracker_folder, "data")
tracker_config_file = os.path.join(tracker_folder, "custom-tracker.yaml")
raw_result_file = os.path.join(result_folder, "raw.txt")
result_file = os.path.join(result_folder, f"FISH{exp_id}.txt")
match_file = os.path.join(tracker_folder, f"FISH{exp_id}-pedestrian-bestmatch.txt")
changes_file = os.path.join(tracker_folder, f"FISH{exp_id}-pedestrian-changes.txt")

cap = cv2.VideoCapture(video)

if (cap.isOpened() == False):
  print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width,frame_height)
# max_frames=100
frame_number=596

import cv2
# cap = cv2.VideoCapture(videopath)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
res, frame = cap.read()
print("before imwrite")
cv2.imwrite("output.jpeg", frame)
print("after imwrite")

from ultralytics import YOLO
model=YOLO('/work/marioeduardo-a/ftc/models/best-organizers.pt')
print("after YOLO")
results=model.predict(frame)
print(results)

