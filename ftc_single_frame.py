import numpy as np
import cv2
from math import floor
import os
import sys

if len(sys.argv)!=3:
    print("provide prefix and frame_number")
    exit()
prefix=sys.argv[1]

frame_number=int(sys.argv[2])
video_dict={'train':"sort_train_mf10000_raw_wbest-organizerss5000f5000fr5000ma2mi0it0.mp4",
                      'development':"sort_development_mf2259_raw_wbest-organizerss5000f5000fr5000ma2mi0it0.mp4",
                      'test':"sort_test_mf10400_raw_wbest-organizerss5000f5000fr5000ma2mi0it0.mp4",
                      'train_raw':"/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4",
                      'development_raw': "/work/marioeduardo-a/ftc/FTC-2024-data/Development/development.mp4",
                      'test_raw':"/work/marioeduardo-a/ftc/FTC-2024-data/Test/test.mp4"}
video=video_dict[prefix]
# video = "outpy_rematch_problem_sort_mario_test_mf10400_raw_wbest-organizerss5000f5000fr5000ma2mi0it0_356_683_812_1071_yolo.mp4"

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
ret, img = cap.read()
if prefix in ['train_raw','development_raw','test_raw']:
    fn=f"{prefix}_f{frame_number:05d}.jpeg"
else:
    fn=f"{video.split('.')[0]}_f{frame_number:05d}.jpeg"
print(f"REMOTEFILE=/work/marioeduardo-a/github/ftc-mario/{fn}")
cv2.imwrite(fn, img)
