# Generate video of training with frame number in the corner (nothing else)

import cv2
from math import floor
from tqdm import tqdm

video = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4"

cap = cv2.VideoCapture(video)
if (cap.isOpened() == False):
  print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width,frame_height)

max_frames=10000
out = cv2.VideoWriter(f'origina_video_ann{max_frames}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

ret, img = cap.read()
# while ret == True:
for i in tqdm(range(max_frames)):
    cv2.putText(img, f'{i+1}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    out.write(img)
    ret, img = cap.read()

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()