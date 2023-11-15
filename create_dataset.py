import cv2
import os

video_file="/home/mario/ftc/FTC-2024-data/Train/train.mp4"
labels_file="/home/mario/ftc/FTC-2024-data/Train/train_gt_mot.txt"
original_folder='/home/mario/data/ftc-train/original"

max_frames=1000

if not os.path.exists(output_folder):
  os.makedirs(output_folder)

cap = cv2.VideoCapture(video_file)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened: " + str(cap.isOpened()))
print("Total frames: " + str(num_frames))

frames = 1
print("Start translate video to images.")
print("Video source: " + video_file)
print("Images destnation: " + original_folder)

while(cap.isOpened()):
    flag, frame = cap.read()
    if not flag:
      print("Process finished!")
      break
    else:
      imgname = 'train' + str(frames).zfill(len(str(num_frames))) + ".jpg"
      newPath = os.path.join(original_folder, imgname)
      cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
      # cv2.imencode('.jpg', frame)[1].tofile(newPath)
    if frames % 100 == 0:
      print(str(frames) + " frames finished...")
    if frames == max_frames:
      break
    frames += 1

cap.release()
#print("Last frame number: " + str(frames - 1))
print("Images outputed: " + str(len(os.listdir(original_folder))))