from sort_mario import Sort
from ultralytics import YOLO
import cv2
from tqdm import tqdm


max_frames=5
weights_fn='best-organizers.pt'

model=YOLO(weights_fn)

mot_tracker = Sort(max_age=2,min_hits=0,iou_threshold=0) 
raw_result_file='sort_mario_test_output.txt'
video = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4"
cap = cv2.VideoCapture(video)
with open(raw_result_file, 'w') as r:
    # frames = 1
    # while cap.isOpened() and frames<=max_frames:
    for frames in range(1,max_frames+1):
    # for frames in tqdm(range(1,max_frames+1)):
        success, frame = cap.read()

        if success:
            # print(frames)
            results = model.predict(frame, verbose=False)
            # Get the boxes and track IDs
            # boxes = results[0].boxes.xywh.cpu()
            boxes = results[0].boxes.xyxy.cpu()
            track_bbs_ids = mot_tracker.update(boxes)
            # if len(boxes) != 10:
            #     print(f'frames: {frames} len(boxes):{len(boxes)}')
            # if(len(boxes<10))
            # print(len(boxes))
            # exit()
            # print(track_bbs_ids)
            # print('res',results)
            # print('bb',boxes)
            # print('ttbb',track_bbs_ids)
            # exit()
            # Plot the tracks
            # for box, track_id in zip(boxes, track_ids):
            for track_bb_id in track_bbs_ids:
                box=track_bb_id[:4]

                track_id=int(track_bb_id[4])
                x1,y1,x2,y2=box
                x, y, w, h = x1,y1,x2-x1,y2-y1
                # x -= w/2
                # y -= h/2
                r.write(f'{frames} {track_id} {x:.2f} {y:.2f} {w:.2f} {h:.2f} -1 -1 -1 -1\n')
        else:
            # Break the loop if the end of the video is reached
            break
        frames += 1
# Release the video capture object and close the display window
# cap.close()
cap.release()
cv2.destroyAllWindows()
