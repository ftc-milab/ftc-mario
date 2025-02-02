# Code based on ftc_hyper_sor but without HOTA evaluation
# The main point is to generate the answer.txt file
# TODO: remove unnecessary functions; or better, move "shared" functions in hyper and submission to single file

from sort import Sort
from ultralytics.engine.results import Boxes
from math import floor
import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd

    
def create_folders(exp_id=None,\
                        max_frames=50, \
                        weights_fn='best-organizers.pt',\
                        tracker_type = 'sort', \
                        max_age=1, \
                        min_hits=3, \
                        iou_threshold=0.3):

    global TrackerName
    global trackers_folder
    global tracker_folder
    global result_folder
    global raw_result_file
    global answer_file
    global result_file
    

    # paths
    TrackerName = f"Train{exp_id}"
    trackers_folder = f"TrackEvalYulun/data/trackers/mot_challenge/FISH{exp_id}-train"
    tracker_folder = os.path.join(trackers_folder, TrackerName)
    result_folder = os.path.join(tracker_folder, "data")
    raw_result_file = os.path.join(result_folder, "raw.txt")
    answer_file=os.path.join(result_folder, "answer.txt")
    result_file = os.path.join(result_folder, f"FISH{exp_id}.txt")
    

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # FISH folder
    FISH=f"TrackEvalYulun/data/gt/mot_challenge/FISH{exp_id}-train/FISH{exp_id}"

    #gt folder
    fn=os.path.join(FISH,"gt")
    if not os.path.exists(fn):
        os.makedirs(fn)

    #PerfectTracker folder
    fn=f"TrackEvalYulun/data/trackers/mot_challenge/FISH{exp_id}-train/PerfectTracker/data"
    if not os.path.exists(fn):
        os.makedirs(fn)

    # gt data
    fn=os.path.join(FISH,"gt/gt.txt")
    with open("TrackEvalYulun/data/gt/mot_challenge/FISH-train/FISH/gt/gt.txt","r") as f:
        with open(fn,"w") as g:
            frames=0
            for line in f:
                g.write(line)
                frames+=1
                if frames> max_frames*10-1:
                    break

    # Perfect tracker data
    cmd=f'cp {FISH}/gt/gt.txt "TrackEvalYulun/data/trackers/mot_challenge/FISH{exp_id}-train/PerfectTracker/data/FISH{exp_id}.txt"'
    os.system(cmd)
    

    # seqinfo
    fn=os.path.join(FISH, "seqinfo.ini")
    with open(fn,"w") as f:
        f.write("[Sequence]\n")
        f.write(f"name=FISH{exp_id}\n")
        f.write(f"seqLength={max_frames}")

    # seqmaps
    seqmaps= "TrackEvalYulun/data/gt/mot_challenge/seqmaps"
    fns=[os.path.join(seqmaps, f'FISH{exp_id}-all.txt'),\
        os.path.join(seqmaps, f'FISH{exp_id}-test.txt'),\
        os.path.join(seqmaps, f'FISH{exp_id}-train.txt')]
    for fn in fns:
        with open(fn,"w") as f:
            f.write("NAME\n")
            f.write(f"FISH{exp_id}")


def track(exp_id=None,\
                        max_frames=50, \
                        weights_fn='best-organizers.pt',\
                        tracker_type = 'sort', \
                        max_age=1, \
                        min_hits=3, \
                        iou_threshold=0.3):
    model=YOLO(weights_fn)
    
    
    # if 'mot_tracker' in locals():
    #     # delete in case it was defined before; this will prevent IDs starting from the
    #     # previous last ID in the previous row
    #     Sort.KalmanBoxTracker.count=0
    #     del mot_tracker
    #create instance of SORT
    
    mot_tracker = Sort(max_age=max_age,min_hits=min_hits,iou_threshold=iou_threshold) 
    mot_tracker.reset_count()
    #tracking the model and write the result to txt
    # video = "/content/drive/MyDrive/FTC-2024-data/Train/train.mp4"
    video = "/work/marioeduardo-a/ftc/FTC-2024-data/Test/test.mp4"
    cap = cv2.VideoCapture(video)
    
    print('frames:',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    with open(raw_result_file, 'w') as r:
        # frames = 1
        # while cap.isOpened() and frames<=max_frames:
        for frames in tqdm(range(1,max_frames+1)):
            success, frame = cap.read()

            if success:
                results = model.predict(frame, verbose=False)
                boxes = results[0].boxes.xyxy.cpu()
                track_bbs_ids = mot_tracker.update(boxes)
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
                    
                    # print(f"id:{track_id} x1:{x1:.4f} y1:{y1:.4f} x2:{x2:.4f} y2:{y2:.4f}",end="")
                    # print(f"x:{x:.4f} y:{y1:.4f} w:{w:.4f} h:{h:.4f}")
                    # print()
                    r.write(f'{frames} {track_id} {x} {y} {w} {h} -1 -1 -1 -1\n')
                
                # cv2.imwrite("debug_sub.jpeg", frame)
                # exit()
            else:
                # Break the loop if the end of the video is reached
                break
            frames += 1
    # Release the video capture object and close the display window
    # cap.close()
    cap.release()
    cv2.destroyAllWindows()

def translate_results():
    with open(raw_result_file, 'r') as f,\
            open(result_file, 'w') as g,\
            open(answer_file, 'w') as answer:
        for line in f:
            line = line.strip()
            frame, bid, left, top, width, height, _, _, _, _ = line.split(" ")
            g.write(f'{frame}, {bid}, {left}, {top}, {width}, {height}, -1, -1, -1, -1\n')
            answer.write(f'{frame} {bid} {top} {left} {width} {height} -1 -1 -1 -1\n')
    

df=pd.read_csv('param-sort-submission.csv')

for index, row in df.iterrows():
    print('(',index+1,'/',len(df),') rows')
                        
    create_folders(exp_id=row.exp_id,\
                        max_frames=row.max_frames, \
                        weights_fn=row.weights_fn,\
                        tracker_type = row.tracker_type, \
                        max_age = row.max_age, \
                        min_hits = row.min_hits, \
                        iou_threshold = row.iou_threshold)
    track(exp_id=row.exp_id,\
                        max_frames=row.max_frames, \
                        weights_fn=row.weights_fn,\
                        tracker_type = row.tracker_type, \
                        max_age = row.max_age, \
                        min_hits = row.min_hits, \
                        iou_threshold = row.iou_threshold)
    translate_results()