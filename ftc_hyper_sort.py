# Generate "tracking" results using SORT tracker
# Input: hyperparameters are read from a csv file
# Output: BBs file; HOTA results; HOTA results summary in CSV file

# from sort import Sort
from sort import Sort
from sort_rematch import Sort as SortRematch
from sort_skip import Sort as SortSkip
from sort_rematch_skip import Sort as SortRematchSkip
from ultralytics.engine.results import Boxes

import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd

hyper_fn="param-sort-results.csv"

def create_tracker_file(exp_id=None,\
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
    global tracker_config_file
    global raw_result_file
    global result_file
    global match_file

    TrackerName = f"Train{exp_id}"
    trackers_folder = f"TrackEvalYulun/data/trackers/mot_challenge/FISH{exp_id}-train"
    tracker_folder = os.path.join(trackers_folder, TrackerName)
    result_folder = os.path.join(tracker_folder, "data")
    tracker_config_file = os.path.join(tracker_folder, "custom-tracker.yaml")
    raw_result_file = os.path.join(result_folder, "raw.txt")
    result_file = os.path.join(result_folder, f"FISH{exp_id}.txt")
    match_file = os.path.join(tracker_folder, f"FISH{exp_id}-pedestrian-bestmatch.txt")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    
def create_folders(exp_id=None,\
                        max_frames=50, \
                        weights_fn='best-organizers.pt',\
                        tracker_type = 'sort', \
                        max_age=1, \
                        min_hits=3, \
                        iou_threshold=0.3):
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

    # gt
    # TrackEvalYulun/data/gt/mot_challenge/FISH-train/FISH/gt/gt.txt
    fn=os.path.join(FISH,"gt/gt.txt")
    with open("TrackEvalYulun/data/gt/mot_challenge/FISH-train/FISH/gt/gt.txt","r") as f:
        with open(fn,"w") as g:
            frames=0
            for line in f:
                g.write(line)
                frames+=1
                if frames> max_frames*10-1:
                    break

    # #Perfect tracker
    cmd=f'cp {FISH}/gt/gt.txt "TrackEvalYulun/data/trackers/mot_challenge/FISH{exp_id}-train/PerfectTracker/data/FISH{exp_id}.txt"'
    os.system(cmd)
    

    #TrackEvalYulun/data/gt/mot_challenge/FISH-train/FISH/seqinfo.ini
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

    if not os.path.exists(hyper_fn):
        with open(hyper_fn,'w') as f:
            f.write('exp_id,max_frames,weights_fn,tracker_type,max_age,min_hits,iou_threshold,')
            f.write("HOTA,DetA,AssA,DetRe,DetPr,AssRe,AssPr,LocA,OWTA,HOTA(0),LocA(0),HOTALocA(0),Dets,GT_Dets,IDs,GT_IDs\n")

def track(exp_id=None,\
                        max_frames=50, \
                        weights_fn='best-organizers.pt',\
                        tracker_type = 'sort', \
                        max_age=1, \
                        min_hits=3, \
                        iou_threshold=0.3):
    model=YOLO(weights_fn)
    
    if tracker_type=='sort':
        mot_tracker = Sort(max_age=max_age,min_hits=min_hits,iou_threshold=iou_threshold) 
    elif tracker_type=='sort_rematch':
        mot_tracker = SortRematch(max_age=max_age,min_hits=min_hits,iou_threshold=iou_threshold) 
    elif tracker_type=='sort_skip':
        mot_tracker = SortSkip(max_age=max_age,min_hits=min_hits,iou_threshold=iou_threshold) 
    elif tracker_type=='sort_rematch_skip':
        mot_tracker = SortRematchSkip(max_age=max_age,min_hits=min_hits,iou_threshold=iou_threshold) 
    

    #create instance of SORT
    
    #tracking the model and write the result to txt
    # video = "/content/drive/MyDrive/FTC-2024-data/Train/train.mp4"
    video = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4"
    cap = cv2.VideoCapture(video)
    with open(raw_result_file, 'w') as r:
        # frames = 1
        # while cap.isOpened() and frames<=max_frames:
        for frames in tqdm(range(1,max_frames+1)):
            success, frame = cap.read()

            if success:
                # print(frames)
                # results = model.track(frame, persist=True, tracker=tracker_config_file,verbose=False)
                # results = model.track(frame, persist=True, tracker=tracker_config_file)
                results = model.predict(frame, verbose=False)
                # Get the boxes and track IDs
                # boxes = results[0].boxes.xywh.cpu()
                boxes = results[0].boxes.xyxy.cpu()
                # track_bbs_ids = results[0].boxes.id.int().cpu().tolist()
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

def translate_results():
    with open(raw_result_file, 'r') as f:
        with open(result_file, 'w') as g:
            for line in f:
                # print(line)
                line = line.strip()
                # print(line)
                # print(line.split(" "))
                frame, bid, left, top, width, height, _, _, _, _ = line.split(" ")
                g.write(f'{frame}, {bid}, {left}, {top}, {width}, {height}, -1, -1, -1, -1\n')
import subprocess


def hota(exp_id=None,\
                        max_frames=50, \
                        weights_fn='best-organizers.pt',\
                        tracker_type = 'sort', \
                        max_age=1, \
                        min_hits=3, \
                        iou_threshold=0.3):
    cmd=f"cd TrackEvalYulun && python scripts/run_mot_challenge.py --BENCHMARK FISH{exp_id} \
        --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL {TrackerName} --METRICS HOTA --USE_PARALLEL False \
        --NUM_PARALLEL_CORES 1 --DO_PREPROC False 1>/dev/null"
    print(cmd)
    os.system(cmd)


def read_hota(exp_id=None,\
                        max_frames=50, \
                        weights_fn='best-organizers.pt',\
                        tracker_type = 'sort', \
                        max_age=1, \
                        min_hits=3, \
                        iou_threshold=0.3):
    fn=os.path.join(tracker_folder,"pedestrian_summary.txt")
    with open(fn) as f:

        line1=f.readline()
        # f.readline
        # print('line1:',line1)
        line2=f.readline()
        # print('line2:',line2)
        line= ','.join(line2.split())

        with open(hyper_fn,'a') as g:
            g.write(f"{exp_id},{max_frames},{weights_fn},{tracker_type},{max_age},{min_hits},{iou_threshold},")
            g.write(line)
            g.write('\n')



df=pd.read_csv('param-sort.csv')

for index, row in df.iterrows():
    print('(',index+1,'/',len(df),') rows')
                        
    create_tracker_file(exp_id=row.exp_id,\
                        max_frames=row.max_frames, \
                        weights_fn=row.weights_fn,\
                        tracker_type = row.tracker_type, \
                        max_age = row.max_age, \
                        min_hits = row.min_hits, \
                        iou_threshold = row.iou_threshold)
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
    hota(exp_id=row.exp_id,\
                        max_frames=row.max_frames, \
                        weights_fn=row.weights_fn,\
                        tracker_type = row.tracker_type, \
                        max_age = row.max_age, \
                        min_hits = row.min_hits, \
                        iou_threshold = row.iou_threshold)
    read_hota(exp_id=row.exp_id,\
                        max_frames=row.max_frames, \
                        weights_fn=row.weights_fn,\
                        tracker_type = row.tracker_type, \
                        max_age = row.max_age, \
                        min_hits = row.min_hits, \
                        iou_threshold = row.iou_threshold)

