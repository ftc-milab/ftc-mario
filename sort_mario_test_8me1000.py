from sort_mario import Sort
from ultralytics import YOLO
import cv2
from tqdm import tqdm



foi=[1110,6187,2480,7759,9677,9826]
foi_idx=0
foi_range=10

# max_frames=foi[-1]
max_frames=10400
# weights_fn='best-organizers.pt'
weights_fn='yolov8m_e1000s100_dp.pt'
# weights_fn='yolov8le1000b144gpu1_epoch900.pt'


# # Delete contents of sort_mario_tr_det.txt
# with open('sort_mario_tr_det.txt','w') as f:
#     f.write("")    

model=YOLO(weights_fn)

# ma,mh,it=17,1,0
ma,mh,it=2,1,0
print(f's{foi[0]}f{max_frames}_ma{ma} mi{mh} it{it}')

mot_tracker = Sort(max_age=ma,min_hits=mh,iou_threshold=it) 

# video = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4"
video = "/work/marioeduardo-a/ftc/FTC-2024-data/Test/test.mp4"

cap = cv2.VideoCapture(video)
with open(f'sort_mario_test_skip-ud_raw_w{weights_fn}s{foi[0]}f{foi[-1]}fr{foi_range}ma{ma}mi{mh}it{it}.txt', 'w') as r,\
     open(f'sort_mario_test_skip-ud_dt-tr_w{weights_fn}s{foi[0]}f{foi[-1]}fr{foi_range}ma{ma}mi{mh}it{it}.txt','w') as f,\
     open(f'sort_mario_test_skip-answer_w{weights_fn}s{foi[0]}f{foi[-1]}fr{foi_range}ma{ma}mi{mh}it{it}.txt','w') as a:
    # for frames in range(1,max_frames+1):
    for frames in tqdm(range(1,max_frames+1)):
        success, frame = cap.read()
        
        if not success:
            break
        
        
        # print(frames)
        results = model.predict(frame, verbose=False)
        # Get the boxes and track IDs
        # boxes = results[0].boxes.xywh.cpu()
        boxes = results[0].boxes.xyxy.cpu()

        track_bbs_ids = mot_tracker.update(boxes)
        if ((foi[foi_idx]-foi_range) <= frames) and (frames <= (foi[foi_idx]+foi_range)):
            f.write(f"frames:{frames:5} dt:{len(boxes):2} tr:{len(track_bbs_ids):2}"    +\
                f'{"        " if len(boxes)==10 else " DT!=10 "}'+\
                f'{"        " if len(track_bbs_ids)==10 else " TR!=10 "}'+\
                '\n')       

        # if (len(boxes)!=10 or len(track_bbs_ids)!=10):
        #     f.write(f"frames:{frames:5} dt:{len(boxes):2} tr:{len(track_bbs_ids):2}"    +\
        #             f'{"        " if len(boxes)==10 else " DT!=10 "}'+\
        #             f'{"        " if len(track_bbs_ids)==10 else " TR!=10 "}'+\
        #             '\n')       
        for track_bb_id in track_bbs_ids:
            box=track_bb_id[:4]

            track_id=int(track_bb_id[4])
            x1,y1,x2,y2=box
            x, y, w, h = x1,y1,x2-x1,y2-y1
            r.write(f'{frames} {track_id} {x} {y} {w} {h} -1 -1 -1 -1\n')
            a.write(f'{frames} {track_id} {y} {x} {w} {h} -1 -1 -1 -1\n')
                
                
        if frames>=foi[foi_idx] and foi_idx<len(foi)-1:
            foi_idx+=1
        frames += 1

cap.release()
cv2.destroyAllWindows()
