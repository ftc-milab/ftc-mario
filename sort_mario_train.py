# from sort_mario import Sort
from sort import Sort as SortOriginal
from sort_mofied import Sort as SortModified
from ultralytics import YOLO
import cv2
from tqdm import tqdm

prefix='train'
sort_type='modified'
if sort_type=='modified':
    Sort=SortModified
else:
    Sort=SortOriginal

# foi=[1110,6187,2480,7759,9677,9826]
foi=[5000]
foi_idx=0
foi_range=5000


max_frames_dict={'train':10000,'development':2259,'test':10400}
max_frames=max_frames_dict[prefix]
# max_frames=2259
# max_frames=10400
weights_fn='best-organizers.pt'
# weights_fn='8lb144e900-ma25mh0it0.pt'
# weights_fn='yolov8le1000b144gpu1_epoch900.pt'

model=YOLO(weights_fn)

ma,mh,it=2,0,0
print(f's{foi[0]}f{max_frames}_ma{ma} mi{mh} it{it}')

mot_tracker = Sort(max_age=ma,min_hits=mh,iou_threshold=it) 

video_dict={'train':"/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4",
            'development': "/work/marioeduardo-a/ftc/FTC-2024-data/Development/development.mp4",
            'test':"/work/marioeduardo-a/ftc/FTC-2024-data/Test/test.mp4"}
video=video_dict[prefix]

# prefix='sort_development'
# prefix='sort_mario_test'



cap = cv2.VideoCapture(video)
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with open(f'sort_{sort_type}_{prefix}_mf{max_frames}_raw_w{weights_fn.split(".")[0]}s{foi[0]}f{foi[-1]}fr{foi_range}ma{ma}mi{mh}it{it}.txt', 'w') as r,\
     open(f'sort_{sort_type}_{prefix}_mf{max_frames}_dt-tr_w{weights_fn.split(".")[0]}s{foi[0]}f{foi[-1]}fr{foi_range}ma{ma}mi{mh}it{it}.txt','w') as f,\
     open(f'sort_{sort_type}_{prefix}_mf{max_frames}_answer_w{weights_fn.split(".")[0]}s{foi[0]}f{foi[-1]}fr{foi_range}ma{ma}mi{mh}it{it}.txt','w') as a:
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
