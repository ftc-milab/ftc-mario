import numpy as np
import cv2
from math import floor
import os

from tqdm import tqdm

exp_id=""
max_frames=10000
global_thickness=2

video = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train.mp4"
label_file = "/work/marioeduardo-a/ftc/FTC-2024-data/Train/train_gt_mot.txt"

# TrackerName = f"Train{exp_id}"
# trackers_folder = f"TrackEvalYulun/data/trackers/mot_challenge/FISH{exp_id}-train"
# tracker_folder = os.path.join(trackers_folder, TrackerName)
# tracker_folder='OfficalYolo_euclidean_rematch_match_thr_1'
tracker_folder='OfficalYolo_euclidean_match_thr_1'
result_folder = os.path.join(tracker_folder, "data")
tracker_config_file = os.path.join(tracker_folder, "custom-tracker.yaml")
raw_result_file = os.path.join(result_folder, "raw.txt")
result_file = os.path.join(result_folder, f"FISH{exp_id}.txt")
match_file = os.path.join(tracker_folder, f"FISH{exp_id}-pedestrian-bestmatch.txt")
changes_file = os.path.join(tracker_folder, f"FISH{exp_id}-pedestrian-changes.txt")


yolo_exp_id="yolo_ow10k"
yolo_TrackerName= f"Train{yolo_exp_id}"
yolo_trackers_folder = f"TrackEvalYulun/data/trackers/mot_challenge/FISH{yolo_exp_id}-train"
yolo_tracker_folder = os.path.join(yolo_trackers_folder, yolo_TrackerName)
yolo_result_folder = os.path.join(yolo_tracker_folder, "data")
yolo_raw_result_file = os.path.join(yolo_result_folder, "raw.txt")

cap = cv2.VideoCapture(video)

if (cap.isOpened() == False):
  print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width,frame_height)

# out = cv2.VideoWriter(os.path.join(tracker_folder, f'outpy{max_frames}.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out = cv2.VideoWriter(os.path.join(tracker_folder, f'outpy_{exp_id}_{max_frames}_yolo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

# colors_gt = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 0, 128), (0, 255, 255), (255, 165, 0), (255, 192, 203)]
colors_gt = [(255, 255, 255),  (127,127,127), (0, 0, 0),#white,gray,black\
             (255, 0, 0), (0, 255, 0), (0, 0, 255), #red,green,blue
             (0, 255, 255), (255, 0, 255), (255, 255, 0), #cyan, magenta, yellow
             (64, 64, 159), #brown
             (255, 255, 255),  (127,127,127), (0, 0, 0),#white,gray,black\
             (127, 0, 0), (0, 127, 0), (0, 0, 127), ##dark red,green,blue
             (0, 127, 127), (127, 0, 127), (127, 127, 0), # dark cyan, magenta, yellow light
             (64, 64  , 92), #dark brown
             (255, 255, 255),  (127,127,127), (0, 0, 0),#white,gray,black\
             (255, 127, 127), (127, 255, 127), (127, 127, 255), ##light red,green,blue
             (127, 255, 255), (255, 127, 255), (255, 255, 127), # light cyan, magenta, yellow light
             (92, 92  , 165), #dark brown
             ] 
colors_tr = [color for color in colors_gt]
# print(len(colors_gt))
# print(len(colors_tr))

changes_num_print=20
# y_pos=[600+i*50 for i in range(changes_num_print)]
changes_pos=[650+i*50 for i in range(changes_num_print)]
changes_text=["" for i in range(changes_num_print)]
changes_color=[None for i in range(changes_num_print)]
changes_curr=0
changes_colors=[(127,0,0),(0,127,0),(0,0,127)]

changes_color_idx=0

changes=open(changes_file, 'r') 

with open(raw_result_file, 'r') as f, \
     open(label_file, 'r') as g, \
     open(match_file, 'r') as m, \
     open(yolo_raw_result_file, 'r') as yolof: 
    frames = 1
    ret, img = cap.read()
    # while ret == True:
    for i in tqdm(range(max_frames)):
        # print('frame: ',i)
        
        # put ids at the bottom
                    
        x0, dx = 1000, 50
        y0, dy = 50, 50
        for ii in range(len(colors_gt)):
            x = x0 + ii%10*dx
            y = y0 + (ii//10)*dy
            cv2.putText(img, f"{(ii+1):2}", (x, y ), cv2.FONT_HERSHEY_SIMPLEX, 1,colors_gt[ii], global_thickness)    
        #best match
        fine_num = m.tell() #current file position
        line = m.readline() 
        line = line.strip()
        if len(line.split(" ")) == 1:
            break
        frame = line.split(" ")[0]
        frame = int(frame)

        if frame > frames:
            m.seek(fine_num)
        else:
        #     # 标注文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            # text = f'{frame}\n' + functools.reduce(lambda x,y: x+'\n'+y, line.split(" ")[1:])
            # text = f'{frame}\n' + functools.reduce(lambda x,y: x+'\n'+y, line.split(" ")[1:])
            # text = "This is \n some text"
            matches=line.split(" ")[1:]
            match_arr=["" for i in range(11)]
            match_arr[0]=f'{frame}  {exp_id}'
            for match in matches:
                idx=int(match.split("-")[0])
                match_arr[idx]=match
            
            y0, dy = 50, 50
            for ii in range(11):
                y = y0 + ii*dy
                if match_arr[ii]=="":
                    cv2.putText(img, f"{ii}-None", (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255), global_thickness)    
                else:
                    cv2.putText(img, match_arr[ii], (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, global_thickness)    
            
            # define colors for tracker in the first frame to match
            if i==0:
                for match in matches:
                    idx,idy=map(int,match.split("-"))
                    colors_tr[idy-1% len(colors_tr)]=colors_gt[idx-1% len(colors_tr)]
                # print(colors_gt)
                # print(colors_tr)
            
            

        # changes file
        fine_num = changes.tell() #current file position
        line1 = changes.readline().strip()
        line2 = changes.readline().strip()
        
        frame = line1.split(" ")[0]
        if frame!='':
            frame = int(frame)
            if frame > frames:
                changes.seek(fine_num)
            else:
            #     # 标注文本
                font = cv2.FONT_HERSHEY_SIMPLEX
                changes_text[changes_curr%changes_num_print]=str(frame)+" "+line2
                changes_color[changes_curr%changes_num_print]=changes_colors[(changes_curr//changes_num_print)%len(changes_colors)]
                changes_curr=changes_curr+1
                # changes_color_idx=changes_curr//changes_num_print
            for j in range(changes_num_print):
                cv2.putText(img, changes_text[j], (50, changes_pos[j] ), font, 1, changes_color[j], global_thickness)
        else:
            changes.seek(fine_num)
            

        while True:
            fine_num = f.tell()
            line = f.readline()
            line = line.strip()
            if len(line.split(" ")) == 1:
                break
            frame, bid, left, top, width, height, _, _, _, _ = line.split(" ")
            frame, bid, left, top, width, height = int(frame), int(bid), float(left), float(top), float(width), float(height)

            if frame > frames:
                f.seek(fine_num)
                break
                #frames = frame
            if bid<=10:
                cv2.rectangle(img, (floor(left),floor(top)), (floor(left+width),floor(top+height)), colors_tr[(bid-1) %  len(colors_tr)], 2)
            elif bid<=20:
                cv2.rectangle(img, (floor(left)-3,floor(top)-3), (floor(left+width)+3,floor(top+height)+3), colors_tr[(bid-1) % len(colors_tr)], 1)
                cv2.rectangle(img, (floor(left)+3,floor(top)+3), (floor(left+width)-3,floor(top+height)-3), colors_tr[(bid-1) %  len(colors_tr)], 1)
            else: 
                cv2.rectangle(img, (floor(left)-5,floor(top)-5), (floor(left+width)+5,floor(top+height)+5), colors_tr[(bid-1) % len(colors_tr)], 1)
                cv2.rectangle(img, (floor(left),floor(top)), (floor(left+width),floor(top+height)), colors_tr[(bid-1) %  len(colors_tr)], 1)
                cv2.rectangle(img, (floor(left)+5,floor(top)+5), (floor(left+width)-5,floor(top+height)-5), colors_tr[(bid-1) %  len(colors_tr)], 1)
                
            
            # 标注文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(bid)
            # cv2.putText(img, text, (floor(left + 20), floor(top + 20)), font, 1, colors[(bid-1) %  len(colors)], 1)
            cv2.putText(img, text, (floor(left), floor(top-5)), font, 1, colors_tr[(bid-1) %  len(colors_tr)], global_thickness)
        
        while True:
            fine_num = yolof.tell()
            line = yolof.readline()
            line = line.strip()
            if len(line.split(" ")) == 1:
                break
            frame, bid, left, top, width, height, _, _, _, _ = line.split(" ")
            frame, bid, left, top, width, height = int(frame), int(bid), float(left), float(top), float(width), float(height)

            if frame > frames:
                yolof.seek(fine_num)
                break

            pts = np.array([[floor(left),floor(top+height)],[floor(left+width/2),floor(top)],[floor(left+width),floor(top+height)]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img, [pts], True, (127,127,127), global_thickness)
            # 标注文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(bid)
            
        while True:
            fine_num = g.tell()
            line = g.readline()
            line = line.strip()
            if len(line.split(" ")) == 1:
                break
            frame, bid, top, left, width, height, _, _, _, _ = line.split(" ")
            frame, bid, left, top, width, height = int(frame), int(bid), float(left), float(top), int(width), int(height)

            if frame > frames:
                g.seek(fine_num)
                frames = frame
                break

            cv2.circle(img, (floor(left+width/2),floor(top+height/2)), floor(width/2), colors_gt[(bid-1) %  len(colors_gt)], global_thickness)
            # 标注文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(bid)
            # cv2.putText(img, text, (floor(left+width/2),floor(top+height/2)), font, 1, colors[(bid-1) %  len(colors)], 1)
            # cv2.putText(img, text,  (floor(left), floor(top+20)), font, 1, colors[(bid-1) %  len(colors)], 1)
            # cv2.putText(img, text,  (floor(left), floor(top+25+25)), font, 1, colors[(bid-1) %  len(colors)], global_thickness)
            cv2.putText(img, text,  (floor(left+50), floor(top+50)), font, 1, colors_gt[(bid-1) %  len(colors_gt)], global_thickness)
        
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # text=f'Frame: {i:5}'
        # cv2.putText(img, text, (100,2000), font, 2, (0, 0, 0), 4)
        out.write(img)
        ret, img = cap.read()


# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()