import numpy as np
import cv2
from math import floor
import os

from tqdm import tqdm

foi=[100]

# foi=[50]
foi_idx=0
foi_range=1000

max_frames=100

video = "/work/marioeduardo-a/ftc/FTC-2024-data/Test/test.mp4"

# raw_result_file = "sort_mario_test_skip-ud_raw_s1110f9826fr10ma25mi0it0.txt"
raw_result_file = "sort_mario_test_skip-ud_raw_wbest-organizers.pts1110f9826fr10ma2mi0it0.txt"
exp_id=raw_result_file.split('.')[0]

cap = cv2.VideoCapture(video)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width,frame_height, cap.get(cv2.CAP_PROP_FPS))

if (cap.isOpened() == False):
  print("Unable to read camera feed")
ss="_".join([str(i) for i in foi])
out = cv2.VideoWriter(f'outpy_{exp_id}_{ss}_yolo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

colors_gt = [(255, 255, 255),  (95,95,95), (0, 0, 0),#white,gray,black\
             (0, 0, 255), (0, 127, 0), (255, 0, 0), #red,green,blue
             (127, 127, 0), (255, 0, 255), (0, 255, 255), #cyan, magenta, yellow
             (64, 64, 159)]
colors_tr = [color for color in colors_gt]

with open(raw_result_file, 'r') as f:
    frames = 1
    ret, img = cap.read()
    for frames in range(max_frames):        
    # for frames in tqdm(range(max_frames)):        
        
        print_flag=(foi[foi_idx]-foi_range <= frames) and (frames <= (foi[foi_idx]+foi_range))

        # print(f'frames:{frames} foi:{foi[foi_idx]} foi_range:{foi_range} foi_idx:{foi_idx} '+\
        #       f'cond1:{(foi[foi_idx]-foi_range <= frames)} '+\
        #         f'cond2:{frames <= (foi[foi_idx]+foi_range)}')
        if print_flag:
            cv2.putText(img, f"{frames+1}  {exp_id} {foi[foi_idx]}", (50, 100 ), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2)    

            # put ids at the top right                    
            x0, dx = 1900, 50
            y0, dy = 100, 50
            for ii in range(len(colors_gt)):
                x = x0 + ii%10*dx
                y = y0 + (ii//10)*dy
                cv2.putText(img, f"{(ii+1):2}", (x, y ), cv2.FONT_HERSHEY_SIMPLEX, 1,colors_gt[ii], 2)    
        
        maxid=0
        while True:
            fine_num = f.tell()
            line = f.readline()
            line = line.strip()
            
            if len(line.split(" ")) == 1:
                break
            frame, bid, left, top, width, height, _, _, _, _ = line.split(" ")
            frame, bid, left, top, width, height = int(frame), int(bid), float(left), float(top), float(width), float(height)
            # print(f" fine_num:{fine_num} frame:{frame} bid:{bid}")
            if bid>=maxid:
                maxid=bid
            
            if frame > frames:
                f.seek(fine_num)
                frames = frame
                break
            
            x1,y1=floor(left),floor(top)
            x2,y2=(floor(left+width),floor(top+height))


            if print_flag:
                if bid<=10:
                    cv2.rectangle(img, (x1,y1), (x2,y2), colors_tr[(bid-1) %  len(colors_tr)], 2)
                elif bid<=20:
                    delta=3
                    cv2.rectangle(img, (x1-delta,y1-delta), (x2+delta,y2+delta), colors_tr[(bid-1) % len(colors_tr)], 1)
                    cv2.rectangle(img, (x1+delta,y1+delta), (x2-delta,y2-delta), colors_tr[(bid-1) %  len(colors_tr)], 1)
                else: 
                    delta=5
                    cv2.rectangle(img, (x1-delta,y1-delta), (x2+delta,y2+delta), colors_tr[(bid-1) % len(colors_tr)], 1)
                    cv2.rectangle(img, (x1,y1), (x2,y2), colors_tr[(bid-1) %  len(colors_tr)], 1)
                    cv2.rectangle(img, (x1+delta,y1+delta), (x2-delta,y2-delta), colors_tr[(bid-1) %  len(colors_tr)], 1)
                
           
            text = str(bid)
            x1,y1=floor(left+width+5),floor(top+height/4)
            if print_flag:
                cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colors_tr[(bid-1) %  len(colors_tr)], 2)
        
        if print_flag:
            cv2.putText(img, f"maxid:{maxid}", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, colors_tr[(maxid-1) %  len(colors_tr)], 2)

        if print_flag:
            out.write(img)
        ret, img = cap.read()
        if frames>=foi[foi_idx] and foi_idx<len(foi)-1:
            foi_idx+=1


# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()