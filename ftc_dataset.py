import os 
import cv2
from tqdm import tqdm
import shutil
import random

# train_p,valid_p,test_p=0.75,0.15,0.10
train_p,valid_p,test_p=0.70,0.3,0.0
data = f'/work/marioeduardo-a/ftc-t{int(train_p*100)}v{int(valid_p*100)}t{int(test_p*100)}'
print(f"data folder: {data}")
if not os.path.exists(data):
    os.makedirs(data)

images=os.path.join(data,"images")
labels=os.path.join(data,"labels")

images_original = os.path.join(images,'original')
labels_original = os.path.join(labels,'original')

images_train = os.path.join(images,'train')
labels_train = os.path.join(labels,'train')

images_valid = os.path.join(images,'valid')
labels_valid = os.path.join(labels,'valid')

images_test = os.path.join(images,'test')
labels_test = os.path.join(labels,'test')


video_train='/work/marioeduardo-a/FTC-2024-data/Train/train.mp4'
labels_train_file='/work/marioeduardo-a/FTC-2024-data/Train/train_gt_mot.txt'
# video_development=os.path.join(data,'FTC-2024-data/Development/development.mp4')
# video_test=os.path.join(data,'FTC-2024-data/Test/test.mp4')

capture = cv2.VideoCapture(video_train)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Copy images from drive ioriginalf they exist there
if not os.path.exists(images_original):
    os.makedirs(images_original)

    # total_frames=100

    for f in tqdm(range(1,total_frames+1)):
        _,frame=capture.read()
        cv2.imwrite(os.path.join(images_original,str(f).zfill(6)+".jpg"),frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # cv2.imencode('.jpg', frame)[1].tofile(newPath)

    capture.release()
    #print("Last frame number: " + str(frames - 1))
    print("Images outputed: " + str(len(os.listdir(images_original))))
#extract labels into each frames to label_foler

if not os.path.exists(labels_original):
    video_width = 0
    video_height = 0
    cap = cv2.VideoCapture(video_train)
    if cap.isOpened():
        video_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    cap.release()

    print("Video width: " + str(video_width))
    print("Video height: " + str(video_height))

    
    os.makedirs(labels_original)


    print("Start translate labels into YOLO format.")
    print("Label source: " + labels_train_file)
    print("Label destnation: " + labels_original)

    current_frame = -1
    with open(labels_train_file) as f:
        current_file = 0
        for line in f:
            line = line.strip()
            frame, bid, top, left, width, height, _, _, _, _ = line.split(" ")
            frame, bid, left, top, width, height = int(frame), int(bid), float(left), float(top), int(width), int(height)
            center_x = left + width / 2
            center_y = top + height/ 2
            rel_center_x = center_x / video_width
            rel_center_y = center_y / video_height
            rel_width = width / video_width
            rel_height = height / video_height
            if frame != current_frame:
                if current_file != 0:
                    current_file.close()
                if frame == total_frames + 1:
                    break
                current_file = open(os.path.join(labels_original, str(frame).zfill(6) + ".txt"), 'w')
            #current_file.write("class_id center_x center_y bbox_width bbox_height\n")
            current_frame = frame
            current_file.write("0 " + str(rel_center_x) + " " + str(rel_center_y) + " " + str(rel_width) + " " + str(rel_height) + "\n")

        current_file.close()
    print("Translation finished!")
    print("Last frame number: " + str(current_frame))
    print("Labels outputed: " + str(len(os.listdir(labels_original))))

# video_development=os.path.join(data,'FTC-2024-data/Development/development.mp4')
# video_test=os.path.join(data,'FTC-2024-data/Test/test.mp4')


if not os.path.exists(images_train):
    ids =[i for i in range(1,total_frames+1)]
    random.shuffle(ids)

    train_offset = int(train_p*total_frames)
    valid_offset = int(train_offset+valid_p*total_frames)
    test_offset  = int(valid_offset+test_p*total_frames)
    print("train_offset:",train_offset)
    print("valid_offset:",valid_offset)
    print("test_offset:",test_offset)
    print()
    train_ids = ids[:train_offset]
    valid_ids = ids[train_offset:valid_offset]
    test_ids  = ids[valid_offset:]


    print('len(train):',len(train_ids))
    print('len(valid_ids):',len(valid_ids))
    print('len(train):',len(test_ids))
    print()
    # Create folders
    types=[images,labels]
    folders=["train","valid","test"]
    

    for t in types:
        for ff in folders:
            folder_fn = os.path.join(t,ff)
            if not os.path.exists(folder_fn):
                os.makedirs(folder_fn)

    #move training set to image_train_folder and label_train_folder
    for i in range(total_frames):
        if i<train_offset:
            destination_folder= "train"
        elif i<valid_offset:
            destination_folder= "valid"
        else:
            destination_folder= "test"
        destination_images=os.path.join(images,destination_folder)
        destination_labels=os.path.join(labels, destination_folder)

        source      = os.path.join(images_original,str(ids[i]).zfill(6)+".jpg")
        destination = os.path.join(destination_images,str(ids[i]).zfill(6)+".jpg")
        shutil.copy(source, destination)

        source      = os.path.join(labels_original,str(ids[i]).zfill(6)+".txt")
        destination = os.path.join(destination_labels,str(ids[i]).zfill(6)+".txt")

        shutil.copy(source, destination)


    # Validate sizes of folders
    #image count in colab vm
    print("Train images: ",len(os.listdir(images_train)))
    print("Train labels: ",len(os.listdir(labels_train)))
    print("Valid images: ",len(os.listdir(images_valid)))
    print("Valid labels: ",len(os.listdir(labels_valid)))
    print("Test images: ",len(os.listdir(images_test)))
    print("Test labels: ",len(os.listdir(labels_test)))
    