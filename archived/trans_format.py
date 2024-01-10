import os

# frame, id, top, left, width, height, conference, x, y, z (if None, use -1 replace)

width = 2456
height = 2058


def extract_number(filename):
    # Extract the numeric part of the file name, e.g. "file10.txt" is extracted as 10
    numbers = filename.split("_")[1].split(".")[0]
    return int(numbers) if numbers else float('inf')


def trans_formate_mot_challenge(input_folder, output_path):
    with open(output_path, 'w', newline='') as res_file:
        # iter folder
        for root, dirs, files in os.walk(input_folder):
            sorted_files = sorted(files, key=extract_number)  # sort files by number
            for file_name in sorted_files:
                full_path = os.path.join(root, file_name)
                frame_id = file_name.split("_")[1].split(".")[0]
                with open(full_path, 'r') as f:
                    # read file content
                    lines = f.readlines()
                    lines = [line.strip() for line in lines]
                    for line in lines:
                        new_line = line.split(" ")
                        res = []
                        temp_top = (float(new_line[2]) * height) - (float(new_line[4]) * height / 2)
                        temp_left = (float(new_line[1]) * width) - (float(new_line[3]) * width / 2)
                        temp_width = float(new_line[3]) * width
                        temp_height = float(new_line[4]) * height
                        res.extend([frame_id, new_line[6], temp_top, temp_left, temp_width, temp_height, new_line[5], -1, -1, -1])
                        for item in res:
                            res_file.write(str(item) + " ")
                        res_file.write("\n")
        res_file.close()

if __name__ == '__main__':
    input_folder = r'/content/fish_tracking_ultralytics/fishtracking/runs/detect/predict/labels'  # change the right path
    output_path =  'answer.txt'  # you can rename it



    trans_formate_mot_challenge(input_folder, output_path)