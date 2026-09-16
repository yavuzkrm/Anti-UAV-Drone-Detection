import os
import json

BASE_LABEL_DIR = './datasets/labels/train'
IMAGE_DIMENSIONS = {
    "infrared": [640, 512],
    "visible": [1920, 1080]
}

def savetxt(video_path):
    try:
        with open(video_path, 'r') as file:
            data = json.load(file)

        gt_data = data['gt_rect']
        
        video_path = video_path.replace('\\', '/')
        parts = video_path.split('/')
        session_id = parts[4]
        label_class = os.path.basename(video_path).split('.')[0]

        for frame_idx, gt_data_item in enumerate(gt_data):

            if not gt_data_item:
                continue

            norm_str = yolonormalization(gt_data_item, label_class)

            label_dir = os.path.join(BASE_LABEL_DIR, session_id)
            
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            frame_name = f"{label_class}I{str(frame_idx).zfill(4)}"
            txt_path = os.path.join(label_dir, f"{frame_name}.txt")
        
            with open(txt_path, 'w') as txtfile:
                txtfile.write(norm_str)
    except Exception as e:
        print(f"Error: {e}, Video: {video_path}")


def yolonormalization(gt_data_item, label_class):
    class_num = 0

    image_width, image_height = IMAGE_DIMENSIONS[label_class]

    center_x_normalized = (gt_data_item[0] + gt_data_item[2]/2) / image_width
    center_y_normalized = (gt_data_item[1] + gt_data_item[3]/2) / image_height
    width_normalized = gt_data_item[2] / image_width
    height_normalized = gt_data_item[3] / image_height

    norm_str = f"{class_num} {center_x_normalized} {center_y_normalized} {width_normalized} {height_normalized}\n"

    return norm_str

def get_all_dir(root_dir):
    video_pathes = []
    for dir in os.listdir(root_dir):
        dir = os.path.join(root_dir, dir)
        if os.path.isdir(dir):
            video_pathes.extend(get_all_dir(dir))
        if dir[-5:] == '.json':
            video_pathes.append(dir)
    return video_pathes

root_dir = './datasets/videos/train_videos'
video_pathes = get_all_dir(root_dir)

for video_path in video_pathes:
    savetxt(video_path)

print("Done!!")