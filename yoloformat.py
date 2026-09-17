"""
JSON to YOLO Format Conversion Module

Converts Anti-UAV dataset JSON annotations to YOLO format.
Normalizes bounding box coordinates based on image dimensions.

YOLO Format:
    class_id center_x_normalized center_y_normalized width_normalized height_normalized
    Values are normalized to 0-1 range based on image dimensions

Example:
    python yoloformat.py
    
    Reads from: ./datasets/videos/train_videos/*.json
    Writes to:  ./datasets/labels/train/*.txt
"""

import os
import json
from config import train_or_val_folder, video_folder

# Base output directory for YOLO labels
BASE_LABEL_DIR = f"./datasets/labels/{train_or_val_folder}"

# Image dimensions for infrared and visible camera feeds
# Used to normalize bounding box coordinates (pixel → 0-1 range)
IMAGE_DIMENSIONS = {
    "infrared": [640, 512],      # Infrared: 640x512
    "visible": [1920, 1080]       # Visible: 1920x1080
}

def savetxt(video_path):
    """
    Convert JSON annotations to YOLO format and save as text files.
    
    Args:
        video_path (str): Full path to JSON annotation file
        
    Process:
        1. Load JSON with ground truth rectangles (gt_rect)
        2. For each frame:
           - If no drone: create empty .txt file
           - If drone present: normalize coordinates and save to .txt
        3. Create one .txt file per frame with matching PNG filename
        
    File Structure:
        Input:  videos/train/video.json
        Output: labels/train/infrared/infraredI0000.txt
                labels/train/infrared/infraredI0001.txt
                ...
    """
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
                norm_str = ""
            else:
                norm_str = yolonormalization(gt_data_item, label_class)

            label_dir = os.path.join(BASE_LABEL_DIR, session_id, label_class)
            
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            frame_name = f"{label_class}I{str(frame_idx).zfill(4)}"
            txt_path = os.path.join(label_dir, f"{frame_name}.txt")
        
            with open(txt_path, 'w') as txtfile:
                txtfile.write(norm_str)
    except Exception as e:
        print(f"Error: {e}, Video: {video_path}")


def yolonormalization(gt_data_item, label_class):
    """
    Normalize bounding box coordinates from pixel to YOLO format (0-1 range).
    
    Args:
        gt_data_item (list): [x, y, width, height] in pixel coordinates
        label_class (str): Camera type - 'infrared' or 'visible'
        
    Returns:
        str: YOLO format string
        
    Math:
        Input: [x_pixel, y_pixel, width_pixel, height_pixel]
        
        center_x_norm = (x_pixel + width_pixel/2) / image_width
        center_y_norm = (y_pixel + height_pixel/2) / image_height
        width_norm = width_pixel / image_width
        height_norm = height_pixel / image_height
        
        Output: "0 center_x center_y width height"
    """
    class_num = 0  # Single class: drone
    
    image_width, image_height = IMAGE_DIMENSIONS[label_class]
    
    # Calculate center point and normalize
    center_x_normalized = (gt_data_item[0] + gt_data_item[2]/2) / image_width
    center_y_normalized = (gt_data_item[1] + gt_data_item[3]/2) / image_height
    # Normalize dimensions
    width_normalized = gt_data_item[2] / image_width
    height_normalized = gt_data_item[3] / image_height
    
    norm_str = f"{class_num} {center_x_normalized} {center_y_normalized} {width_normalized} {height_normalized}\n"
    
    return norm_str

def get_all_dir(root_dir):
    """
    Recursively find all JSON annotation files.
    
    Args:
        root_dir (str): Starting directory path
        
    Returns:
        list: Full paths to all .json files found
    """
    video_pathes = []
    for dir in os.listdir(root_dir):
        dir = os.path.join(root_dir, dir)
        if os.path.isdir(dir):
            # Recursively search subdirectories
            video_pathes.extend(get_all_dir(dir))
        # Check if this is a JSON annotation file
        if dir[-5:] == '.json':
            video_pathes.append(dir)
    return video_pathes

# Main execution
if __name__ == '__main__':
    # Find all JSON annotation files in the configured folder
    root_dir = f"./datasets/videos/{video_folder}"
    video_pathes = get_all_dir(root_dir)
    
    print(f"Found {len(video_pathes)} annotation file(s)")
    
    # Convert each JSON file to YOLO format
    for video_path in video_pathes:
        savetxt(video_path)
    
    print("Done!!")