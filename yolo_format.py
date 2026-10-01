"""
JSON to YOLO label conversion.

Writes one .txt per frame with the same name as its PNG, in YOLO format:
    class_id center_x center_y width height     (all normalized to 0-1)
Frames without a drone get an empty file (background image).

    datasets/videos/{split}_videos/<recording>/infrared.json
    -> datasets/labels/{split}/infrared/<recording>_infraredI0000.txt

Example:
    python yoloformat.py             # train, val and test
    python yoloformat.py test        # only one split
"""

import sys
import json
from pathlib import Path

SPLITS = ["train", "val", "test"]

# Frame size of each camera, used to normalize pixel boxes
IMAGE_DIMENSIONS = {
    "infrared": [640, 512],
    "visible": [1920, 1080],
}


def savetxt(json_path, output_root):
    """Convert one annotation file into per-frame label files"""
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)

        gt_data = data['gt_rect']
        # exist[i] == 0 means no drone in frame i (gt_rect may still hold a stale box)
        exist_data = data.get('exist', [1] * len(gt_data))

        camera = json_path.stem               # "infrared" or "visible"
        recording = json_path.parent.name     # e.g. 20190925_101846_1_1

        label_dir = output_root / camera
        label_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, gt_data_item in enumerate(gt_data):
            exists = frame_idx < len(exist_data) and exist_data[frame_idx]
            if not exists or len(gt_data_item) < 4 or gt_data_item[2] <= 0 or gt_data_item[3] <= 0:
                norm_str = ""  # background frame
            else:
                norm_str = yolonormalization(gt_data_item, camera)

            txt_path = label_dir / f"{recording}_{camera}I{str(frame_idx).zfill(4)}.txt"
            txt_path.write_text(norm_str)
    except Exception as e:
        print(f"Error: {e}, File: {json_path}")


def yolonormalization(gt_data_item, camera):
    """[x, y, w, h] in pixels -> 'class cx cy w h' normalized to 0-1"""
    class_num = 0  # single class: drone

    image_width, image_height = IMAGE_DIMENSIONS[camera]

    # Clip box to image borders (some annotations extend outside the frame)
    x1 = max(0, gt_data_item[0])
    y1 = max(0, gt_data_item[1])
    x2 = min(image_width, gt_data_item[0] + gt_data_item[2])
    y2 = min(image_height, gt_data_item[1] + gt_data_item[3])
    if x2 <= x1 or y2 <= y1:
        return ""

    center_x_normalized = (x1 + x2) / 2 / image_width
    center_y_normalized = (y1 + y2) / 2 / image_height
    width_normalized = (x2 - x1) / image_width
    height_normalized = (y2 - y1) / image_height

    return f"{class_num} {center_x_normalized} {center_y_normalized} {width_normalized} {height_normalized}\n"


if __name__ == '__main__':
    splits = sys.argv[1:] or SPLITS

    for split in splits:
        json_root = Path(f"./datasets/videos/{split}_videos")
        output_root = Path(f"./datasets/labels/{split}")

        json_pathes = sorted(json_root.rglob("*.json"))
        print(f"[{split}] Found {len(json_pathes)} annotation file(s)")

        for json_path in json_pathes:
            savetxt(json_path, output_root)

    print("Done!!")
