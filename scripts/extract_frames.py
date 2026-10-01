"""
Video to frame extraction.

Saves every frame of each MP4 as a PNG. Frames that already exist are skipped,
so an interrupted run can simply be restarted.

    datasets/videos/{split}_videos/<recording>/infrared.mp4
    -> datasets/images/{split}/infrared/<recording>_infraredI0000.png

Example:
    python scripts/extract_frames.py         # train, val and test
    python scripts/extract_frames.py test    # only one split
"""

import sys
import cv2
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root, so the script runs from any directory
SPLITS = ["train", "val", "test"]


def videotoimage(video_path, output_root):
    """Extract all frames of one video into output_root/<camera>/"""
    start_time = datetime.now()
    cap = cv2.VideoCapture(str(video_path))
    try:
        camera = video_path.stem               # "infrared" or "visible"
        recording = video_path.parent.name     # e.g. 20190925_101846_1_1

        video_dir = output_root / camera
        video_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            img = video_dir / f"{recording}_{camera}I{str(frame_count).zfill(4)}.png"
            if not img.exists():
                # Low compression: larger files but much faster to write
                cv2.imwrite(str(img), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            frame_count += 1

        print(f"Video: {recording}/{camera}, frame count: {frame_count}, "
              f"time: {datetime.now() - start_time}")
    except Exception as e:
        print(f"Error in {video_path}: {e}")
    finally:
        cap.release()


if __name__ == '__main__':
    splits = sys.argv[1:] or SPLITS

    for split in splits:
        video_root = ROOT / "datasets" / "videos" / f"{split}_videos"
        output_root = ROOT / "datasets" / "images" / split

        video_pathes = sorted(video_root.rglob("*.mp4"))
        print(f"[{split}] Found {len(video_pathes)} video(s)")

        for video_path in video_pathes:
            videotoimage(video_path, output_root)

    print("Done!!")
