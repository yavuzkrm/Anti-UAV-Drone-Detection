"""
Video to Frame Extraction Module

Converts MP4 video files to individual PNG frames while preserving directory structure.
Supports skipping already-processed frames to avoid re-processing.

Example:
    python framecut.py
    
    Reads from: ./datasets/videos/train_videos/
    Writes to:  ./datasets/images/train/
"""

import os
import cv2
from datetime import datetime
from config import train_or_val_folder, video_folder

# Base output directory for extracted frames
BASE_VIDEO_DIR = f"./datasets/images/{train_or_val_folder}"

def videotoimage(video_path):
    """
    Extract all frames from a video file and save as PNG images.
    
    Args:
        video_path (str): Full path to the MP4 video file
        
    Behavior:
        - Creates directory structure: BASE_VIDEO_DIR/session_id/camera_type/
        - Skips frames that already exist (for resuming interrupted processing)
        - Saves frames as: cameraI0000.png, cameraI0001.png, etc.
        - Prints processing summary with frame count and execution time
    """
    try:
        start_time = datetime.now()

        frame_count = 0
        success = True

        cap = cv2.VideoCapture(video_path)
        video_class = os.path.basename(video_path).split('.')[0]
        # video_dir = os.path.join(os.path.dirname(video_path), video_class) # Aynı dosyaların içine koymak için
        
        # Belirtilen klasöre koymak için
        video_path = video_path.replace('\\', '/') # get_all_dir() içinde windows yol eklerken '\' ekler. Bölme yaparken hata almamak için.
        parts = video_path.split('/')
        session_id = parts[4] 


        video_dir = os.path.join(BASE_VIDEO_DIR, session_id, video_class)

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        while success:
            success, frame = cap.read()
            if not success:
                break
            params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
            img = os.path.join(video_dir, f"{video_class}I{str(frame_count).zfill(4)}.png")
            if os.path.exists(img):
                frame_count += 1
                continue
            cv2.imwrite(img, frame, params)
            frame_count += 1

        finish_time = datetime.now()
        print(f"Video: {video_dir}, frame count: {frame_count}, time: {finish_time - start_time}")
    except Exception as e: 
        print(f"Error in {video_path}: {e}")
    finally:
        cap.release()

def get_all_dir(root_dir):
    """
    Recursively find all MP4 video files in directory structure.
    
    Args:
        root_dir (str): Starting directory path
        
    Returns:
        list: Full paths to all .mp4 files found
    """
    video_pathes = []
    for dir in os.listdir(root_dir):
        dir = os.path.join(root_dir, dir)
        if os.path.isdir(dir):
            # Recursively search subdirectories
            video_pathes.extend(get_all_dir(dir))
        # Check if this is an MP4 file
        if dir[-4:] == '.mp4':
            video_pathes.append(dir)
    return video_pathes

# Main execution
if __name__ == '__main__':
    # Find all videos in the configured folder
    root_dir = f"./datasets/videos/{video_folder}"
    video_pathes = get_all_dir(root_dir)
    
    print(f"Found {len(video_pathes)} video(s)")
    
    # Process each video
    for video_path in video_pathes:
        videotoimage(video_path)
    
    print("Done!!")