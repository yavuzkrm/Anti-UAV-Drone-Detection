import os
import cv2
from datetime import datetime

BASE_VIDEO_DIR = './datasets/images/train'

def videotoimage(video_path):
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
            cv2.imwrite(os.path.join(video_dir, f"{video_class}I{str(frame_count).zfill(4)}.png"), frame, params)
            frame_count += 1

        finish_time = datetime.now()
        print(f"Video: {video_dir}, frame count: {frame_count}, time: {finish_time - start_time}")
    except Exception as e: 
        print(f"Error in {video_path}: {e}")
    finally:
        cap.release()

def get_all_dir(root_dir):
    video_pathes = []
    for dir in os.listdir(root_dir):
        dir = os.path.join(root_dir, dir)
        if os.path.isdir(dir):
            video_pathes.extend(get_all_dir(dir))
        if dir[-4:] == '.mp4':
            video_pathes.append(dir)
    return video_pathes

root_dir = './datasets/videos/train_videos'
video_pathes = get_all_dir(root_dir)

for video_path in video_pathes:
    videotoimage(video_path)

print("Done!!")