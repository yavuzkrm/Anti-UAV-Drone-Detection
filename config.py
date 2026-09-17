"""
Configuration File for Dataset Processing

Controls which dataset split (train/val/test) and which video folder to process.
Change these values to process different parts of the dataset.

Examples:
    # Process training videos
    train_or_val_folder = "train"
    video_folder = "train_videos"
    
    # Process validation videos
    train_or_val_folder = "val"
    video_folder = "val_videos"
    
    # Process test videos
    train_or_val_folder = "test"
    video_folder = "test_videos"
"""

# Dataset split to process: "train", "val", or "test"
# This controls where extracted frames and labels are saved
# Output: ./datasets/images/{train_or_val_folder}/
#         ./datasets/labels/{train_or_val_folder}/
train_or_val_folder = "test"

# Video folder source: "train_videos", "val_videos", or "test_videos"
# This tells the scripts which videos to read
# Input: ./datasets/videos/{video_folder}/
video_folder = "test_videos"
