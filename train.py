"""
YOLOv8 Model Training Script

Trains a YOLOv8 nano model on the Anti-UAV drone detection dataset.

Usage:
    python train.py
    
Output:
    - Trained weights: runs/detect/train/weights/best.pt
    - Training metrics: runs/detect/train/results.csv
    - Training plots: runs/detect/train/*.png
"""

from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize YOLOv8 nano model with pretrained ImageNet weights
    # The 'n' in yolov8n stands for 'nano' - fastest, smallest model
    # Other options: yolov8s (small), yolov8m (medium), yolov8l (large), yolov8x (extra large)
    model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data="data.yaml",           # Dataset configuration file
        epochs=10,                  # Number of training epochs (full passes through data)
        imgsz=640,                  # Input image size (will be resized to this)
        batch=8,                    # Batch size (frames per GPU batch)
        workers=2,                  # Number of data loading workers
        device=0                    # GPU device (0 = first GPU, or 'cpu' for CPU)
        # Optional improvements:
        # patience=3,               # Early stopping patience (stop if no improvement)
        # save=True,               # Save training artifacts
        # amp=True,                # Automatic Mixed Precision (faster training)
        # hsv_h=0.015,             # HSV hue augmentation
        # hsv_s=0.7,               # HSV saturation augmentation
        # hsv_v=0.4,               # HSV value augmentation
    )
    
    print("Training complete!")