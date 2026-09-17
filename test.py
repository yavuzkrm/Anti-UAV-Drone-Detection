"""
Model Evaluation and Testing Script

Evaluates the trained YOLOv8 model on test dataset and generates predictions.
Outputs metrics and visualization of predictions.

Usage:
    python test.py
    
Output:
    - Test metrics: mAP50, mAP50-95, precision, recall
    - Prediction visualizations: runs/test_results/*.jpg
"""

from ultralytics import YOLO
import glob

if __name__ == '__main__':
    # Load the best trained model
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Evaluate on test set
    # This computes metrics like mAP (mean Average Precision) against ground truth labels
    print("Validating on test set...")
    results = model.val(data="data.yaml", split="test")
    
    # Print evaluation metrics
    print(f"\n=== Test Results ===")
    print(f"mAP50 (IoU=0.50):     {results.box.map50:.4f}")  # Precision at IoU threshold 0.50
    print(f"mAP50-95 (IoU=0.50-0.95): {results.box.map:.4f}")  # Average precision across IoU thresholds
    
    # Generate predictions on sample images (first 100 test images)
    # This shows visual predictions with bounding boxes
    print("\nGenerating predictions on sample images...")
    test_images = glob.glob("./datasets/images/test/**/*.png", recursive=True)[:100]
    
    if test_images:
        results = model.predict(
            source=test_images,           # Images to predict on
            conf=0.5,                     # Confidence threshold (0-1, only show predictions > 0.5)
            save=True,                    # Save prediction visualizations
            save_dir="runs/test_results"  # Output directory for visualizations
        )
        print(f"Saved {len(test_images)} prediction visualizations to runs/test_results/")
    else:
        print("No test images found. Make sure to process test videos first.")
    
    print("Testing complete!")
