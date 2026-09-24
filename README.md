# Anti-UAV Drone Detection

A YOLO-based object detection system for drone detection using the Anti-UAV dataset. Designed for defense industry applications.

## Project Overview

Complete pipeline for drone detection in infrared and visible spectrum:
- **Video to Frame Conversion**: Extracts frames with tracking
- **Label Format Conversion**: JSON → YOLO format
- **Model Training**: YOLOv8 fine-tuning
- **Testing & Evaluation**: Comprehensive metrics

## Dataset

**Anti-UAV Dataset**
- Infrared (640×512) and Visible (1920×1080) video
- Ground truth bounding box annotations (JSON)
- Multi-modal drone detection

### Folder Structure
```
datasets/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Pipeline

### 1. Video to Frame Extraction
```bash
python framecut.py
```
Converts MP4 videos to PNG frames, preserves directory hierarchy.

### 2. JSON to YOLO Conversion
```bash
python yoloformat.py
```
Transforms Anti-UAV JSON annotations to normalized YOLO format.
- Handles infrared (640×512) and visible (1920×1080) dimensions
- Supports empty frames (no drone)

### 3. Model Training
```bash
python train.py
```
Fine-tunes YOLOv8n on drone detection.

**Configuration**:
- Model: YOLOv8n
- Epochs: 10
- Batch: 8
- Input: 640×640

### 4. Testing & Inference
```bash
python test.py
```
Evaluates model on test set, generates predictions.

## Results

### Test Performance
| Metric | Score |
|--------|-------|
| **mAP50** | 87.5% |
| **mAP50-95** | 49.2% |
| **Precision** | 84% |
| **Recall** | 80% |

**Data**: 10 videos (10,000 frames)  
**Status**: Proof-of-concept validation ✓

## Installation

```bash
# Requirements
pip install opencv-python ultralytics torch torchvision

# Setup
git clone https://github.com/yavuzkrm/Anti-UAV-Drone-Detection
cd Anti-UAV-Drone-Detection
```

## Configuration

**config.py** - Dataset split
```python
train_or_val_folder = "train"  # or "val", "test"
video_folder = "train_videos"
```

**data.yaml** - YOLO configuration
```yaml
path: ./datasets
train: images/train
val: images/val
test: images/test
nc: 1
names: ['drone']
```

## Usage Workflow

```bash
# 1. Prepare data
python framecut.py
python yoloformat.py

# 2. Train
python train.py

# 3. Test
python test.py
```

## Technical Details

### Key Features
- Multi-modal detection (IR + visible)
- Dimension-aware normalization (640×512 vs 1920×1080)
- Recursive dataset traversal
- Cross-platform compatibility (Windows/Linux)
- Empty frame handling

### Architecture
- Framework: YOLOv8 (Ultralytics)
- Backbone: CSP Darknet
- Input: 640×640 (auto-scaled)
- Output: Bounding box predictions

## File Structure
```
Anti-UAV-Drone-Detection/
├── framecut.py           # Video extraction
├── yoloformat.py         # Label conversion
├── train.py              # Training script
├── test.py               # Evaluation script
├── config.py             # Configuration
├── data.yaml             # Dataset config
└── README.md
```

## Defense Applications

- Autonomous drone detection systems
- Surveillance infrastructure enhancement
- Real-time threat assessment
- Multi-spectrum analysis (IR + visible)

## Future Work

- [ ] Scale to 100+ videos
- [ ] Larger models (YOLOv8m/l)
- [ ] Multi-class detection (drone types)
- [ ] Real-time streaming inference
- [ ] ONNX/TensorRT optimization
- [ ] Drone tracking module

## References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- YOLO: https://arxiv.org/abs/2301.04335

## Author

**Yavuz Kerem**  
Computer Engineering, Ankara University  
Target: Defense Industry

---

**Status**: MVP Complete | **Last Updated**: September 2026
