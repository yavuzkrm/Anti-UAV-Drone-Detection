# Anti-UAV Drone Detection

A YOLO-based drone detector for the Anti-UAV dataset, trained on infrared and visible video. Designed with defense industry practice in mind: evaluation on unseen recordings, per-sensor reporting, and a joint vs per-sensor model comparison.

## Dataset

**Anti-UAV Dataset**
- Infrared (640×512) and visible (1920×1080) video, recorded simultaneously
- Ground truth bounding boxes per frame (JSON)
- Test recordings are fully separate from train/val recordings

```
datasets/
├── videos/{train,val,test}_videos/<recording>/{infrared,visible}.{mp4,json}
├── images/{train,val,test}/{infrared,visible}/
└── labels/{train,val,test}/{infrared,visible}/
```

## Problem & Approach

The first version trained on infrared at 640, then fine-tuned the same model on visible frames. Validation looked good, but the test score collapsed. Two causes:

1. **Data leakage in validation** - every recording in the original val folder also appears in train (same day, sky and drone). Val mAP measured memorization, not generalization; the test set uses different recordings.
2. **Catastrophic forgetting** - fine-tuning only on visible overwrote what the model had learned on infrared.

Fixes:
- **Recording-based split** (`scripts/make_splits.py`): whole recordings are held out for validation, so val mAP reflects unseen flights.
- **Three models compared on the same splits**: one joint model (infrared + visible together, no forgetting) and one model per sensor.
- **Per-sensor test scores**, so a weak sensor is not hidden in a combined number.

## Pipeline

```bash
# 1. Prepare data (no argument = train, val and test)
python scripts/extract_frames.py    # MP4 -> PNG frames
python scripts/convert_labels.py    # JSON -> YOLO labels
python scripts/make_splits.py       # recording-based image lists in splits/

# 2. Train (settings in configs/train.yaml)
python train.py joint
python train.py infrared
python train.py visible

# 3. Evaluate all trained models on the test set and compare
python evaluate.py
```

`extract_frames.py` and `convert_labels.py` also accept a single split, e.g. `python scripts/extract_frames.py test`. All paths are resolved from the repo root, so the scripts can be run from any directory.

## Configuration

**configs/train.yaml** - training settings. `common` is shared by every mode, `modes` holds only what differs:

| Mode | Data | imgsz |
|------|------|-------|
| joint | `configs/data.yaml` | 960 |
| infrared | `configs/data_ir.yaml` | 640 (native resolution) |
| visible | `configs/data_visible.yaml` | 960 |

Epochs and batch can be overridden from the command line: `python train.py visible 50 8`.

**imgsz**: YOLO scales each image so its long side equals imgsz, keeping the aspect ratio. At 640 a typical visible drone shrinks to ~21 px; at 960 it stays ~31 px. Evaluation always uses the imgsz saved in the model, so train and test cannot get out of sync.

## Results

Pending - fill in after training:

| Sensor | Joint model mAP50 | Per-sensor model mAP50 |
|--------|-------------------|------------------------|
| Infrared | - | - |
| Visible | - | - |

*Earlier proof of concept (YOLOv8n, 5 videos, 10 epochs): mAP50 87.5%, mAP50-95 49.2%. Not comparable with the current setup.*

## Installation

```bash
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch first, following [pytorch.org](https://pytorch.org/get-started/locally/).

## File Structure
```
Anti-UAV-Drone-Detection/
├── configs/
│   ├── train.yaml            # Training settings
│   ├── data.yaml             # Dataset: infrared + visible
│   ├── data_ir.yaml          # Dataset: infrared only
│   └── data_visible.yaml     # Dataset: visible only
├── scripts/
│   ├── extract_frames.py     # Video -> frames
│   ├── convert_labels.py     # JSON -> YOLO labels
│   └── make_splits.py        # Recording-based train/val/test lists
├── train.py                  # Training (joint / infrared / visible)
├── evaluate.py               # Test set evaluation and model comparison
├── requirements.txt
├── LICENSE
└── README.md
```

Generated, not tracked by git: `datasets/` (frames and labels), `splits/` (image lists), `models/` (best weights), `runs/` (Ultralytics outputs).
```

## Defense Applications

- Autonomous drone detection systems
- Surveillance infrastructure enhancement
- Real-time threat assessment
- Multi-spectrum analysis (IR + visible)

## Future Work

- [ ] False alarm rate and recall by drone size
- [ ] Drone tracking module (temporal consistency)
- [ ] Decision-level fusion of infrared and visible
- [ ] ONNX/TensorRT optimization and FPS measurement
- [ ] Multi-class detection (drone types)

## References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- YOLO: https://arxiv.org/abs/2301.04335

## Author

**Yavuz Kerem**  
Computer Engineering, Ankara University

## License

MIT - see [LICENSE](LICENSE).
