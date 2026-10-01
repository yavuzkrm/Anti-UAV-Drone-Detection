"""
YOLO training for the drone detector.

Three modes, all defined in train.yaml:
    joint     - one model on infrared + visible frames
    infrared  - infrared-only model
    visible   - visible-only model
Shared settings live under 'common', per-mode differences under 'modes'.

Example:
    python make_splits.py            # build splits/*.txt first
    python train.py joint
    python train.py infrared 50      # override epochs
    python train.py visible 30 8     # override epochs and batch
"""

import shutil
import sys
import yaml
from ultralytics import YOLO
from pathlib import Path

CONFIG_FILE = "train.yaml"


class Trainer:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        with open(CONFIG_FILE) as file:
            self.cfg = yaml.safe_load(file)

    def train(self, train_mode, epochs=None, batch=None):
        """Train one mode and copy its best weights to models/best_{mode}.pt"""
        # Mode values override common ones, command line values override both
        params = {**self.cfg["common"], **self.cfg["modes"][train_mode]}
        if epochs is not None:
            params["epochs"] = epochs
        if batch is not None:
            params["batch"] = batch

        model = YOLO(params.pop("model"))

        try:
            results = model.train(**params, name=train_mode)

            src = Path(model.trainer.save_dir) / "weights" / "best.pt"
            dst = self.models_dir / f"best_{train_mode}.pt"

            if src.exists():
                shutil.copy(src, dst)
                print(f"✅ Model Saved: {dst}")
                print(f"   Val mAP50: {results.box.map50:.4f}")
                return str(dst)
            else:
                raise FileNotFoundError(f"Training failed: {src} not found")

        except Exception as e:
            print(f"❌ Training Error: {e}")
            return None


if __name__ == "__main__":
    train_mode = sys.argv[1] if len(sys.argv) > 1 else "joint"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else None

    trainer = Trainer()

    modes = trainer.cfg["modes"]
    if train_mode not in modes:
        print(f"❌ Unknown mode: {train_mode} - valid: {list(modes)}")
        sys.exit(1)

    # The data yaml's train list must exist (written by make_splits.py)
    with open(modes[train_mode]["data"]) as file:
        train_list = yaml.safe_load(file)["train"]
    if not Path(train_list).exists():
        print(f"❌ {train_list} not found - run: python make_splits.py")
        sys.exit(1)

    model_path = trainer.train(train_mode=train_mode, epochs=epochs, batch=batch)

    sys.exit(0 if model_path else 1)
