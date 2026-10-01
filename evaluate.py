"""
Model Evaluation & Comparison

Evaluates the joint model and the per-sensor models on the untouched test set,
then prints a comparison table (joint vs separate model, per sensor).

imgsz is NOT passed here: every checkpoint remembers the imgsz it was trained
with, and model.val() / model.predict() use it automatically.

Example:
    python scripts/make_splits.py             # test lists must exist
    python evaluate.py                        # all models found in models/
    python evaluate.py infrared               # only models/best_infrared.pt
    python evaluate.py --skip-images          # metrics only, no visualizations
"""

from ultralytics import YOLO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
SPLIT_DIR = ROOT / "splits"

# Data yaml for each test set (their 'test:' line points to splits/test*.txt)
TEST_SETS = {
    "all": CONFIG_DIR / "data.yaml",
    "infrared": CONFIG_DIR / "data_ir.yaml",
    "visible": CONFIG_DIR / "data_visible.yaml",
}

# Image lists used for the visual check in predict_batch
TEST_LISTS = {
    "all": SPLIT_DIR / "test.txt",
    "infrared": SPLIT_DIR / "test_infrared.txt",
    "visible": SPLIT_DIR / "test_visible.txt",
}

# Which test sets each model is evaluated on.
# A per-sensor model never saw the other sensor, so it is only tested on its own.
MODELS = {
    "joint": ["all", "infrared", "visible"],
    "infrared": ["infrared"],
    "visible": ["visible"],
}

MODELS_DIR = ROOT / "models"


class Test:
    def __init__(self, mode: str):
        self.mode = mode
        self.model = YOLO(MODELS_DIR / f"best_{mode}.pt")
        # Saved in the checkpoint at training time, used automatically by val/predict
        self.imgsz = self.model.overrides.get("imgsz")

    def validate(self, set_name):
        """Validation on one test set"""
        print(f"📊 [{self.mode}] Validating on test set: {set_name} (imgsz={self.imgsz})")
        results = self.model.val(
            data=str(TEST_SETS[set_name]),
            split="test",
            batch=16,
            plots=False,
            name=f"test_{self.mode}_{set_name}",
            exist_ok=True,
        )

        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr
        }

    def predict_batch(self, set_name, conf=0.25, max_images=50):
        """Predict a few test images and save visualizations for a visual check"""
        with open(TEST_LISTS[set_name]) as f:
            paths = [line.strip() for line in f if line.strip()]

        if not paths:
            print("❌ No test images found")
            return 0

        # Take images evenly spread over the list (covers both sensors for "all")
        step = max(1, len(paths) // max_images)
        test_images = paths[::step][:max_images]

        print(f"🔍 [{self.mode}] Predicting on {len(test_images)} images...")
        detected = 0
        # stream=True yields one result at a time instead of holding all in memory
        for result in self.model.predict(
            source=test_images,
            conf=conf,
            save=True,
            name=f"test_predictions_{self.mode}",
            exist_ok=True,
            stream=True,
            verbose=False
        ):
            detected += int(len(result.boxes) > 0)

        print(f"   Images with a detection: {detected}/{len(test_images)} "
              f"(saved to runs/detect/test_predictions_{self.mode})")
        return detected

    def run(self, skip_images=False):
        """Evaluate this model on all of its test sets"""
        metrics = {set_name: self.validate(set_name) for set_name in MODELS[self.mode]}

        if not skip_images:
            # "all" for the joint model, the model's own sensor otherwise
            self.predict_batch(MODELS[self.mode][0])

        return metrics


def print_results(all_results):
    """Per-model detail table + joint vs separate comparison"""
    print("\n" + "=" * 60)
    print("✅ TEST RESULTS")
    print("=" * 60)
    print(f"   {'model':<10}{'set':<10}{'mAP50':>9}{'mAP50-95':>10}{'P':>8}{'R':>8}")
    for mode, metrics in all_results.items():
        for set_name, m in metrics.items():
            print(f"   {mode:<10}{set_name:<10}{m['mAP50']:>9.4f}{m['mAP50-95']:>10.4f}"
                  f"{m['precision']:>8.4f}{m['recall']:>8.4f}")

    # Comparison only makes sense when the joint model and at least one sensor model exist
    if "joint" in all_results and len(all_results) > 1:
        print(f"\n   mAP50 - joint vs separate")
        print(f"   {'sensor':<10}{'joint':>9}{'separate':>10}")
        for sensor in ["infrared", "visible"]:
            if sensor in all_results:
                joint = all_results["joint"][sensor]["mAP50"]
                separate = all_results[sensor][sensor]["mAP50"]
                print(f"   {sensor:<10}{joint:>9.4f}{separate:>10.4f}")
    print("=" * 60)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip_images = "--skip-images" in sys.argv

    modes = args if args else list(MODELS)
    unknown = [m for m in modes if m not in MODELS]
    if unknown:
        print(f"❌ Unknown mode(s): {unknown} - valid: {list(MODELS)}")
        sys.exit(1)

    missing_lists = [p for p in TEST_LISTS.values() if not p.exists()]
    if missing_lists:
        print(f"❌ Test lists not found: {[str(p) for p in missing_lists]} - run: python scripts/make_splits.py")
        sys.exit(1)

    all_results = {}
    for mode in modes:
        if not (MODELS_DIR / f"best_{mode}.pt").exists():
            print(f"⚠️  models/best_{mode}.pt not found - skipping (train it with: python train.py {mode})")
            continue
        all_results[mode] = Test(mode).run(skip_images=skip_images)

    if not all_results:
        print("❌ No trained models found in models/")
        sys.exit(1)

    print_results(all_results)
