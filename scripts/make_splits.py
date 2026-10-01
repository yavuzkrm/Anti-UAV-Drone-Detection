"""
Recording-based train/val/test image lists.

The original val folder comes from the same recordings as train (same day, sky
and drone), so val mAP looked great while the test set, recorded separately,
failed. Here whole recordings are held out for validation instead, so val mAP
reflects performance on unseen flights. All lists use the same held-out
recordings, so the joint and per-sensor models are compared fairly.

Files are not moved - YOLO reads the lists and finds each label by replacing
/images/ with /labels/ in the path.

Example:
    python scripts/make_splits.py

    Writes: splits/{train,val,test}.txt                 (infrared + visible)
            splits/{train,val,test}_infrared.txt
            splits/{train,val,test}_visible.txt
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root, so the script runs from any directory
DATASET_DIR = ROOT / "datasets" / "images"
SPLIT_DIR = ROOT / "splits"
MODALITIES = ["infrared", "visible"]

# Recordings (first 15 chars of the file name) held out for validation.
# One daytime + one evening recording, ~15% of the frames.
VAL_RECORDINGS = {"20190925_141417", "20190925_194211"}

# Keep every Nth frame. Consecutive video frames are almost identical,
# so subsampling speeds up training a lot without losing much information.
TRAIN_FRAME_STEP = 3
VAL_FRAME_STEP = 5


def frame_index(path):
    """Extract frame number from names like 20190925_101846_1_1_infraredI0042.png"""
    return int(Path(path).stem.rsplit("I", 1)[1])


def write_list(name, paths):
    (SPLIT_DIR / name).write_text("\n".join(paths) + "\n")
    print(f"{name}: {len(paths)} images")


if __name__ == "__main__":
    SPLIT_DIR.mkdir(exist_ok=True)

    train, val = [], []
    for modality in MODALITIES:
        modality_train, modality_val = [], []
        # Pool the original train and val folders, then re-split by recording
        for split in ["train", "val"]:
            for path in sorted((DATASET_DIR / split / modality).glob("*.png")):
                recording = path.name[:15]
                if recording in VAL_RECORDINGS:
                    if frame_index(path) % VAL_FRAME_STEP == 0:
                        modality_val.append(str(path))
                elif frame_index(path) % TRAIN_FRAME_STEP == 0:
                    modality_train.append(str(path))

        # Per-sensor lists
        write_list(f"train_{modality}.txt", modality_train)
        write_list(f"val_{modality}.txt", modality_val)

        train += modality_train
        val += modality_val

    # Joint lists = union of the per-sensor lists
    write_list("train.txt", train)
    write_list("val.txt", val)

    # Test set is untouched (all frames), plus per-sensor lists for separate scores
    test_ir = sorted(str(p) for p in (DATASET_DIR / "test" / "infrared").glob("*.png"))
    test_vis = sorted(str(p) for p in (DATASET_DIR / "test" / "visible").glob("*.png"))
    write_list("test.txt", test_ir + test_vis)
    write_list("test_infrared.txt", test_ir)
    write_list("test_visible.txt", test_vis)

    print("Done!!")
