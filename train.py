import shutil
import os
from ultralytics import YOLO

def trainInfrared():
    model = YOLO("yolov8m.pt")

    results = model.train(
        data="data_ir.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        patience=30,
        name="ir_v2_200video",  # ✅ Otomatik folder naming
    )
    
    # Copy best.pt to models/
    src = "runs/detect/ir_v2_200video/weights/best.pt"
    dst = "models/best_ir_v2.pt"
    os.makedirs("models", exist_ok=True)
    shutil.copy(src, dst)
    print(f"✅ Saved: {dst}")
    
    return dst

def trainVisible(ir_model_path):
    model = YOLO(ir_model_path)  # Transfer learning

    results = model.train(
        data="data_visible.yaml",
        epochs=30,  # Fine-tune, daha az epoch
        imgsz=1280,
        batch=16,
        workers=4,
        device=0,
        patience=30,
        name="visible_v2_finetuned",  # ✅ Otomatik folder naming
    )
    
    # Copy best.pt
    src = "runs/detect/visible_v2_finetuned/weights/best.pt"
    dst = "models/best_visible_v2.pt"
    os.makedirs("models", exist_ok=True)
    shutil.copy(src, dst)
    print(f"✅ Saved: {dst}")
    
    return dst

if __name__ == "__main__":
    print("🟢 IR Training...")
    ir_model = trainInfrared()
    
    print("\n🟢 Visible Fine-tuning...")
    visible_model = trainVisible(ir_model)
    
    print(f"\n✅ Done!")
    print(f"  IR: {ir_model}")
    print(f"  Visible: {visible_model}")
