from ultralytics import YOLO
import glob


if __name__ == '__main__':
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Val'i test'e karşı test et
    results = model.val(data="data.yaml", split="test")
    print(f"Test mAP50: {results.box.map50}")
    print(f"Test mAP50-95: {results.box.map}")
    
    # 100 örnek tahmin et (görsel için)
    test_images = glob.glob("./datasets/images/test/**/*.png", recursive=True)[:100]
    
    results = model.predict(source=test_images, conf=0.5, save=True, save_dir="runs/test_results")