from pathlib import Path
from ultralytics import YOLO

def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    yaml_path = project_root / "find-airport-1" / "data.yaml"

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        name="sentinel_airport_v3",
        device=0,
        project=str(project_root / "runs" / "detect"),

        lr0=0.01,            # 恢復預設學習率
        weight_decay=0.0005, # 恢復預設正則化

        degrees=0.0,         # 關閉 180 度大旋轉
        scale=0.0,           # 關閉大範圍縮放
        fliplr=0.5,          # 保留左右翻轉
        flipud=0.5,          # 保留上下翻轉
        mosaic=1.0           # 保留拼接
    )

    print("Finish")

if __name__ == "__main__":
    main()