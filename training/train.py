import os
from pathlib import Path
from ultralytics import YOLO

def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    yaml_path = project_root / "find-airport-1" / "data.yaml"

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=50,                  # 訓練 50 輪
        imgsz=640,                  # 圖片尺寸 640x640
        batch=16,                   # 5070 Ti 16GB 顯存直接開到 16
        workers=4,                  # 資料載入執行緒
        name="sentinel_airport_v1",  # 輸出專案名稱
        device=0,                   # 使用 GPU 0
        project=str(project_root / "runs" / "detect") # 統一將訓練結果輸出至根目錄的 runs/
    )

    print("Finish")

if __name__ == "__main__":
    main()