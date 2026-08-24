import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # 1. 取得專案根目錄 (iTHome/)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    # 2. 定義 data.yaml 路徑
    yaml_path = project_root / "find-airport-1" / "data.yaml"

    # 3. 載入預訓練模型基底
    model = YOLO("yolov8n.pt")

    # 4. 啟動優化訓練
    results = model.train(
        data=str(yaml_path),
        epochs=100,                 # 上限提升至 100 輪
        patience=15,                # 早停機制：15 輪 Loss 沒降就自動停
        imgsz=640,                  # 圖片尺寸 640x640
        batch=16,                   # 5070 Ti 滿血運作
        workers=4,
        name="sentinel_airport_v2",  # 輸出成果目錄名稱
        device=0,
        project=str(project_root / "runs" / "detect"),

        lr0=0.001,                  # 降低初始學習率，讓收斂更平穩
        weight_decay=0.005,         # 加強 L2 正則化

        degrees=180.0,              # 180 度旋轉 (搭配鏡像可達成 360 度無死角)
        fliplr=0.5,                 # 左右鏡像翻轉
        flipud=0.5,                 # 上下鏡像翻轉
        mosaic=1.0,                 # Mosaic 4 圖拼接增強
        scale=0.5                   # 縮放增強 (0.5 ~ 1.5 倍)
    )

    print("Finish")

if __name__ == "__main__":
    main()