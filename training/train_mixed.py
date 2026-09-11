"""
Day 22: 用混合資料集（整圖 + 切片）訓練，目標是同時保有全局視野與局部細節。

除了 data 換成 find-airport-mixed 之外，其餘超參數刻意跟 Day 20 的
train_tiled.py 一模一樣，這樣三個模型比較起來才只差在`資料`這一個變因。
"""
from pathlib import Path

from ultralytics import YOLO


def train_mixed_obb():
    project_root = Path(__file__).resolve().parent.parent
    yaml_path = project_root / "find-airport-mixed" / "data.yaml"

    model = YOLO("yolov8n-obb.pt")

    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        project=str(project_root / "runs" / "train"),
        name="airport_obb_mixed",

        fliplr=0.5,
        flipud=0.5,
        degrees=90.0,   # 跟 train_tiled 對齊，把資料集之外的變因鎖住
    )


if __name__ == "__main__":
    train_mixed_obb()
