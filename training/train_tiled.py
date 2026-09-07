from pathlib import Path
from ultralytics import YOLO


def train_tiled_obb():
    project_root = Path(__file__).resolve().parent.parent
    yaml_path = project_root / "find-airport-tiled" / "data.yaml"

    model = YOLO("yolov8n-obb.pt")

    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        project=str(project_root / "runs" / "train"),
        name="airport_obb_tiled",

        fliplr=0.5,
        flipud=0.5,
        degrees=90.0,   # 切片後跑道方向更零碎，直接讓模型看遍所有旋轉角
    )


if __name__ == "__main__":
    train_tiled_obb()
