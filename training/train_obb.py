from pathlib import Path
from ultralytics import YOLO

def train_obb_model():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    yaml_path = project_root / "find-airport-1" / "data.yaml"

    model = YOLO("yolov8n-obb.pt") 

    results = model.train(
        data=str(yaml_path),
        epochs=50,             
        imgsz=640,
        batch=16,
        device="0",            
        project=str(project_root / "runs" / "train"),
        name="airport_obb_v1", 
        
        fliplr=0.5,python
        flipud=0.5
    )

if __name__ == "__main__":
    train_obb_model()