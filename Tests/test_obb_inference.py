from pathlib import Path
from ultralytics import YOLO

def run_obb_inference():
    # 1. 自動定位路徑
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    model_path = project_root / "runs" / "train" / "airport_obb_v1-4" / "weights" / "best.pt"
    test_source = project_root / "data" / "raw" / "som_san_air.tif"
    
    model = YOLO(model_path)

    results = model.predict(
        source=str(test_source),
        conf=0.3,       # 信心門檻
        save=True,      # 自動畫上斜框並存檔
        project=str(project_root / "runs" / "predict"),
        name="airport_obb_result"
    )
    
if __name__ == "__main__":
    run_obb_inference()