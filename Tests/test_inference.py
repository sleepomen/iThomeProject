from pathlib import Path
from ultralytics import YOLO

def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    # 自動鎖定最新的 v2 模型
    detect_dir = project_root / "runs" / "detect"
    best_pts = list(detect_dir.glob("**/weights/best.pt"))
    latest_best_pt = max(best_pts, key=lambda p: p.stat().st_mtime)

    model = YOLO(str(latest_best_pt))

    test_source = project_root / "find-airport-1" / "test" / "images"

    results = model.predict(
        source=str(test_source),
        conf=0.1,                                  
        iou=0.50,                                      # 保持 NMS 過濾重疊框
        save=True,
        project=str(project_root / "runs" / "predict"),
        name="airport_test_result_v3"                   # 存到新資料夾
    )

    output_dir = project_root / "runs" / "predict" / "airport_test_result_cleaned"
    print(f"Finish")

if __name__ == "__main__":
    main()