import cv2
import numpy as np
import rasterio
from pathlib import Path
from ultralytics import YOLO

def run_obb_inference():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    model_path = project_root / "runs" / "train" / "airport_obb_v1-4" / "weights" / "best.pt"
    test_source = project_root / "data" / "raw" / "som_san_air.tif"

    model = YOLO(model_path)

    try:
        with rasterio.open(str(test_source)) as src:
            img_array = src.read([1, 2, 3])
            img_array = np.transpose(img_array, (1, 2, 0))

            if img_array.dtype == np.uint16 or img_array.max() > 255:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)

            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
    except Exception as e:
        print(f"error: {e}")
        return

    results = model.predict(
        source=img_bgr,
        conf=0.3,
        save=True,
        project=str(project_root / "runs" / "predict"),
        name="airport_obb_result"
    )
    
    print("Finish")

if __name__ == "__main__":
    run_obb_inference()