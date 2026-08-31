import numpy as np
import rasterio
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2

def run_sahi_inference():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    model_path = project_root / "runs" / "train" / "airport_obb_v1-4" / "weights" / "best.pt"
    test_source = project_root / "data" / "raw" / "som_san_air.tif"
    output_dir = project_root / "runs" / "predict" / "sahi_result"

    with rasterio.open(str(test_source)) as src:
        img_array = src.read([1, 2, 3])
        img_array = np.transpose(img_array, (1, 2, 0))

        if img_array.dtype == np.uint16 or img_array.max() > 255:
            img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        #加入 CLAHE 影像增強
        # 將影像轉為 LAB 色彩空間
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 設定 CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        enhanced_lab = cv2.merge((cl, a_channel, b_channel))
        img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # 透過 SAHI 的 AutoDetectionModel 封裝 YOLO 權重
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=0.25,
        device="cuda:0" # 呼叫你的 RTX 5070 Ti
    )

    # 直接將 Numpy 矩陣餵給 SAHI，設定 640x640 切片與 20% 重疊
    result = get_sliced_prediction(
        img_array,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result.export_visuals(export_dir=str(output_dir), file_name="sahi_airport")
    print(f"Finish")

if __name__ == "__main__":
    run_sahi_inference()