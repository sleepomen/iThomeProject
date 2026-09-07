"""
Day 20: 同一張 GeoTIFF、同一套前處理，比較兩個權重在 SAHI 切片推論下的表現。

  A. baseline  : 整圖訓練 (airport_obb_v1-4)，SAHI 切 640
  B. tiled     : 切片訓練 (airport_obb_tiled)，SAHI 切 320（跟訓練切片尺寸對齊）
"""
import cv2
import numpy as np
import rasterio
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ROOT = Path(__file__).resolve().parent.parent
TIF = ROOT / "data" / "raw" / "som_san_air.tif"
OUT = ROOT / "runs" / "predict" / "day20_compare"

CASES = [
    ("baseline_slice640", ROOT / "runs/train/airport_obb_v1-4/weights/best.pt", 640),
    ("tiled_slice320", ROOT / "runs/train/airport_obb_tiled/weights/best.pt", 320),
]


def load_image() -> np.ndarray:
    """Day 18 的前處理：16-bit 降維 + LAB 空間 CLAHE 局部打光。"""
    with rasterio.open(str(TIF)) as src:
        img = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
    if img.dtype != np.uint8 or img.max() > 255:
        img = (img / img.max() * 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def main():
    img = load_image()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"影像尺寸: {img.shape}")

    for name, weight, slice_size in CASES:
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(weight),
            confidence_threshold=0.25,
            device="cuda:0",
        )
        result = get_sliced_prediction(
            img, model,
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            verbose=0,
        )
        preds = result.object_prediction_list
        scores = [round(p.score.value, 3) for p in preds]
        print(f"[{name}] slice={slice_size} 偵測到 {len(preds)} 個目標 conf={scores}")
        result.export_visuals(export_dir=str(OUT), file_name=name, hide_labels=False)


if __name__ == "__main__":
    main()
