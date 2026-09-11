"""
Day 21: 診斷 Day 20 留下的「一條跑道被拆成六個框」。

先做兩件事：
  1. 把 SAHI 的 postprocess_type / match_metric / threshold 掃過一輪，看內建
     合併機制到底救不救得回來。
  2. 把碎片兩兩之間的 IOU / IOS 算出來，證明它們根本沒重疊，
     所以任何以「重疊率」為判準的後處理都不可能把它們縫在一起。
"""
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.geotiff_io import SENTINEL2_FIRST_THREE, read_geotiff, to_uint8_rgb  # noqa: E402
from inference.pipeline import enhance  # noqa: E402
TIF = ROOT / "data" / "raw" / "som_san_air.tif"
WEIGHT = ROOT / "runs/train/airport_obb_tiled/weights/best.pt"
SLICE = 320


def load_image() -> np.ndarray:
    """
    Day 18 的前處理，Day 23 起改走 inference/geotiff_io（rasterio 被
    Smart App Control 擋掉了），行為保持一致以維持與前幾天的可比性。
    """
    stack, _ = read_geotiff(TIF)
    return enhance(to_uint8_rgb(stack, bands=SENTINEL2_FIRST_THREE, stretch="max"))


def obb_polygon(pred) -> Polygon:
    """SAHI 把 OBB 的四個角點塞在 segmentation 裡，撈出來還原成 Polygon。"""
    seg = pred.mask.segmentation[0]
    return Polygon(np.asarray(seg, dtype=np.float64).reshape(-1, 2)).buffer(0)


def sweep(img, model):
    print("\n=== SAHI 內建後處理掃描 ===")
    print(f"{'postprocess':<12}{'metric':<8}{'thr':<7}{'框數':<6}conf")
    for ptype, metric, thr in itertools.product(
        ("NMS", "NMM", "GREEDYNMM"), ("IOU", "IOS"), (0.5, 0.2, 0.05)
    ):
        result = get_sliced_prediction(
            img, model,
            slice_height=SLICE, slice_width=SLICE,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            postprocess_type=ptype,
            postprocess_match_metric=metric,
            postprocess_match_threshold=thr,
            verbose=0,
        )
        preds = result.object_prediction_list
        scores = [round(p.score.value, 2) for p in preds]
        print(f"{ptype:<12}{metric:<8}{thr:<7}{len(preds):<6}{scores}")


def overlap_matrix(img, model):
    """算碎片兩兩之間的 IOU 與 IOS，看重疊率到底有多低。"""
    result = get_sliced_prediction(
        img, model,
        slice_height=SLICE, slice_width=SLICE,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        verbose=0,
    )
    preds = result.object_prediction_list
    polys = [obb_polygon(p) for p in preds]
    n = len(polys)
    print(f"\n=== 碎片兩兩重疊率 (n={n}, SAHI 預設 GREEDYNMM/IOS/0.5 之後) ===")
    print("     " + "".join(f"{j:>14}" for j in range(n)))
    for i in range(n):
        row = [f"{i:<5}"]
        for j in range(n):
            if i == j:
                row.append(f"{'-':>14}")
                continue
            inter = polys[i].intersection(polys[j]).area
            union = polys[i].union(polys[j]).area
            small = min(polys[i].area, polys[j].area)
            row.append(f"{inter / union:>6.3f}/{inter / small:<7.3f}")
        print("".join(row))
    print("（格式 IOU/IOS）")

    # 碎片之間的「最短距離」才是真正該拿來當判準的量
    print("\n=== 碎片兩兩最短邊界距離 (px) ===")
    print("     " + "".join(f"{j:>8}" for j in range(n)))
    for i in range(n):
        cells = "".join(f"{'-':>8}" if i == j else f"{polys[i].distance(polys[j]):>8.1f}"
                        for j in range(n))
        print(f"{i:<5}{cells}")
    return preds, polys


def main():
    img = load_image()
    print(f"影像尺寸: {img.shape}")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(WEIGHT),
        confidence_threshold=0.25,
        device="cuda:0",
    )
    sweep(img, model)
    overlap_matrix(img, model)


if __name__ == "__main__":
    main()
