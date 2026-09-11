"""
Day 22: 混合模型 vs 切片模型，同一張 GeoTIFF、同一套 Day 21 幾何合併。

要回答的問題只有一個：混合資料集能不能把 Day 21 那個`過胖的合併框`瘦回去，
同時不要退回 Day 18「只看得見半截跑道」的狀態。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Tests.test_nmm_postprocess import load_image  # noqa: E402
from Tests.test_fragment_merge import draw, GREEN, ORANGE  # noqa: E402
from inference.merge_fragments import merge_predictions  # noqa: E402

OUT = ROOT / "runs" / "predict" / "day22_compare"

CASES = [
    ("tiled  slice320", ROOT / "runs/train/airport_obb_tiled/weights/best.pt", 320),
    ("mixed  slice320", ROOT / "runs/train/airport_obb_mixed/weights/best.pt", 320),
    ("mixed  slice640", ROOT / "runs/train/airport_obb_mixed/weights/best.pt", 640),
]


def run_case(img, weight: Path, slice_size: int):
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
    return preds, merge_predictions(preds)


def main():
    img = load_image()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    OUT.mkdir(parents=True, exist_ok=True)

    panels = []
    for name, weight, slice_size in CASES:
        if not weight.exists():
            print(f"[skip] 找不到權重 {weight}")
            continue
        preds, merged = run_case(img, weight, slice_size)
        print(f"\n[{name}] 原始 {len(preds)} 框 -> 合併後 {len(merged)} 框")

        boxes = []
        for m in merged:
            multi = len(m["members"]) > 1
            aspect = m["length"] / max(m["width"], 1e-6)
            tag = (f"merged x{len(m['members'])} {m['score']:.2f}" if multi
                   else f"{m['score']:.2f}")
            boxes.append((m["points"], GREEN if multi else ORANGE, tag))
            print(f"   members={str(m['members']):<12} conf={m['score']:.3f} "
                  f"角度={m['angle']:6.1f} 長={m['length']:6.1f} "
                  f"寬={m['width']:6.1f} 長寬比={aspect:.2f}")

        panels.append(draw(bgr, boxes, f"{name}  ({len(merged)} boxes)"))
        cv2.imwrite(str(OUT / f"{name.replace(' ', '')}.jpg"), panels[-1])

    if panels:
        sep = np.full((panels[0].shape[0], 8, 3), 255, dtype=np.uint8)
        strip = panels[0]
        for p in panels[1:]:
            strip = np.hstack([strip, sep, p])
        path = OUT / "day22_side_by_side.jpg"
        cv2.imwrite(str(path), strip)
        print(f"\n輸出: {path}")


if __name__ == "__main__":
    main()
