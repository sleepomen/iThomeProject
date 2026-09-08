"""
Day 21: 用幾何合併把 Day 20 的六個碎框縫成一條跑道，並輸出左右對照圖。

  左 = SAHI 預設 GREEDYNMM 之後的原始碎框
  右 = inference/merge_fragments.py 的分群 + 主軸擬合結果
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Tests.test_nmm_postprocess import load_image, SLICE, WEIGHT  # noqa: E402
from inference.merge_fragments import merge_predictions, obb_polygon  # noqa: E402

OUT = ROOT / "runs" / "predict" / "day21_merge"

RED = (60, 60, 235)      # 原始碎框
GREEN = (90, 220, 90)    # 合併後的跑道
ORANGE = (40, 150, 245)  # 主軸擬合時被剔除的離群碎片


def draw(img, boxes, title):
    """畫框 + 上方標題列（BGR）。"""
    canvas = img.copy()
    for pts, color, label in boxes:
        pts = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, color, 3, cv2.LINE_AA)
        x, y = pts[0][0]
        cv2.putText(canvas, label, (int(x), max(int(y) - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    bar = np.zeros((44, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def main():
    img = load_image()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    OUT.mkdir(parents=True, exist_ok=True)

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(WEIGHT),
        confidence_threshold=0.25,
        device="cuda:0",
    )
    result = get_sliced_prediction(
        img, model,
        slice_height=SLICE, slice_width=SLICE,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        verbose=0,
    )
    preds = result.object_prediction_list
    print(f"[before] SAHI 預設後處理輸出 {len(preds)} 個碎框")

    before = [(np.asarray(obb_polygon(p).exterior.coords[:-1]), RED,
               f"{p.score.value:.2f}") for p in preds]

    merged = merge_predictions(preds)
    after = []
    print(f"[after ] 幾何合併後 {len(merged)} 個框")
    for m in merged:
        multi = len(m["members"]) > 1
        color = GREEN if multi else ORANGE
        tag = f"merged x{len(m['members'])} {m['score']:.2f}" if multi else f"{m['score']:.2f}"
        after.append((m["points"], color, tag))
        print(f"   members={str(m['members']):<12} dropped={str(m['dropped']):<10} "
              f"conf={m['score']:.3f} 角度={m['angle']:6.1f}  "
              f"長={m['length']:6.1f} 寬={m['width']:6.1f} "
              f"長寬比={m['length'] / max(m['width'], 1e-6):.2f}")

    left = draw(bgr, before, f"A. SAHI GREEDYNMM  ({len(before)} boxes)")
    right = draw(bgr, after, f"B. geometry merge  ({len(after)} boxes)")
    sep = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)

    cv2.imwrite(str(OUT / "day21_merge_before.jpg"), left)
    cv2.imwrite(str(OUT / "day21_merge_after.jpg"), right)
    path = OUT / "day21_side_by_side.jpg"
    cv2.imwrite(str(path), np.hstack([left, sep, right]))
    print(f"\n輸出: {path}")


if __name__ == "__main__":
    main()
