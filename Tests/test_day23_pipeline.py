"""
Day 23: 端到端管線驗收，順便回答一個 Day 18 就埋下的問題——
        我們到底有沒有把`對的三個波段`餵給模型。

三組前處理，其餘參數完全相同（混合模型、切 320、conf 0.25、CLAHE、幾何合併）：

  A. bands=(0,1,2) + max 拉伸        -> Day 18 到 Day 22 一路沿用的做法
  B. bands=(3,2,1) + max 拉伸        -> 只修波段
  C. bands=(3,2,1) + 百分位數拉伸     -> 波段與拉伸都修（管線的新預設）
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.pipeline import AirportDetector  # noqa: E402
from inference.visualize import (  # noqa: E402
    detections_to_boxes, draw_boxes, hstack_panels,
)

TIF = ROOT / "data" / "raw" / "som_san_air.tif"
OUT = ROOT / "runs" / "predict" / "day23_pipeline"

CASES = [
    ("A. bands 0,1,2 + max", (0, 1, 2), "max"),
    ("B. bands 3,2,1 + max", (3, 2, 1), "max"),
    ("C. bands 3,2,1 + percentile", (3, 2, 1), "percentile"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = []

    for title, bands, stretch in CASES:
        detector = AirportDetector(bands=bands, stretch=stretch)
        result = detector.detect(TIF)

        print(f"\n[{title}]  原始 {result.raw_count} 框 -> 合併後 {len(result.detections)} 框")
        print(f"   EPSG:{result.georef.epsg}  影像 {result.image.shape}")
        for d in result.detections:
            cx, cy = d.center
            print(f"   碎片x{d.n_fragments} conf={d.score:.3f} 角度={d.angle:6.1f} "
                  f"長={d.length:6.1f} 寬={d.width:6.1f} 長寬比={d.aspect:.2f} "
                  f"中心=({cx:.0f},{cy:.0f})")

        bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
        panel = draw_boxes(bgr, detections_to_boxes(result.detections),
                           f"{title}  ({len(result.detections)} boxes)")
        cv2.imwrite(str(OUT / f"{title.split('.')[0]}.jpg"), panel)
        panels.append(panel)

    path = OUT / "day23_side_by_side.jpg"
    cv2.imwrite(str(path), hstack_panels(panels))
    print(f"\n輸出: {path}")


if __name__ == "__main__":
    main()
