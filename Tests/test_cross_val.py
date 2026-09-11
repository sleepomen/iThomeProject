"""
Day 22: 三個模型 x 兩個驗證集的交叉評估。

Day 20 做過 2x2，證明了整圖模型在切片上是 0.000。
今天多了一個混合模型，看它能不能同時站穩兩邊。

  模型：baseline（整圖訓練）/ tiled（切片訓練）/ mixed（混合訓練）
  驗證集：原始整圖 valid / 切片 valid
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    ("baseline", ROOT / "runs/train/airport_obb_v1-4/weights/best.pt"),
    ("tiled", ROOT / "runs/train/airport_obb_tiled/weights/best.pt"),
    ("mixed", ROOT / "runs/train/airport_obb_mixed/weights/best.pt"),
]

DATASETS = [
    ("整圖 valid", ROOT / "Find-airport-1" / "data.yaml"),
    ("切片 valid", ROOT / "find-airport-tiled" / "data.yaml"),
]


def main():
    rows = []
    for m_name, weight in MODELS:
        if not weight.exists():
            print(f"[skip] 找不到權重 {weight}")
            continue
        for d_name, yaml_path in DATASETS:
            metrics = YOLO(str(weight)).val(
                data=str(yaml_path),
                split="val",
                imgsz=640,
                device="0",
                project=str(ROOT / "runs" / "val"),
                name=f"{m_name}__{yaml_path.parent.name}",
                exist_ok=True,
                plots=False,
                verbose=False,
            )
            b = metrics.box
            rows.append((m_name, d_name, b.mp, b.mr, b.map50, b.map))

    print("\n| 模型 | 驗證集 | P | R | mAP50 | mAP50-95 |")
    print("|---|---|---|---|---|---|")
    for m, d, p, r, m50, m95 in rows:
        print(f"| {m} | {d} | {p:.3f} | {r:.3f} | {m50:.3f} | {m95:.3f} |")


if __name__ == "__main__":
    main()
