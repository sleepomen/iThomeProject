"""
Day 20: 用 Shapely 布林運算，把 640x640 的整機場圖切成小切片，
        並把被切斷的 Polygon 標註重新映射成局部座標。

解決 Day 19 提到的「感受野領域偏移」：推論時 SAHI 怎麼切，訓練時就怎麼切。
"""
import random
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

# ---------------- 參數 ----------------
TILE_SIZE = 320          # 切片邊長（原圖 640 的一半 = 2 倍放大視野）
OVERLAP = 0.2            # 重疊率，跟 SAHI 推論時保持一致
OUT_SIZE = 640           # 切完後放大回 640 餵給 YOLO
MIN_AREA_RATIO = 0.05    # 碎片小於原多邊形 5% 就丟掉（避免只剩一個角的噪音標註）
MIN_ABS_AREA = 64        # 碎片絕對面積下限 (px^2)
NEG_KEEP_RATIO = 0.15    # 純背景切片保留率
SIMPLIFY_TOL = 0.8       # 多邊形簡化容差 (px)
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "Find-airport-1"
DST_ROOT = ROOT / "find-airport-tiled"


def tile_offsets(total: int, tile: int, stride: int) -> list[int]:
    """產生切片起點；最後一格貼齊右/下邊界，避免邊緣資料被丟掉。"""
    if total <= tile:
        return [0]
    offsets = list(range(0, total - tile + 1, stride))
    if offsets[-1] != total - tile:
        offsets.append(total - tile)
    return offsets


def load_polygons(label_path: Path, w: int, h: int) -> list[tuple[int, Polygon]]:
    """讀 YOLO segmentation 標註 → 還原成像素座標的 Shapely Polygon。"""
    polygons = []
    if not label_path.exists():
        return polygons

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + 至少 3 個點
            continue
        cls_id = int(float(parts[0]))
        coords = np.asarray(parts[1:], dtype=np.float64)
        if coords.size % 2:
            coords = coords[:-1]
        pts = coords.reshape(-1, 2) * np.array([w, h])

        poly = Polygon(pts)
        if not poly.is_valid:
            # 標註人員手繪的輪廓常常自相交，buffer(0) 是幾何界的萬用修復術
            poly = make_valid(poly).buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        polygons.append((cls_id, poly))
    return polygons


def geom_to_parts(geom) -> list[Polygon]:
    """intersection 可能吐出 Polygon / MultiPolygon / GeometryCollection，全部攤平。"""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty]
    return []


def to_yolo_line(cls_id: int, poly: Polygon, ox: int, oy: int, tile: int) -> str | None:
    """把全局座標的碎片平移成局部座標，再正規化成 YOLO 格式。"""
    poly = poly.simplify(SIMPLIFY_TOL, preserve_topology=True)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None

    xs, ys = poly.exterior.coords.xy
    pts = np.stack([np.asarray(xs), np.asarray(ys)], axis=1)[:-1]  # 去掉重複的收尾點
    if len(pts) < 3:
        return None

    pts[:, 0] = np.clip(pts[:, 0] - ox, 0, tile) / tile
    pts[:, 1] = np.clip(pts[:, 1] - oy, 0, tile) / tile
    flat = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
    return f"{cls_id} {flat}"


def process_split(split: str, rng: random.Random) -> dict:
    src_img_dir = SRC_ROOT / split / "images"
    src_lbl_dir = SRC_ROOT / split / "labels"
    dst_img_dir = DST_ROOT / split / "images"
    dst_lbl_dir = DST_ROOT / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(TILE_SIZE * (1 - OVERLAP)))
    stats = {"src": 0, "pos": 0, "neg_kept": 0, "neg_drop": 0, "frag": 0, "frag_drop": 0}

    for img_path in sorted(src_img_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        stats["src"] += 1
        h, w = img.shape[:2]
        polygons = load_polygons(src_lbl_dir / f"{img_path.stem}.txt", w, h)

        for oy in tile_offsets(h, TILE_SIZE, stride):
            for ox in tile_offsets(w, TILE_SIZE, stride):
                window = box(ox, oy, ox + TILE_SIZE, oy + TILE_SIZE)

                lines = []
                for cls_id, poly in polygons:
                    if not poly.intersects(window):
                        continue
                    # 核心：切片視窗與跑道輪廓做布林交集
                    for part in geom_to_parts(poly.intersection(window)):
                        if part.area < MIN_ABS_AREA or part.area < poly.area * MIN_AREA_RATIO:
                            stats["frag_drop"] += 1
                            continue
                        line = to_yolo_line(cls_id, part, ox, oy, TILE_SIZE)
                        if line:
                            lines.append(line)
                            stats["frag"] += 1

                if lines:
                    stats["pos"] += 1
                elif rng.random() < NEG_KEEP_RATIO:
                    stats["neg_kept"] += 1
                else:
                    stats["neg_drop"] += 1
                    continue

                tile_img = img[oy:oy + TILE_SIZE, ox:ox + TILE_SIZE]
                if OUT_SIZE != TILE_SIZE:
                    tile_img = cv2.resize(tile_img, (OUT_SIZE, OUT_SIZE),
                                          interpolation=cv2.INTER_CUBIC)

                name = f"{img_path.stem}_x{ox}_y{oy}"
                cv2.imwrite(str(dst_img_dir / f"{name}.jpg"), tile_img)
                (dst_lbl_dir / f"{name}.txt").write_text("\n".join(lines))

    return stats


def write_yaml():
    yaml_path = DST_ROOT / "data.yaml"
    yaml_path.write_text(
        "names:\n- airport\n"
        "nc: 1\n"
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
    )
    return yaml_path


def main():
    rng = random.Random(SEED)
    total = {}
    for split in ("train", "valid", "test"):
        if not (SRC_ROOT / split / "images").exists():
            continue
        s = process_split(split, rng)
        total[split] = s
        kept = s["pos"] + s["neg_kept"]
        print(f"[{split:5s}] 原圖 {s['src']:3d} -> 切片 {kept:4d} "
              f"(正樣本 {s['pos']}, 背景保留 {s['neg_kept']}, 背景丟棄 {s['neg_drop']}) "
              f"| 標註碎片 保留 {s['frag']} / 濾除 {s['frag_drop']}")

    yaml_path = write_yaml()
    grand = sum(v["pos"] + v["neg_kept"] for v in total.values())
    print(f"\n總計 {grand} 張切片 -> {DST_ROOT}")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
