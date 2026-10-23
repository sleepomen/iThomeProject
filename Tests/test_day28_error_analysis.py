"""
Day 28: 誤判分析。Day 27 的成績是召回 5/8、框精確率 12.5%、負樣本全軍覆沒，
這裡拆成三個問題各做一個實驗：

  [1] 調高信心門檻能不能把誤判濾掉？      -> 命中框與誤判框的分數分布
  [2] 是不是色彩分布不同（domain shift）？ -> 把 EOX 影像直方圖對齊成訓練圖的樣子再跑一次
  [3] 模型學到的「機場」到底是什麼？       -> 把訓練標註對位回 GeoTIFF、換成經緯度、對 OSM 圍籬
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Tests"))

from inference.evaluate import load_airports, score_feature, to_metric, utm_zone_for  # noqa: E402
from inference.geo_utils import pixel_to_lonlat  # noqa: E402
from inference.geotiff_io import GeoRef, read_geotiff, to_uint8_rgb, write_geotiff  # noqa: E402
from inference.pipeline import AirportDetector  # noqa: E402
from inference.tiles import bbox_around  # noqa: E402
from inference.webmap import render_static  # noqa: E402
from test_day27_multi_scene import contact_sheet, print_table, run  # noqa: E402

TIF = ROOT / "data" / "raw" / "som_san_air.tif"
SCENES = ROOT / "data" / "s2cloudless"
MATCHED = ROOT / "data" / "s2cloudless_matched"
DATASET = ROOT / "Find-airport-1"
OUT = ROOT / "runs" / "predict" / "day28_errors"


# ---------------- [1] 分數門檻 ----------------
def threshold_sweep():
    d = json.loads((ROOT / "runs/predict/day27_multi/day27_rgb.json").read_text(encoding="utf-8"))
    s = np.array([f["properties"]["score"] for f in d["features"]])
    hit = np.array([f["properties"]["hit"] for f in d["features"]])
    print(f"  命中框 {hit.sum()} 個，分數 {s[hit].min():.3f} ~ {s[hit].max():.3f}，"
          f"中位數 {np.median(s[hit]):.3f}")
    print(f"  誤判框 {(~hit).sum()} 個，分數 {s[~hit].min():.3f} ~ {s[~hit].max():.3f}，"
          f"中位數 {np.median(s[~hit]):.3f}")
    print(f"  {'門檻':>6}{'剩幾框':>8}{'命中':>6}{'精確率':>9}")
    for t in (0.25, 0.3, 0.4, 0.5, 0.6):
        k = s >= t
        prec = (k & hit).sum() / k.sum() if k.sum() else float("nan")
        print(f"  {t:>6.2f}{k.sum():>8}{(k & hit).sum():>6}{prec:>9.1%}")


# ---------------- [2] 色彩對齊 ----------------
def match_histogram(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    逐通道直方圖對齊：讓 src 每個通道的累積分布 (CDF) 跟 ref 一樣。

    作法是兩張圖各自算 CDF，對 src 的每個灰階值，找 ref 中累積比例最接近的灰階值，
    做成 256 格的查表。這不會改變影像的`結構`，只改變`亮度與色彩的分布`。
    """
    out = np.empty_like(src)
    for c in range(3):
        s_hist = np.bincount(src[..., c].ravel(), minlength=256).cumsum()
        r_hist = np.bincount(ref[..., c].ravel(), minlength=256).cumsum()
        s_cdf, r_cdf = s_hist / s_hist[-1], r_hist / r_hist[-1]
        lut = np.searchsorted(r_cdf, s_cdf).clip(0, 255).astype(np.uint8)
        out[..., c] = lut[src[..., c]]
    return out


def build_matched_scenes(reference: np.ndarray):
    MATCHED.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SCENES / "manifest.json").read_text(encoding="utf-8"))
    for sc in manifest:
        rgb, georef = read_geotiff(SCENES / sc["file"])
        write_geotiff(MATCHED / sc["file"], match_histogram(rgb, reference), georef)
    shutil.copy(SCENES / "manifest.json", MATCHED / "manifest.json")


# ---------------- [3] 標註稽核 ----------------
def register(train_gray, ref_kp, ref_desc, sift, matcher):
    """
    用 SIFT 特徵把訓練圖對位回原始 GeoTIFF。

    Roboflow 的增強只有裁切、旋轉、翻轉，沒有縮放，所以用`相似變換`
    （平移 + 旋轉 + 等比縮放，4 個自由度）就夠了，比 homography 穩。
    縮放比算出來不是 1 的，就是對位失敗。
    """
    kp, desc = sift.detectAndCompute(train_gray, None)
    if desc is None:
        return None, 0, 0.0
    good = [a for a, b in matcher.knnMatch(desc, ref_desc, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < 6:
        return None, 0, 0.0
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good])
    m, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
    if m is None:
        return None, 0, 0.0
    return m, int(inliers.sum()), float(np.hypot(m[0, 0], m[1, 0]))


def label_audit(airports):
    stack, georef = read_geotiff(TIF)
    ref = to_uint8_rgb(stack, bands=(0, 1, 2), stretch="max")
    sift = cv2.SIFT_create(8000)
    ref_kp, ref_desc = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), None)
    matcher = cv2.BFMatcher()

    # 每張唯一來源圖有 3 個增強版本，挑對位最穩（inlier 最多）的那個
    best = {}
    for p in sorted(glob.glob(str(DATASET / "*" / "images" / "*.jpg"))):
        split = Path(p).parts[-3]
        source = os.path.basename(p).split("-0_jpg")[0]
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        m, n_in, scale = register(gray, ref_kp, ref_desc, sift, matcher)
        if m is None or not 0.97 < scale < 1.03 or n_in < 7:
            continue
        if source not in best or n_in > best[source]["inliers"]:
            best[source] = {"path": p, "split": split, "m": m, "inliers": n_in, "scale": scale}

    tpe = next(a for a in airports if a["iata"] == "TPE")
    features, rows = [], []
    for source, b in sorted(best.items(), key=lambda kv: kv[1]["split"]):
        label_path = b["path"].replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        label_path = os.path.splitext(label_path)[0] + ".txt"
        h, w = cv2.imread(b["path"]).shape[:2]
        for line in open(label_path, encoding="utf-8"):
            vals = np.array(line.split()[1:], dtype=np.float64).reshape(-1, 2) * [w, h]
            # 訓練圖像素 -> GeoTIFF 像素 -> 經緯度
            tif_px = vals @ b["m"][:, :2].T + b["m"][:, 2]
            lon, lat = pixel_to_lonlat(georef, tif_px[:, 0], tif_px[:, 1])
            ring = [[float(a), float(c)] for a, c in zip(lon, lat)]
            # Roboflow 的多邊形偶爾自相交，buffer(0) 修完可能裂成好幾塊，留最大的那塊
            poly = Polygon(ring).buffer(0)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            ring = [list(xy) for xy in poly.exterior.coords]
            f = {"type": "Feature",
                 "geometry": {"type": "Polygon", "coordinates": [ring]},
                 "properties": {"score": 1.0, "source": source[:8], "split": b["split"]}}
            f["properties"].update(score_feature(f, [tpe]))
            features.append(f)
            rows.append((b["split"], source[:8], b["inliers"], b["scale"],
                         f["properties"]["box_area_km2"], f["properties"]["inside_frac"],
                         f["properties"]["coverage"]))

    print(f"  對位成功 {len(best)}/10 張唯一來源圖")
    print(f"  {'split':<6}{'source':<10}{'inl':>4}{'scale':>7}{'km²':>7}"
          f"{'在機場內':>9}{'蓋到機場':>9}")
    for r in rows:
        print(f"  {r[0]:<6}{r[1]:<10}{r[2]:>4}{r[3]:>7.3f}{r[4]:>7.2f}{r[5]:>9.1%}{r[6]:>9.1%}")

    union = None
    for f in features:
        g = Polygon(f["geometry"]["coordinates"][0])
        union = g if union is None else union.union(g)
    fc_union = {"type": "Feature", "geometry": {"type": "Polygon",
                                                "coordinates": [list(map(list, union.convex_hull.exterior.coords))]},
                "properties": {}}
    hull = score_feature(fc_union, [tpe])
    tpe_km2 = to_metric(tpe["geom"], utm_zone_for(*tpe["centroid"])).area / 1e6
    print(f"\n  {len(features)} 個標註多邊形，平均 {np.mean([r[4] for r in rows]):.2f} km²；"
          f"OSM 桃園機場圍籬 {tpe_km2:.2f} km²")
    print(f"  全部標註的凸包：{hull['inside_frac']:.1%} 在機場內，蓋到機場 {hull['coverage']:.1%}")

    # Day 24 那個偏移的框，跟每個標註重疊多少（需要先跑過 test_day25_map.py）
    day25 = ROOT / "runs/predict/day25_map/detections_scored.geojson"
    if day25.exists():
        zone = utm_zone_for(*tpe["centroid"])
        box = to_metric(Polygon(json.loads(day25.read_text(encoding="utf-8"))
                                ["features"][0]["geometry"]["coordinates"][0]), zone)
        print("\n  Day 24 的框包住了每個標註的幾成：")
        fracs = []
        for f in features:
            g = to_metric(Polygon(f["geometry"]["coordinates"][0]), zone)
            fracs.append((g.intersection(box).area / g.area, f["properties"]["source"]))
        for frac, source in sorted(fracs, reverse=True):
            print(f"    {source}  {frac:6.1%}")

    view = bbox_around(121.226, 25.083, 9000, 6500)
    img = render_static(view, 14, {"type": "FeatureCollection", "features": features}, [tpe],
                        source="esri", title="training labels projected to WGS84 vs OSM fence",
                        show_score=False)
    cv2.imwrite(str(OUT / "day28_label_audit.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return features


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    airports = load_airports()

    print("[1] 調高門檻能不能救？")
    threshold_sweep()

    print("\n[2] 色彩對齊：把 EOX 場景的直方圖對齊成訓練圖（B1/B2/B3 + max 拉伸）")
    stack, _ = read_geotiff(TIF)
    reference = to_uint8_rgb(stack, bands=(0, 1, 2), stretch="max")
    build_matched_scenes(reference)
    detector = AirportDetector(bands=(0, 1, 2), stretch="max")
    rows, tiles, _ = run(detector, "matched", airports, write_db=True, scenes=MATCHED)
    print_table(rows)
    cv2.imwrite(str(OUT / "day28_matched_sheet.jpg"), contact_sheet(tiles),
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 同一個場景三種長相並排：原始 EOX / 直方圖對齊 / 訓練圖
    eox, _ = read_geotiff(SCENES / "tpe.tif")
    matched, _ = read_geotiff(MATCHED / "tpe.tif")
    crop = lambda im: cv2.resize(im[:780, :1150], (575, 390))  # noqa: E731
    panel = np.hstack([crop(eox), np.full((390, 8, 3), 255, np.uint8), crop(matched),
                       np.full((390, 8, 3), 255, np.uint8), crop(reference)])
    cv2.imwrite(str(OUT / "day28_color_domain.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    print("\n[3] 標註稽核：訓練標註換成經緯度，對 OSM 桃園機場圍籬")
    label_audit(airports)


if __name__ == "__main__":
    main()
