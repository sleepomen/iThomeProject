"""
Day 27: 從 EOX Sentinel-2 cloudless 抓多張測試場景，存成 GeoTIFF。

原本計畫用 Copernicus Data Space 下載 L2A 原檔，但要註冊帳號、處理 OAuth、
下載整個 100 km 見方的 tile 再自己裁。這裡改用 EOX 已經做好的去雲鑲嵌圖：
一樣是 Sentinel-2、一樣是 10m，只是已經合成成 8-bit 真彩色，不用帳號。

代價是`波段`：EOX 只給 RGB（B4/B3/B2），沒有我們訓練時用的 B1。
這件事 Day 27 會直接撞上。

場景分兩類：
  positive  有機場，中心對準 OSM 機場圍籬的形心
  negative  沒有機場，但有長條狀的人造物（港口、工業區、都市），專門用來騙模型
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.evaluate import airports_in_view, load_airports  # noqa: E402
from inference.geotiff_io import write_geotiff  # noqa: E402
from inference.tiles import (  # noqa: E402
    bbox_around, fetch_mosaic, ground_resolution, resample_to_ground,
)

OUT_DIR = ROOT / "data" / "s2cloudless"
ZOOM = 14                 # 台灣緯度約 8.7 m/px，最接近 10m 的一層
TARGET_RES = 10.0         # 重採樣成訓練資料的 10m，避免尺度差 15%
WIDTH_M, HEIGHT_M = 12_000, 8_000   # 跟 som_san_air.tif（13.5 x 9.2 km）差不多大

POSITIVE = ["TPE", "TSA", "RMQ", "CYI", "TNN", "KHH", "HUN", "TTT"]

NEGATIVE = {
    "neg_banqiao":   (121.455, 25.012, "板橋／新莊市區"),
    "neg_taipeiport": (121.375, 25.150, "台北港"),
    "neg_taichungport": (120.505, 24.275, "台中港"),
    "neg_mailiao":   (120.195, 23.785, "麥寮六輕工業區"),
    "neg_changhua":  (120.450, 23.950, "彰化農田"),
}


def scenes(airports):
    by_iata = {a["iata"]: a for a in airports}
    for iata in POSITIVE:
        ap = by_iata[iata]
        lon, lat = ap["centroid"]
        yield iata.lower(), lon, lat, ap.get("name_en") or ap["name"]
    for name, (lon, lat, desc) in NEGATIVE.items():
        yield name, lon, lat, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="只抓這幾個場景")
    args = ap.parse_args()

    airports = load_airports()
    manifest = []
    print(f"第 {ZOOM} 層實地解析度 {ground_resolution(ZOOM, 24.0):.2f} m/px，"
          f"重採樣到 {TARGET_RES:.0f} m/px\n")

    for name, lon, lat, desc in scenes(airports):
        if args.only and name not in args.only:
            continue
        bbox = bbox_around(lon, lat, WIDTH_M, HEIGHT_M)
        rgb, georef = fetch_mosaic(bbox, ZOOM, "s2cloudless")
        rgb, georef = resample_to_ground(rgb, georef, lat, TARGET_RES)
        path = write_geotiff(OUT_DIR / f"{name}.tif", rgb, georef)

        inside = [a["iata"] for a in airports_in_view(airports, bbox)]
        manifest.append({"scene": name, "desc": desc, "center": [lon, lat],
                         "bbox": bbox, "airports": inside, "file": path.name})
        print(f"{name:<18} {rgb.shape[1]}x{rgb.shape[0]}  機場 {inside or '無'}  {desc}")

    if not args.only:
        (OUT_DIR / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{len(manifest)} 個場景 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
