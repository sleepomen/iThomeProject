"""
Day 24: 把 Day 23 的管線接上經緯度，輸出 GeoJSON。

偵測框的四個角點從`像素`變成`經緯度`，再對照桃園國際機場的公開座標，
看模型框到的到底是不是那條跑道。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.geo_utils import (  # noqa: E402
    detections_to_geojson, gmaps_link, pixel_to_lonlat,
)
from inference.pipeline import AirportDetector  # noqa: E402

TIF = ROOT / "data" / "raw" / "som_san_air.tif"
OUT = ROOT / "runs" / "predict" / "day24_geo"

# 桃園國際機場 05L/23R 跑道中點（公開座標，當作地面真相）
TPE_RUNWAY = (121.2328, 25.0777)


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    detector = AirportDetector()
    result = detector.detect(TIF)
    print(f"原始 {result.raw_count} 框 -> 合併後 {len(result.detections)} 框\n")

    for i, d in enumerate(result.detections):
        cx, cy = d.center
        lon, lat = pixel_to_lonlat(result.georef, cx, cy)
        lon, lat = float(lon), float(lat)

        # 用對角兩個角點的實地距離回推跑道長度，順便驗一下像素尺度
        pts = np.asarray(d.points, dtype=np.float64)
        plon, plat = pixel_to_lonlat(result.georef, pts[:, 0], pts[:, 1])
        edges = [haversine_m(plon[j], plat[j], plon[(j + 1) % 4], plat[(j + 1) % 4])
                 for j in range(4)]
        long_edge, short_edge = max(edges), min(edges)

        dist = haversine_m(lon, lat, *TPE_RUNWAY)
        print(f"[{i}] conf={d.score:.3f} 碎片x{d.n_fragments}")
        print(f"    中心 {lat:.6f}N, {lon:.6f}E   距桃園機場跑道 {dist:,.0f} m")
        print(f"    實地尺寸 {long_edge:,.0f} m x {short_edge:,.0f} m"
              f"  （像素 {d.length:.0f} x {d.width:.0f}）")
        print(f"    {gmaps_link(lon, lat)}")

    fc = detections_to_geojson(result.detections, result.georef,
                               source=TIF.name, model="airport_obb_mixed")
    path = OUT / "detections.geojson"
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n寫出 {path}  ({len(fc['features'])} features, "
          f"{path.stat().st_size:,} bytes)")
    print(json.dumps(fc["features"][0]["geometry"], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
