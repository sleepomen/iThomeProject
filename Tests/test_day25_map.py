"""
Day 25: 把 Day 24 的 GeoJSON 丟上真的底圖。

1. 重跑 Day 24 的偵測，拿到桃園那張圖的 GeoJSON
2. 用 OSM 的機場圍籬多邊形替框打分數（取代 Day 24 只用一個 ARP 點）
3. 輸出 folium 互動地圖 + 兩張靜態對照圖（Esri 高解析 / Sentinel-2 cloudless）
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.evaluate import load_airports, score_feature  # noqa: E402
from inference.geo_utils import detections_to_geojson, pixel_to_lonlat  # noqa: E402
from inference.pipeline import AirportDetector  # noqa: E402
from inference.tiles import bbox_around, ground_resolution  # noqa: E402
from inference.visualize import hstack_panels  # noqa: E402
from inference.webmap import build_map, render_static, save_map  # noqa: E402

TIF = ROOT / "data" / "raw" / "som_san_air.tif"
OUT = ROOT / "runs" / "predict" / "day25_map"
TPE_ARP = (121.2328, 25.0767)          # 25°04'36"N 121°13'58"E，同 Day 24


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    result = AirportDetector().detect(TIF)
    fc = detections_to_geojson(result.detections, result.georef,
                               source=TIF.name, model="airport_obb_mixed")

    airports = load_airports()
    tpe = next(a for a in airports if a["iata"] == "TPE")
    print(f"OSM 桃園機場圍籬 {tpe['osm_id']}，"
          f"中心 {tpe['centroid'][1]:.4f}N {tpe['centroid'][0]:.4f}E")

    for f in fc["features"]:
        f["properties"].update(score_feature(f, airports))
        p = f["properties"]
        print(f"\n框 score={p['score']}  配對機場 {p['iata']}")
        print(f"  框面積              {p['box_area_km2']:.2f} km²")
        print(f"  框落在機場內的比例  {p['inside_frac']:.1%}")
        print(f"  機場被框到的比例    {p['coverage']:.1%}")
        print(f"  命中 (>= 50%)?      {p['hit']}")

    (OUT / "detections_scored.geojson").write_text(
        json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")

    # 互動地圖
    m = build_map(fc, airports=[tpe], markers=[(*TPE_ARP, "TPE ARP")],
                  center=TPE_ARP, zoom=13)
    html = save_map(m, OUT / "day25_map.html")
    print(f"\n互動地圖 {html}  ({html.stat().st_size:,} bytes)")

    # 靜態圖：同一個範圍、兩種底圖
    view = bbox_around(121.226, 25.085, 7000, 5600)
    z = 14
    print(f"靜態圖第 {z} 層，實地解析度 {ground_resolution(z, 25.08):.2f} m/px")
    panels = [
        render_static(view, z, fc, [tpe], [(*TPE_ARP, "ARP")], source=src, title=title)
        for src, title in (("esri", "Esri imagery (~0.5 m)"),
                           ("s2cloudless", "Sentinel-2 cloudless (10 m)"))
    ]
    h = min(p.shape[0] for p in panels)
    panels = [p[:h] for p in panels]
    cv2.imwrite(str(OUT / "day25_side_by_side.jpg"), hstack_panels(panels),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    # 整張 GeoTIFF 的範圍畫在地圖上，看看原始影像涵蓋哪裡
    h_px, w_px = result.image.shape[:2]
    lon, lat = pixel_to_lonlat(result.georef, [0, w_px, w_px, 0], [0, 0, h_px, h_px])
    print("GeoTIFF 四角：", [(round(float(a), 4), round(float(b), 4)) for a, b in zip(lon, lat)])


if __name__ == "__main__":
    main()
