"""
Day 24: 帶經緯網格與地面控制點的成果圖。

一旦有了經緯度，就可以拿`公開座標`當地面控制點來驗收，
這是 Day 20 到 Day 23 一路在像素空間裡做不到的事。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.geo_utils import lonlat_to_pixel, pixel_to_lonlat  # noqa: E402
from inference.pipeline import AirportDetector  # noqa: E402
from inference.visualize import GREEN, hstack_panels  # noqa: E402

TIF = ROOT / "data" / "raw" / "som_san_air.tif"
OUT = ROOT / "runs" / "predict" / "day24_geo"

# 桃園國際機場（RCTP）的機場基準點 ARP，ICAO 公告座標 25°04'36"N 121°13'58"E
TPE_ARP = (121.2328, 25.0767)

CYAN = (235, 220, 60)
YELLOW = (60, 230, 235)
GREY = (150, 150, 150)


def draw_graticule(canvas, georef, step=0.02):
    """沿著整數經緯度畫格線並標刻度。"""
    h, w = canvas.shape[:2]
    lons, lats = pixel_to_lonlat(georef, [0, w, 0, w], [0, 0, h, h])
    lon0, lon1 = float(min(lons)), float(max(lons))
    lat0, lat1 = float(min(lats)), float(max(lats))

    for lon in np.arange(np.ceil(lon0 / step) * step, lon1, step):
        t = np.linspace(lat0, lat1, 50)
        c, r = lonlat_to_pixel(georef, np.full_like(t, lon), t)
        cv2.polylines(canvas, [np.stack([c, r], -1).astype(np.int32)],
                      False, GREY, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{lon:.2f}E", (int(c[0]) + 4, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    for lat in np.arange(np.ceil(lat0 / step) * step, lat1, step):
        t = np.linspace(lon0, lon1, 50)
        c, r = lonlat_to_pixel(georef, t, np.full_like(t, lat))
        cv2.polylines(canvas, [np.stack([c, r], -1).astype(np.int32)],
                      False, GREY, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{lat:.2f}N", (6, int(r[0]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def banner(canvas, text):
    bar = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = AirportDetector().detect(TIF)
    georef = result.georef
    bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)

    canvas = draw_graticule(bgr.copy(), georef)
    ac, ar = lonlat_to_pixel(georef, *TPE_ARP)
    ac, ar = int(ac), int(ar)
    print(f"TPE ARP {TPE_ARP[1]:.4f}N {TPE_ARP[0]:.4f}E -> 像素 ({ac}, {ar})")

    for d in result.detections:
        cv2.polylines(canvas, [np.asarray(d.points, np.int32).reshape(-1, 1, 2)],
                      True, GREEN, 3, cv2.LINE_AA)
        cx, cy = d.center
        lon, lat = pixel_to_lonlat(georef, cx, cy)
        print(f"偵測中心 {float(lat):.4f}N {float(lon):.4f}E -> 像素 ({cx:.0f}, {cy:.0f})")
        cv2.line(canvas, (int(cx), int(cy)), (ac, ar), YELLOW, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (int(cx), int(cy)), YELLOW,
                       cv2.MARKER_TILTED_CROSS, 24, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"pred {float(lat):.4f}N {float(lon):.4f}E",
                    (int(cx) - 150, int(cy) - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, YELLOW, 2, cv2.LINE_AA)

    cv2.drawMarker(canvas, (ac, ar), CYAN, cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)
    cv2.putText(canvas, "TPE ARP (published)", (ac + 14, ar + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 2, cv2.LINE_AA)

    full = banner(canvas, "Day24  WGS84 graticule + published ground control point")
    cv2.imwrite(str(OUT / "day24_graticule.jpg"), full)

    # 右邊：機場周邊的放大圖，看框跟跑道到底差多少
    x0, y0 = max(ac - 420, 0), max(ar - 330, 0)
    x1, y1 = min(ac + 300, canvas.shape[1]), min(ar + 200, canvas.shape[0])
    crop = cv2.resize(canvas[y0:y1, x0:x1], None, fx=1.9, fy=1.9,
                      interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / "day24_zoom.jpg"), banner(crop, "zoom: box vs runways"))
    print(f"輸出: {OUT}")


if __name__ == "__main__":
    main()
