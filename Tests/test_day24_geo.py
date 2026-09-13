"""
Day 24: 自己寫的 UTM 逆投影要怎麼驗？

pyproj 被 Smart App Control 擋掉，沒有第二套實作可以對答案，
所以這裡疊三層驗證，一層比一層獨立：

  1. 往返一致 —— 正投影再逆投影回來，只能抓打錯字，抓不到系統性錯誤
  2. 子午線弧長 —— 用數值積分獨立算一次，再對上公開的 WGS84 象限弧長常數
  3. 地面真相 —— 把影像四角與偵測框中心轉成經緯度，看看落在地球上哪裡
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.geo_utils import (  # noqa: E402
    A, E2, UTMZone, _meridional_arc, gmaps_link, lonlat_to_utm,
    pixel_to_lonlat, utm_to_lonlat, zone_from_epsg,
)
from inference.geotiff_io import read_geotiff  # noqa: E402

TIF = ROOT / "data" / "raw" / "som_san_air.tif"

# 公開常數：WGS84 赤道到極點的子午線象限弧長（公尺）
MERIDIAN_QUADRANT = 10001965.729


def test_roundtrip():
    print("=== 驗證 1：往返一致性 ===")
    zone = UTMZone(51, north=True)
    rng = np.random.default_rng(0)
    lon = zone.central_meridian + rng.uniform(-3, 3, 2000)
    lat = rng.uniform(0, 70, 2000)

    east, north = lonlat_to_utm(lon, lat, zone)
    lon2, lat2 = utm_to_lonlat(east, north, zone)

    # 換算成公尺比較有感：1 度緯度約 111 km
    err_m = np.hypot((lon2 - lon) * 111320 * np.cos(np.radians(lat)),
                     (lat2 - lat) * 110574)
    print(f"  2000 個隨機點，最大往返誤差 {err_m.max() * 1000:.4f} mm")
    assert err_m.max() < 1e-3


def test_meridian_arc():
    print("=== 驗證 2：子午線弧長（數值積分 + 公開常數）===")
    # 子午圈曲率半徑 R(phi) 沿緯度積分 = 弧長，跟級數展開是兩條完全不同的路
    def radius(phi):
        return A * (1 - E2) / (1 - E2 * np.sin(phi) ** 2) ** 1.5

    for deg in (10, 25, 45, 60, 90):
        phi = np.radians(deg)
        grid = np.linspace(0.0, phi, 200001)
        numeric = np.trapezoid(radius(grid), grid)
        series = float(_meridional_arc(phi))
        print(f"  phi={deg:2d}°  級數 {series:14.4f}  積分 {numeric:14.4f}  "
              f"差 {abs(series - numeric) * 1000:7.3f} mm")
        assert abs(series - numeric) < 1e-3

    quadrant = float(_meridional_arc(np.radians(90.0)))
    print(f"  象限弧長 {quadrant:.3f} m  vs 公開值 {MERIDIAN_QUADRANT:.3f} m  "
          f"差 {abs(quadrant - MERIDIAN_QUADRANT) * 1000:.1f} mm")
    assert abs(quadrant - MERIDIAN_QUADRANT) < 0.01


def test_central_meridian():
    print("=== 驗證 2b：中央經線上的退化解 ===")
    # 在中央經線上，正投影應該退化成 easting = 500000、northing = k0 * M
    zone = UTMZone(51, north=True)
    for lat in (0.0, 25.0, 60.0):
        e, n = lonlat_to_utm(zone.central_meridian, lat, zone)
        expect_n = 0.9996 * float(_meridional_arc(np.radians(lat)))
        print(f"  lat={lat:5.1f}  E={float(e):12.4f}（應為 500000）"
              f"  N={float(n):13.4f}  誤差 {abs(float(n) - expect_n) * 1000:.4f} mm")
        assert abs(float(e) - 500000.0) < 1e-6
        assert abs(float(n) - expect_n) < 1e-6


def test_ground_truth():
    print("=== 驗證 3：地面真相 ===")
    stack, georef = read_geotiff(TIF)
    h, w = stack.shape[:2]
    zone = zone_from_epsg(georef.epsg)
    print(f"  EPSG:{georef.epsg} -> UTM {zone.zone}{'N' if zone.north else 'S'}"
          f"  中央經線 {zone.central_meridian:.0f}°E")
    print(f"  影像 {w} x {h}")

    corners = {"左上": (0, 0), "右上": (w, 0), "左下": (0, h), "右下": (w, h),
               "中心": (w / 2, h / 2)}
    for name, (c, r) in corners.items():
        lon, lat = pixel_to_lonlat(georef, c, r)
        print(f"  {name}  ({c:6.1f},{r:6.1f}) -> {float(lat):.6f}N, {float(lon):.6f}E")

    lon, lat = pixel_to_lonlat(georef, w / 2, h / 2)
    print(f"  影像中心: {gmaps_link(float(lon), float(lat))}")


if __name__ == "__main__":
    test_roundtrip()
    test_meridian_arc()
    test_central_meridian()
    test_ground_truth()
