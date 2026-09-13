"""
Day 24: 純 Python 的 UTM <-> WGS84 轉換，取代 pyproj。

Day 15 的版本靠 `rasterio` + `pyproj`，兩個套件現在都被 Windows 的
Smart App Control 擋掉（見 Day 23 / Day 24）。讀圖的部分 Day 23 已經用
tifffile 換掉了，剩下的座標轉換這裡自己算。

投影公式用 Snyder《Map Projections: A Working Manual》的橫麥卡托級數解，
在 UTM 分帶（中央經線 ±3°）範圍內誤差小於 1 mm，對 10m 解析度的
Sentinel-2 來說綽綽有餘。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------- WGS84 橢球與 UTM 常數 ----------------
A = 6378137.0                       # 長半軸（赤道半徑）
F = 1.0 / 298.257223563             # 扁率
E2 = F * (2.0 - F)                  # 第一偏心率平方 e^2
EP2 = E2 / (1.0 - E2)               # 第二偏心率平方 e'^2
K0 = 0.9996                         # UTM 中央經線縮放比
FALSE_EASTING = 500000.0
FALSE_NORTHING_SOUTH = 10000000.0   # 南半球才有的 y 偏移


@dataclass(frozen=True)
class UTMZone:
    """一個 UTM 分帶。EPSG 326xx = 北半球 xx 帶，327xx = 南半球。"""
    zone: int
    north: bool

    @property
    def central_meridian(self) -> float:
        """分帶的中央經線（度）。zone 1 是 -177，每帶 6 度。"""
        return self.zone * 6.0 - 183.0

    @property
    def false_northing(self) -> float:
        return 0.0 if self.north else FALSE_NORTHING_SOUTH

    @property
    def epsg(self) -> int:
        return (32600 if self.north else 32700) + self.zone


def zone_from_epsg(epsg: int) -> UTMZone:
    """把 EPSG 代碼拆成分帶號與南北半球。"""
    if epsg is None:
        raise ValueError("GeoTIFF 沒有帶 EPSG 代碼，無法決定投影")
    if 32601 <= epsg <= 32660:
        return UTMZone(epsg - 32600, north=True)
    if 32701 <= epsg <= 32760:
        return UTMZone(epsg - 32700, north=False)
    raise ValueError(f"EPSG:{epsg} 不是 WGS84 UTM 分帶，這裡只實作了 UTM")


def _meridional_arc(phi: np.ndarray) -> np.ndarray:
    """從赤道到緯度 phi 的子午線弧長 M（公尺）。"""
    return A * (
        (1 - E2 / 4 - 3 * E2**2 / 64 - 5 * E2**3 / 256) * phi
        - (3 * E2 / 8 + 3 * E2**2 / 32 + 45 * E2**3 / 1024) * np.sin(2 * phi)
        + (15 * E2**2 / 256 + 45 * E2**3 / 1024) * np.sin(4 * phi)
        - (35 * E2**3 / 3072) * np.sin(6 * phi)
    )


def utm_to_lonlat(easting, northing, zone: UTMZone):
    """UTM 投影座標 -> WGS84 經緯度（度）。支援純量與 numpy 陣列。"""
    x = np.asarray(easting, dtype=np.float64) - FALSE_EASTING
    y = np.asarray(northing, dtype=np.float64) - zone.false_northing

    # 1. 由弧長反推`足點緯度` phi1：先算等量的球面緯度 mu，再用級數修正回橢球
    m = y / K0
    e1 = (1 - np.sqrt(1 - E2)) / (1 + np.sqrt(1 - E2))
    mu = m / (A * (1 - E2 / 4 - 3 * E2**2 / 64 - 5 * E2**3 / 256))
    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1**3 / 32) * np.sin(2 * mu)
            + (21 * e1**2 / 16 - 55 * e1**4 / 32) * np.sin(4 * mu)
            + (151 * e1**3 / 96) * np.sin(6 * mu)
            + (1097 * e1**4 / 512) * np.sin(8 * mu))

    # 2. 在足點上求曲率半徑，再沿著垂直中央經線的方向展開級數
    sin1, cos1, tan1 = np.sin(phi1), np.cos(phi1), np.tan(phi1)
    c1 = EP2 * cos1**2
    t1 = tan1**2
    n1 = A / np.sqrt(1 - E2 * sin1**2)                    # 卯酉圈曲率半徑
    r1 = A * (1 - E2) / (1 - E2 * sin1**2) ** 1.5         # 子午圈曲率半徑
    d = x / (n1 * K0)

    lat = phi1 - (n1 * tan1 / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * EP2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * EP2 - 3 * c1**2) * d**6 / 720
    )
    lon = np.radians(zone.central_meridian) + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * EP2 + 24 * t1**2) * d**5 / 120
    ) / cos1

    return np.degrees(lon), np.degrees(lat)


def lonlat_to_utm(lon, lat, zone: UTMZone):
    """WGS84 經緯度 -> UTM 投影座標。只用來做往返驗證。"""
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    dlam = np.radians(np.asarray(lon, dtype=np.float64) - zone.central_meridian)

    sin_p, cos_p, tan_p = np.sin(phi), np.cos(phi), np.tan(phi)
    n = A / np.sqrt(1 - E2 * sin_p**2)
    t = tan_p**2
    c = EP2 * cos_p**2
    a_ = dlam * cos_p
    m = _meridional_arc(phi)

    east = FALSE_EASTING + K0 * n * (
        a_
        + (1 - t + c) * a_**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * EP2) * a_**5 / 120
    )
    north = zone.false_northing + K0 * (
        m + n * tan_p * (
            a_**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a_**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * EP2) * a_**6 / 720
        )
    )
    return east, north


def pixel_to_lonlat(georef, col, row):
    """
    像素 (col, row) -> WGS84 (lon, lat)。

    兩步：仿射矩陣把像素換成投影座標（Day 23 的 GeoRef 負責），
    再由這裡的逆投影換成經緯度。
    """
    x, y = georef.pixel_to_proj(col, row)
    return utm_to_lonlat(x, y, zone_from_epsg(georef.epsg))


def gmaps_link(lon: float, lat: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


# ---------------- GeoJSON 輸出 ----------------
def _ring_ccw(ring: list[list[float]]) -> list[list[float]]:
    """
    RFC 7946 要求多邊形外環`逆時針`（右手定則）。

    像素座標的 row 往下增加、經緯度的 lat 往上增加，換算之後繞行方向會`翻面`，
    所以不能沿用像素空間的順序，得在經緯度上重算一次帶號面積。
    """
    area = sum(ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
               for i in range(len(ring) - 1))
    return ring if area > 0 else ring[::-1]


def detection_to_feature(det, georef, **extra) -> dict:
    """把一個 Detection 轉成 GeoJSON Feature（Polygon，WGS84 經緯度）。"""
    pts = np.asarray(det.points, dtype=np.float64)
    lon, lat = pixel_to_lonlat(georef, pts[:, 0], pts[:, 1])
    ring = [[float(a), float(b)] for a, b in zip(lon, lat)]
    ring.append(ring[0])                       # GeoJSON 的環必須閉合

    cx, cy = det.center
    clon, clat = pixel_to_lonlat(georef, cx, cy)
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [_ring_ccw(ring)]},
        "properties": {
            "score": round(float(det.score), 4),
            "n_fragments": int(det.n_fragments),
            "angle_deg": round(float(det.angle), 2),
            "length_px": round(float(det.length), 1),
            "width_px": round(float(det.width), 1),
            "aspect": round(float(det.aspect), 2),
            "center": [round(float(clon), 6), round(float(clat), 6)],
            "gmaps": gmaps_link(float(clon), float(clat)),
            **extra,
        },
    }


def detections_to_geojson(detections, georef, **extra) -> dict:
    """整份 FeatureCollection。RFC 7946 拿掉了 crs 欄位，座標一律是 WGS84。"""
    return {
        "type": "FeatureCollection",
        "features": [detection_to_feature(d, georef, **extra) for d in detections],
    }


def lonlat_to_pixel(georef, lon, lat):
    """
    WGS84 (lon, lat) -> 像素 (col, row)，`pixel_to_lonlat` 的反向。

    GeoRef 存的是 2x3 仿射矩陣，不是方陣沒辦法直接求逆，
    補上一列 [0, 0, 1] 湊成 3x3（齊次座標）之後再反矩陣。
    """
    x, y = lonlat_to_utm(lon, lat, zone_from_epsg(georef.epsg))
    m = np.vstack([georef.transform, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(m)
    pts = np.stack([np.asarray(x, dtype=np.float64),
                    np.asarray(y, dtype=np.float64),
                    np.ones_like(np.asarray(x, dtype=np.float64))], axis=-1)
    out = pts @ inv.T
    return out[..., 0], out[..., 1]
