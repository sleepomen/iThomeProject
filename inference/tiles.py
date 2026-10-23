"""
Day 25: 網路圖磚（XYZ / WMTS）工具。

folium 在瀏覽器裡畫底圖，但文章要放靜態圖、Day 27 要拿圖磚當測試影像，
都得在 Python 這邊自己把圖磚抓下來拼起來。

圖磚世界的規則只有三條：
  1. 投影一律是 Web Mercator（EPSG:3857）
  2. 第 z 層把整個世界切成 2^z x 2^z 張 256 px 的方塊
  3. 方塊編號 (x, y) 從左上角（180°W, 85.05°N）開始數
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import requests

from inference.geo_utils import (
    MERCATOR_HALF, WEB_MERCATOR, lonlat_to_mercator,
)
from inference.geotiff_io import GeoRef

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "tiles_cache"
TILE = 256
USER_AGENT = "iThome-airport-demo/1.0 (personal blog, non-commercial)"

SOURCES = {
    # EOX 的 Sentinel-2 cloudless：把一整年的 Sentinel-2 L2A 去雲後拼成的鑲嵌圖，
    # 10m 解析度、真彩色（B4/B3/B2）。不用帳號就能抓，Day 27 拿來代替 Copernicus。
    "s2cloudless": {
        "url": ("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/"
                "default/GoogleMapsCompatible/{z}/{y}/{x}.jpg"),
        "attribution": ('<a href="https://s2maps.eu">Sentinel-2 cloudless</a> by '
                        "EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2020)"),
        "max_zoom": 15,
    },
    # Esri 的高解析度影像，0.3~1m 等級，只拿來給人眼對照，不餵模型
    "esri": {
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        "attribution": "Tiles &copy; Esri &mdash; Esri, Maxar, Earthstar Geographics",
        "max_zoom": 18,
    },
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "&copy; OpenStreetMap contributors",
        "max_zoom": 19,
    },
}


def resolution(zoom: int) -> float:
    """第 zoom 層一個像素在 Web Mercator 平面上是幾公尺（赤道上的實地距離）。"""
    return 2 * MERCATOR_HALF / (TILE * 2 ** zoom)


def ground_resolution(zoom: int, lat: float) -> float:
    """
    同一層在緯度 lat 的`實地`解析度。

    Web Mercator 會把高緯度拉大 1/cos(lat) 倍，所以平面上的 1 px
    在台灣（約 24°N）只有赤道的 0.91 倍長。
    """
    return resolution(zoom) * np.cos(np.radians(lat))


def _global_px(lon, lat, zoom):
    """經緯度 -> 第 zoom 層的全域像素座標（世界左上角為原點）。"""
    x, y = lonlat_to_mercator(lon, lat)
    res = resolution(zoom)
    return (x + MERCATOR_HALF) / res, (MERCATOR_HALF - y) / res


def _fetch_tile(source: str, z: int, x: int, y: int,
                session: requests.Session) -> np.ndarray:
    """抓一張圖磚，先查本機快取。圖磚伺服器是別人的，同一張不要抓兩次。"""
    cfg = SOURCES[source]
    ext = ".png" if cfg["url"].endswith(".png") else ".jpg"
    path = CACHE_DIR / source / str(z) / str(x) / f"{y}{ext}"
    if not path.exists():
        url = cfg["url"].format(z=z, x=x, y=y)
        for attempt in range(3):
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                break
            time.sleep(1.5 * (attempt + 1))
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"圖磚解碼失敗: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fetch_mosaic(bbox: tuple[float, float, float, float], zoom: int,
                 source: str = "s2cloudless") -> tuple[np.ndarray, GeoRef]:
    """
    抓下 bbox = (west, south, east, north) 範圍的圖磚，拼成一張 RGB 影像。

    回傳的 GeoRef 是 EPSG:3857 的仿射矩陣，可以直接丟進
    `pixel_to_lonlat` / `lonlat_to_pixel`，跟 Day 24 的 UTM GeoTIFF 用法一樣。
    """
    if zoom > SOURCES[source]["max_zoom"]:
        raise ValueError(f"{source} 最多到第 {SOURCES[source]['max_zoom']} 層")
    west, south, east, north = bbox
    px0, py0 = _global_px(west, north, zoom)
    px1, py1 = _global_px(east, south, zoom)
    px0, py0 = int(np.floor(px0)), int(np.floor(py0))
    px1, py1 = int(np.ceil(px1)), int(np.ceil(py1))

    tx0, ty0 = px0 // TILE, py0 // TILE
    tx1, ty1 = (px1 - 1) // TILE, (py1 - 1) // TILE

    canvas = np.zeros(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 3), dtype=np.uint8)
    with requests.Session() as s:
        s.headers["User-Agent"] = USER_AGENT
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile = _fetch_tile(source, zoom, tx, ty, s)
                canvas[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE,
                       (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = tile[:TILE, :TILE]

    # 裁掉超出 bbox 的部分，只留下剛好覆蓋 bbox 的像素
    ox, oy = px0 - tx0 * TILE, py0 - ty0 * TILE
    rgb = canvas[oy:oy + (py1 - py0), ox:ox + (px1 - px0)].copy()

    res = resolution(zoom)
    transform = np.array([[res, 0.0, px0 * res - MERCATOR_HALF],
                          [0.0, -res, MERCATOR_HALF - py0 * res]])
    return rgb, GeoRef(transform=transform, epsg=WEB_MERCATOR)


def bbox_around(lon: float, lat: float, width_m: float, height_m: float):
    """以 (lon, lat) 為中心、實地寬高為 width_m x height_m 的經緯度範圍。"""
    dlat = height_m / 2 / 111_320.0
    dlon = width_m / 2 / (111_320.0 * np.cos(np.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def resample_to_ground(rgb: np.ndarray, georef: GeoRef, lat: float,
                       target: float = 10.0) -> tuple[np.ndarray, GeoRef]:
    """
    Day 27: Web Mercator 的像素在台灣只有 8.7m（實地），縮成 10m 再餵模型。

    縮小之後仿射矩陣的像素大小要跟著放大同樣倍率，原點（左上角）不變，
    座標才對得上。這裡的 scale 是`投影平面`上的，實地解析度隨緯度變，
    所以用影像中心緯度換算。
    """
    factor = georef.transform[0, 0] * np.cos(np.radians(lat)) / target
    h, w = rgb.shape[:2]
    new_w, new_h = round(w * factor), round(h * factor)
    out = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    t = georef.transform.copy()
    t[0, 0] *= w / new_w
    t[1, 1] *= h / new_h
    return out, GeoRef(transform=t, epsg=georef.epsg)
