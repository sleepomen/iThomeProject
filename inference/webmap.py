"""
Day 25: 把偵測結果畫成地圖。

兩種輸出：
  build_map()     folium 互動網頁地圖，可切換底圖、點框看屬性
  render_static() 用 tiles.fetch_mosaic 抓底圖、OpenCV 畫框，給文章放的靜態圖

兩個函式吃的都是 Day 24 的 GeoJSON FeatureCollection，不碰像素座標。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import folium
import numpy as np
from shapely.geometry import mapping

from inference.geo_utils import lonlat_to_pixel
from inference.tiles import SOURCES, fetch_mosaic

HIT_COLOR = "#2ecc71"      # 框過半面積在機場內
MISS_COLOR = "#e74c3c"     # 框沒落在機場上（誤判）
AIRPORT_COLOR = "#3498db"  # OSM 機場邊界


def _popup_html(props: dict) -> str:
    rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                   for k, v in props.items() if k != "gmaps")
    link = f'<a href="{props["gmaps"]}" target="_blank">Google Maps</a>' \
        if "gmaps" in props else ""
    return f"<table style='font-size:12px'>{rows}</table>{link}"


def build_map(detections: dict, airports: list[dict] | None = None,
              markers: list[tuple[float, float, str]] | None = None,
              center: tuple[float, float] | None = None,
              zoom: int = 13) -> folium.Map:
    """
    detections: GeoJSON FeatureCollection，properties 裡有 hit 就依此上色
    airports:   evaluate.load_airports() 的結果，畫成藍色邊界
    markers:    額外的點，(lon, lat, 說明)，例如 Day 24 的 ARP
    center:     (lon, lat)；沒給就取第一個框的中心

    注意 folium 吃的座標是 `(lat, lon)`，跟 GeoJSON 的 `[lon, lat]` 相反，
    但丟 GeoJSON 物件給 folium.GeoJson 時它會自己處理，只有手動放點才要反過來。
    """
    feats = detections.get("features", [])
    if center is None:
        center = tuple(feats[0]["properties"]["center"]) if feats else (121.0, 23.7)

    m = folium.Map(location=[center[1], center[0]], zoom_start=zoom, tiles=None,
                   control_scale=True)
    for key, name in (("s2cloudless", "Sentinel-2 cloudless (10m)"),
                      ("esri", "Esri 高解析衛星"),
                      ("osm", "OpenStreetMap")):
        folium.TileLayer(tiles=SOURCES[key]["url"], attr=SOURCES[key]["attribution"],
                         name=name, max_zoom=SOURCES[key]["max_zoom"]).add_to(m)

    if airports:
        layer = folium.FeatureGroup(name="OSM 機場邊界")
        for ap in airports:
            folium.GeoJson(
                mapping(ap["geom"]),
                style_function=lambda _: {"color": AIRPORT_COLOR, "weight": 2,
                                          "fillOpacity": 0.08, "dashArray": "6 4"},
                tooltip=f"{ap['iata']} {ap.get('name_en') or ap.get('name')}",
            ).add_to(layer)
        layer.add_to(m)

    det_layer = folium.FeatureGroup(name="模型偵測框")
    for f in feats:
        hit = f["properties"].get("hit")
        color = HIT_COLOR if hit else MISS_COLOR if hit is not None else "#f1c40f"
        folium.GeoJson(
            f,
            style_function=lambda _, c=color: {"color": c, "weight": 3, "fillOpacity": 0.15},
            popup=folium.Popup(_popup_html(f["properties"]), max_width=360),
            tooltip=f"score {f['properties']['score']:.2f}",
        ).add_to(det_layer)
    det_layer.add_to(m)

    for lon, lat, label in markers or []:
        folium.Marker([lat, lon], tooltip=label,
                      icon=folium.Icon(color="blue", icon="plane", prefix="fa")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _to_px(georef, coords):
    arr = np.asarray(coords, dtype=np.float64)
    col, row = lonlat_to_pixel(georef, arr[:, 0], arr[:, 1])
    return np.stack([col, row], axis=-1).round().astype(np.int32)


def _rings(geom):
    """Polygon / MultiPolygon 的外環座標，畫圖用。"""
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    return [list(p.exterior.coords) for p in polys]


def render_static(bbox, zoom: int, detections: dict | None = None,
                  airports: list[dict] | None = None,
                  markers: list[tuple[float, float, str]] | None = None,
                  source: str = "esri", title: str | None = None,
                  show_score: bool = True) -> np.ndarray:
    """
    抓 bbox 範圍的底圖，把機場邊界、偵測框、標記點畫上去，回傳 BGR 影像。

    座標流程跟 Day 24 的 GeoTIFF 一模一樣，只是這次 GeoRef 是 EPSG:3857：
        GeoJSON 經緯度 -> lonlat_to_pixel -> 底圖像素
    """
    rgb, georef = fetch_mosaic(bbox, zoom, source)
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = canvas.copy()

    def bgr(hex_color):
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))

    for ap in airports or []:
        for ring in _rings(ap["geom"]):
            pts = _to_px(georef, ring)
            cv2.polylines(canvas, [pts], True, bgr(AIRPORT_COLOR), 2, cv2.LINE_AA)
            cv2.fillPoly(overlay, [pts], bgr(AIRPORT_COLOR))

    for f in (detections or {}).get("features", []):
        hit = f["properties"].get("hit")
        color = bgr(HIT_COLOR if hit else MISS_COLOR if hit is not None else "#f1c40f")
        pts = _to_px(georef, f["geometry"]["coordinates"][0])
        cv2.polylines(canvas, [pts], True, color, 3, cv2.LINE_AA)
        if not show_score:
            continue
        x, y = pts[:, 0].min(), pts[:, 1].min()
        cv2.putText(canvas, f"{f['properties']['score']:.2f}", (int(x), max(int(y) - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    canvas = cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0)

    for lon, lat, label in markers or []:
        (px, py), = _to_px(georef, [(lon, lat)])
        cv2.drawMarker(canvas, (int(px), int(py)), (255, 255, 0), cv2.MARKER_CROSS, 28, 3)
        cv2.putText(canvas, label, (int(px) + 14, int(py) + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    if title:
        bar = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        canvas = np.vstack([bar, canvas])
    return canvas


def save_map(m: folium.Map, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    return path
