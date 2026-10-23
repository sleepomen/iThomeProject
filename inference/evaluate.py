"""
Day 25: 拿 OSM 機場邊界當地面真相，替每個偵測框打分數。

Day 24 的判準是「ARP 在不在框裡」，一個點只能回答是或否。
換成多邊形之後可以問兩個更有用的問題：

    inside_frac  框的面積有幾成落在機場圍籬內（框得準不準）
    coverage     機場的面積有幾成被框到（框得全不全）

面積一定要在`公尺`的平面上算。經緯度的一度在東西向和南北向長度不同，
直接拿度數算面積，台灣這個緯度東西向會被高估約 10%。
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from shapely.geometry import Point, shape
from shapely.ops import transform

from inference.geo_utils import UTMZone, haversine_m, lonlat_to_utm

ROOT = Path(__file__).resolve().parent.parent
AIRPORTS = ROOT / "data" / "reference" / "tw_airports.geojson"

# 框要有過半面積落在機場圍籬內才算`命中`
HIT_THRESHOLD = 0.5


def load_airports(path: str | Path = AIRPORTS) -> list[dict]:
    """讀 OSM 機場邊界，每筆是 {'iata', 'name_en', 'geom'(shapely), ...}。"""
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in fc["features"]:
        rec = dict(f["properties"])
        rec["geom"] = shape(f["geometry"])
        out.append(rec)
    return out


def utm_zone_for(lon: float, lat: float) -> UTMZone:
    return UTMZone(int((lon + 180) // 6) + 1, north=lat >= 0)


def to_metric(geom, zone: UTMZone):
    """把經緯度幾何搬到 UTM 平面（公尺），面積與距離才有意義。"""
    return transform(partial(_utm_xy, zone=zone), geom)


def _utm_xy(x, y, z=None, zone=None):
    return lonlat_to_utm(x, y, zone)


def score_feature(feature: dict, airports: list[dict]) -> dict:
    """
    一個 GeoJSON 偵測框對上所有機場，回傳最佳配對與各項指標。

    回傳的 dict 會直接塞回 feature["properties"]，所以只放可以 JSON 化的值。
    """
    box = shape(feature["geometry"])
    c = box.centroid
    zone = utm_zone_for(c.x, c.y)
    box_m = to_metric(box, zone)

    best = None
    for ap in airports:
        ap_m = to_metric(ap["geom"], zone)
        inter = box_m.intersection(ap_m).area
        dist = box_m.distance(ap_m)               # 0 = 有碰到
        cand = {
            "iata": ap["iata"],
            "inside_frac": inter / box_m.area if box_m.area else 0.0,
            "coverage": inter / ap_m.area if ap_m.area else 0.0,
            "dist_to_airport_m": dist,
            "dist_to_centroid_m": float(haversine_m(c.x, c.y, *ap["centroid"])),
        }
        if best is None or (cand["inside_frac"], -cand["dist_to_airport_m"]) > \
                (best["inside_frac"], -best["dist_to_airport_m"]):
            best = cand

    best = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()}
    best["box_area_km2"] = round(box_m.area / 1e6, 3)
    best["hit"] = best["inside_frac"] >= HIT_THRESHOLD
    return best


def airports_in_view(airports: list[dict], bbox) -> list[dict]:
    """bbox = (west, south, east, north) 範圍內`中心點`有落進來的機場。"""
    w, s, e, n = bbox
    return [ap for ap in airports
            if w <= ap["centroid"][0] <= e and s <= ap["centroid"][1] <= n]


def point_in_airport(lon: float, lat: float, airports: list[dict]) -> str | None:
    p = Point(lon, lat)
    for ap in airports:
        if ap["geom"].contains(p):
            return ap["iata"]
    return None
