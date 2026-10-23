"""
Day 25: 從 OpenStreetMap 抓台灣本島機場的`邊界多邊形`，當作地面真相。

Day 24 只用一個公開的機場基準點（ARP）驗框，一個點只能回答「在不在框裡」，
回答不了「框到的面積有多少是機場」。OSM 的 aeroway=aerodrome 是整座機場的
圍籬範圍，正好對上資料集的類別 `airport`。

輸出：data/reference/tw_airports.geojson（資料授權 ODbL，© OpenStreetMap contributors）
"""
from __future__ import annotations

import json
from pathlib import Path

import requests
from shapely.geometry import LineString, MultiPolygon, Polygon, mapping
from shapely.ops import linemerge, polygonize, unary_union

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "reference" / "tw_airports.geojson"
OVERPASS = "https://overpass-api.de/api/interpreter"

# 台灣本島（不含澎湖、金門、馬祖）
BBOX = (21.8, 119.9, 25.4, 122.1)     # south, west, north, east（Overpass 的順序）

QUERY = f"""
[out:json][timeout:90];
(
  way["aeroway"="aerodrome"]["iata"]{BBOX};
  relation["aeroway"="aerodrome"]["iata"]{BBOX};
);
out geom;
"""


def _way_polygon(el) -> Polygon | None:
    coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
    if len(coords) < 4 or coords[0] != coords[-1]:
        return None
    return Polygon(coords)


def _relation_polygon(el):
    """
    multipolygon relation 的外環常常被拆成好幾段 way，
    要先 linemerge 接起來再 polygonize 才拼得回完整的多邊形。
    """
    outer = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
             for m in el.get("members", [])
             if m.get("role") == "outer" and m.get("geometry")]
    inner = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
             for m in el.get("members", [])
             if m.get("role") == "inner" and m.get("geometry")]
    shell = unary_union(list(polygonize(linemerge(outer))))
    holes = unary_union(list(polygonize(linemerge(inner)))) if inner else None
    return shell.difference(holes) if holes is not None else shell


def main():
    resp = requests.post(OVERPASS, data={"data": QUERY}, timeout=120,
                         headers={"User-Agent": "iThome-airport-demo/1.0"})
    resp.raise_for_status()

    features = []
    for el in resp.json()["elements"]:
        geom = _way_polygon(el) if el["type"] == "way" else _relation_polygon(el)
        if geom is None or geom.is_empty:
            print(f"跳過 {el['type']}/{el['id']}：幾何不完整")
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)           # OSM 偶爾有自相交，buffer(0) 是標準修法
        if isinstance(geom, Polygon):
            geom = MultiPolygon([geom])

        tags = el["tags"]
        c = geom.centroid
        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "iata": tags.get("iata"),
                "icao": tags.get("icao"),
                "name": tags.get("name"),
                "name_en": tags.get("name:en"),
                "military": tags.get("military") == "airfield"
                            or tags.get("landuse") == "military",
                "osm_id": f"{el['type']}/{el['id']}",
                "centroid": [round(c.x, 6), round(c.y, 6)],
            },
        })

    features.sort(key=lambda f: f["properties"]["iata"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "type": "FeatureCollection",
        "attribution": "© OpenStreetMap contributors, ODbL 1.0",
        "features": features,
    }, ensure_ascii=False), encoding="utf-8")

    for f in features:
        p = f["properties"]
        print(f"{p['iata']}  {p['icao'] or '----'}  {p['name_en'] or p['name']}"
              f"{'  (軍用)' if p['military'] else ''}")
    print(f"\n寫出 {OUT}  {len(features)} 座機場")


if __name__ == "__main__":
    main()
