"""
Day 26: 偵測結果進 PostGIS，順便拿容器裡的 PROJ 替 Day 24 手寫的投影對答案。

1. 對答案：Windows 上 pyproj 被 Smart App Control 擋掉，但 PostGIS 容器裡
   有完整的 PROJ。同一批 UTM 座標分別用 geo_utils 與 ST_Transform 轉經緯度，比差多少
2. 建表、灌入 OSM 機場、寫入 Day 24 的偵測
3. 用 SQL 重算 Day 25 的指標，跟 shapely 的結果互相驗證
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import postgis  # noqa: E402
from inference.evaluate import load_airports, score_feature  # noqa: E402
from inference.geo_utils import (  # noqa: E402
    detections_to_geojson, haversine_m, mercator_to_lonlat, utm_to_lonlat, zone_from_epsg,
)
from inference.pipeline import AirportDetector  # noqa: E402

TIF = ROOT / "data" / "raw" / "som_san_air.tif"
MODEL = "airport_obb_mixed"
TPE_ARP = (121.2328, 25.0767)


def _proj_lonlat(cur, xs, ys, epsg):
    """用 PostGIS 的 ST_Transform（底層是 PROJ）把投影座標轉成經緯度。"""
    cur.execute("""
        SELECT ST_X(g), ST_Y(g) FROM (
            SELECT ST_Transform(ST_SetSRID(ST_MakePoint(x, y), %s), 4326) AS g
            FROM unnest(%s::float8[], %s::float8[]) AS t(x, y)
        ) s
    """, (epsg, np.asarray(xs).tolist(), np.asarray(ys).tolist()))
    return np.array(cur.fetchall())


def _proj_utm(cur, lon, lat, epsg):
    """反方向：經緯度 -> UTM，也交給 PROJ，讓測試點完全不經過我們的程式。"""
    cur.execute("""
        SELECT ST_X(g), ST_Y(g) FROM (
            SELECT ST_Transform(ST_SetSRID(ST_MakePoint(x, y), 4326), %s) AS g
            FROM unnest(%s::float8[], %s::float8[]) AS t(x, y)
        ) s
    """, (epsg, np.asarray(lon).tolist(), np.asarray(lat).tolist()))
    return np.array(cur.fetchall())


def check_against_proj(cur, georef):
    """geo_utils vs PROJ。分三種範圍，看級數解在哪裡開始失準。"""
    rng = np.random.default_rng(26)
    zone = zone_from_epsg(georef.epsg)
    cm = zone.central_meridian

    def report(label, lon_true, lat_true):
        utm = _proj_utm(cur, lon_true, lat_true, georef.epsg)
        lon, lat = utm_to_lonlat(utm[:, 0], utm[:, 1], zone)
        err = haversine_m(lon, lat, lon_true, lat_true) * 1000
        print(f"  {label:<22} {len(lon_true)} 點  最大 {err.max():8.4f} mm  "
              f"中位數 {np.median(err):.4f} mm")

    # (a) 影像範圍內：實際會用到的區域，直接拿像素換出來的 UTM 座標比
    cols, rows = rng.uniform(0, 1349, 500), rng.uniform(0, 923, 500)
    xa, ya = georef.pixel_to_proj(cols, rows)
    ref = _proj_lonlat(cur, xa, ya, georef.epsg)
    lon, lat = utm_to_lonlat(xa, ya, zone)
    err = haversine_m(lon, lat, ref[:, 0], ref[:, 1]) * 1000
    print(f"  {'影像範圍內（桃園）':<22} 500 點  最大 {err.max():8.4f} mm  "
          f"中位數 {np.median(err):.4f} mm")

    # (b) 分帶內：中央經線 ±3°、赤道到 60°N
    report("分帶內 ±3°, 0~60°N", rng.uniform(cm - 3, cm + 3, 500), rng.uniform(0, 60, 500))
    # (c) 分帶外：±3° ~ ±6°，UTM 本來就不該這樣用，看看會壞多少
    off = rng.uniform(3, 6, 500) * rng.choice([-1, 1], 500)
    report("分帶外 ±3~6°, 0~60°N", cm + off, rng.uniform(0, 60, 500))

    xs = rng.uniform(-2e7, 2e7, 500)
    ys = rng.uniform(-1.5e7, 1.5e7, 500)
    ref = _proj_lonlat(cur, xs, ys, 3857)
    lon, lat = mercator_to_lonlat(xs, ys)
    err = haversine_m(lon, lat, ref[:, 0], ref[:, 1]) * 1000
    print(f"  {'Web Mercator 全球':<22} 500 點  最大 {err.max():8.6f} mm")


def main():
    result = AirportDetector().detect(TIF)
    fc = detections_to_geojson(result.detections, result.georef, source=TIF.name, model=MODEL)
    airports = load_airports()

    with postgis.connect() as cur:
        cur.execute("SELECT postgis_full_version()")
        ver = cur.fetchone()[0]
        print("PostGIS:", ver.split(" LIBXML")[0])

        print("\n[1] geo_utils 對 PROJ")
        check_against_proj(cur, result.georef)

        print("\n[2] 建表與寫入")
        postgis.init_schema(cur)
        n = postgis.upsert_airports(cur, airports)
        removed = postgis.delete_runs(cur, TIF.name, MODEL)
        run_id = postgis.insert_run(cur, TIF.name, MODEL, result.footprint(), fc,
                                    params={"slice": 320, "overlap": 0.2, "conf": 0.25})
        print(f"  airports {n} 筆，刪除舊 run {removed} 筆，新 run_id = {run_id}")

        print("\n[3] SQL 重算 Day 25 的指標（括號內是 shapely 的結果）")
        py = [score_feature(f, airports) for f in fc["features"]]
        for row, ref in zip(postgis.score_run(cur, run_id), py):
            print(f"  det#{row['id']} score={row['score']:.3f} -> {row['iata']}")
            print(f"    inside_frac {row['inside_frac']:.2%}  ({ref['inside_frac']:.2%})")
            print(f"    coverage    {row['coverage']:.2%}  ({ref['coverage']:.2%})")
            print(f"    box_km2     {row['box_km2']:.3f}  ({ref['box_area_km2']:.3f})")

        cur.execute("""
            SELECT ST_Contains(d.geom, p), ST_Distance(d.geom::geography, p::geography),
                   ST_Distance(ST_Centroid(d.geom)::geography, p::geography)
            FROM detections d, ST_SetSRID(ST_MakePoint(%s, %s), 4326) AS p
            WHERE d.run_id = %s
        """, (*TPE_ARP, run_id))
        contains, d_edge, d_center = cur.fetchone()
        print(f"\n  ARP 在框內? {contains}  到框邊 {d_edge:,.0f} m  到框中心 {d_center:,.0f} m")

        print("\n[4] 這張圖裡有哪些機場沒被框到")
        print("  ", postgis.missed_airports(cur, run_id) or "沒有")

        cur.execute("""
            SELECT a.iata, round((ST_Centroid(d.geom)::geography <-> a.geom::geography)::numeric)
            FROM detections d, airports a WHERE d.run_id = %s
            ORDER BY ST_Centroid(d.geom) <-> a.geom LIMIT 3
        """, (run_id,))
        print("\n[5] 離框中心最近的三座機場（KNN）")
        for iata, dist in cur.fetchall():
            print(f"   {iata}  {int(dist):,} m")


if __name__ == "__main__":
    main()
