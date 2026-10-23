"""
Day 26: 偵測結果落地到 PostGIS。

Day 3 開好的 docker-compose（postgis/postgis:15-3.3）從 Day 8 之後就沒動過，
今天終於把它接上。三張表：

    airports    OSM 機場圍籬（地面真相）
    runs        每跑一次偵測一筆：哪張圖、哪個模型、影像涵蓋範圍 footprint
    detections  偵測框，掛在某個 run 底下

footprint 很重要：沒有它就只能問「框到了什麼」，問不了
「這張圖裡`明明有`哪座機場卻沒框到」（漏判）。
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json, execute_values
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS airports (
    iata      text PRIMARY KEY,
    icao      text,
    name      text,
    name_en   text,
    military  boolean,
    osm_id    text,
    geom      geometry(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS airports_geom_idx ON airports USING GIST (geom);

CREATE TABLE IF NOT EXISTS runs (
    id          serial PRIMARY KEY,
    source      text NOT NULL,
    model       text NOT NULL,
    params      jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    footprint   geometry(Polygon, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_footprint_idx ON runs USING GIST (footprint);

CREATE TABLE IF NOT EXISTS detections (
    id           serial PRIMARY KEY,
    run_id       integer NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    score        real NOT NULL,
    n_fragments  integer,
    angle_deg    real,
    length_px    real,
    width_px     real,
    props        jsonb,
    geom         geometry(Polygon, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS detections_geom_idx ON detections USING GIST (geom);
CREATE INDEX IF NOT EXISTS detections_run_idx ON detections (run_id);
"""


def dsn() -> dict:
    load_dotenv(ROOT / ".env")
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "vessel"),
        "password": os.environ["POSTGRES_PASSWORD"],
        "dbname": os.getenv("POSTGRES_DB", "gis_data"),
    }


@contextmanager
def connect():
    """with connect() as cur: ...  離開時自動 commit，出錯就 rollback。"""
    conn = psycopg2.connect(**dsn())
    try:
        with conn, conn.cursor() as cur:
            yield cur
    finally:
        conn.close()


def init_schema(cur) -> None:
    cur.execute(SCHEMA)


def upsert_airports(cur, airports: list[dict]) -> int:
    """
    OSM 的資料會更新，用 iata 當主鍵 upsert，重跑不會重複插入。
    幾何以 GeoJSON 文字傳進去，由 ST_GeomFromGeoJSON 解析，
    比在 Python 端組 WKT 不容易把經緯度順序搞反。
    """
    rows = [(ap["iata"], ap.get("icao"), ap.get("name"), ap.get("name_en"),
             ap.get("military"), ap.get("osm_id"), json.dumps(mapping(ap["geom"])))
            for ap in airports]
    execute_values(cur, """
        INSERT INTO airports (iata, icao, name, name_en, military, osm_id, geom)
        VALUES %s
        ON CONFLICT (iata) DO UPDATE SET
            icao = EXCLUDED.icao, name = EXCLUDED.name, name_en = EXCLUDED.name_en,
            military = EXCLUDED.military, osm_id = EXCLUDED.osm_id, geom = EXCLUDED.geom
    """, rows, template="(%s, %s, %s, %s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))")
    return len(rows)


def insert_run(cur, source: str, model: str, footprint: list[list[float]],
               feature_collection: dict, params: dict | None = None) -> int:
    """
    寫入一次偵測：一筆 runs + N 筆 detections，回傳 run_id。

    footprint 是影像四個角的 [lon, lat]（不用閉合，這裡補）。
    """
    ring = [list(map(float, p)) for p in footprint]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    cur.execute("""
        INSERT INTO runs (source, model, params, footprint)
        VALUES (%s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
        RETURNING id
    """, (source, model, Json(params or {}),
          json.dumps({"type": "Polygon", "coordinates": [ring]})))
    run_id = cur.fetchone()[0]

    keep = {"score", "n_fragments", "angle_deg", "length_px", "width_px"}
    rows = []
    for f in feature_collection.get("features", []):
        p = f["properties"]
        rows.append((run_id, p["score"], p.get("n_fragments"), p.get("angle_deg"),
                     p.get("length_px"), p.get("width_px"),
                     Json({k: v for k, v in p.items() if k not in keep}),
                     json.dumps(f["geometry"])))
    if rows:
        execute_values(cur, """
            INSERT INTO detections
                (run_id, score, n_fragments, angle_deg, length_px, width_px, props, geom)
            VALUES %s
        """, rows, template="(%s, %s, %s, %s, %s, %s, %s, "
                            "ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))")
    return run_id


# ---------------- 查詢 ----------------
# 面積與距離都轉成 geography 再算：geometry(4326) 的單位是`度`，
# ST_Area 會吐出「平方度」這種沒有意義的數字。

SQL_SCORE_RUN = """
SELECT d.id, d.score,
       a.iata,
       ST_Area(ST_Intersection(d.geom, a.geom)::geography)
           / ST_Area(d.geom::geography)                       AS inside_frac,
       ST_Area(ST_Intersection(d.geom, a.geom)::geography)
           / ST_Area(a.geom::geography)                       AS coverage,
       ST_Area(d.geom::geography) / 1e6                       AS box_km2
FROM detections d
LEFT JOIN LATERAL (
    -- 每個框只配一座機場：重疊面積最大的那座；完全沒碰到就配最近的
    SELECT a.* FROM airports a
    ORDER BY ST_Area(ST_Intersection(d.geom, a.geom)::geography) DESC,
             d.geom <-> a.geom
    LIMIT 1
) a ON true
WHERE d.run_id = %s
ORDER BY d.score DESC
"""

SQL_MISSED_AIRPORTS = """
-- 漏判：機場整個落在影像範圍內，卻沒有任何一個框跟它重疊
SELECT a.iata, a.name_en
FROM runs r
JOIN airports a ON ST_Within(a.geom, r.footprint)
WHERE r.id = %s
  AND NOT EXISTS (
      SELECT 1 FROM detections d
      WHERE d.run_id = r.id AND ST_Intersects(d.geom, a.geom))
"""


def score_run(cur, run_id: int) -> list[dict]:
    cur.execute(SQL_SCORE_RUN, (run_id,))
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def missed_airports(cur, run_id: int) -> list[tuple[str, str]]:
    cur.execute(SQL_MISSED_AIRPORTS, (run_id,))
    return cur.fetchall()


def delete_runs(cur, source: str, model: str) -> int:
    """刪掉同一張圖、同一個模型的舊紀錄（detections 會跟著 CASCADE 刪掉）。"""
    cur.execute("DELETE FROM runs WHERE source = %s AND model = %s", (source, model))
    return cur.rowcount
