"""
Day 30: 端到端演示的收尾圖。

前面用 CLI 跑完：
    python -m inference.cli fetch 120.7305 22.0382 --name hcn
    python -m inference.cli detect data/s2cloudless/hcn.tif data/raw/som_san_air.tif \
        --map --db -o runs/predict/day30_demo

這支腳本只做兩件事：把 CLI 吐出的 GeoJSON 畫到高解析底圖上，
以及從 PostGIS 撈出 Day 27 ~ Day 30 所有紀錄做成總表。
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import postgis  # noqa: E402
from inference.evaluate import load_airports  # noqa: E402
from inference.tiles import bbox_around  # noqa: E402
from inference.webmap import render_static  # noqa: E402

OUT = ROOT / "runs" / "predict" / "day30_demo"


def main():
    fc = json.loads((OUT / "hcn.geojson").read_text(encoding="utf-8"))
    hcn = [a for a in load_airports() if a["iata"] == "HCN"]
    for f in fc["features"]:
        p = f["properties"]
        print(f"score={p['score']:.3f}  hit={p['hit']}  在機場內 {p['inside_frac']:.1%}"
              f"  {p['box_area_km2']:.2f} km²  距機場 {p['dist_to_airport_m']:,.0f} m")

    img = render_static(bbox_around(120.7305, 22.0382, 12000, 8000), 14, fc, hcn,
                        source="esri", title="HCN Hengchun: CLI output on Esri imagery")
    cv2.imwrite(str(OUT / "day30_hcn.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

    with postgis.connect() as cur:
        cur.execute("""
            WITH d AS (
                SELECT r.params->>'tag' AS tag, d.id,
                       EXISTS (SELECT 1 FROM airports a
                               WHERE ST_Area(ST_Intersection(d.geom, a.geom)::geography)
                                     >= 0.5 * ST_Area(d.geom::geography)) AS hit
                FROM runs r JOIN detections d ON d.run_id = r.id
                WHERE r.params ? 'tag'
            )
            SELECT tag, count(*) AS boxes, count(*) FILTER (WHERE hit) AS hits
            FROM d GROUP BY tag ORDER BY tag
        """)
        print(f"\n{'實驗':<10}{'框':>6}{'命中':>6}{'精確率':>9}")
        for tag, boxes, hits in cur.fetchall():
            print(f"{tag:<10}{boxes:>6}{hits:>6}{hits / boxes:>9.1%}")

        cur.execute("SELECT count(*) FROM runs")
        n_runs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM detections")
        print(f"\n資料庫裡共 {n_runs} 次偵測、{cur.fetchone()[0]} 個框")


if __name__ == "__main__":
    main()
