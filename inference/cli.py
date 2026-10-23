"""
Day 29: 把 Day 23 ~ Day 28 的東西包成一個指令。

    python -m inference.cli fetch 121.55 25.07 --name tsa
    python -m inference.cli detect data/s2cloudless/tsa.tif --map --db
    python -m inference.cli runs
    python -m inference.cli db-init

之前每做一件事就要開一支 Tests/ 腳本、改裡面寫死的路徑。
CLI 的價值不在省打字，而是把「一次偵測」該產出的東西固定下來：
GeoJSON、評分、地圖、資料庫紀錄，每次都一樣，不會漏。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="衛星影像機場偵測：GeoTIFF -> 偵測 -> GeoJSON -> 地圖 / PostGIS")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path("runs") / "cli"


def _parse_bands(text: str) -> tuple[int, int, int]:
    parts = [int(x) for x in text.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter("--bands 要三個以逗號分隔的索引，例如 0,1,2")
    return tuple(parts)


@app.command()
def detect(
    tifs: list[Path] = typer.Argument(..., exists=True, dir_okay=False,
                                      help="一或多張 GeoTIFF"),
    out: Path = typer.Option(DEFAULT_OUT, "--out", "-o", help="輸出資料夾"),
    weights: Optional[Path] = typer.Option(None, help="權重檔，預設是 Day 22 的混合模型"),
    confidence: float = typer.Option(0.25, "--conf", min=0.0, max=1.0),
    bands: str = typer.Option("0,1,2", help="拿哪三個波段當 RGB"),
    stretch: str = typer.Option("max", help="max 或 percentile"),
    score: bool = typer.Option(True, help="用 OSM 機場圍籬替每個框打分數"),
    make_map: bool = typer.Option(False, "--map", help="輸出 folium 互動地圖"),
    db: bool = typer.Option(False, "--db", help="寫入 PostGIS（需要 docker compose up）"),
    device: str = typer.Option("cuda:0", help="cuda:0 或 cpu"),
):
    """對 GeoTIFF 跑偵測，輸出 GeoJSON（可選：評分、地圖、資料庫）。"""
    # 各指令只 import 自己用得到的模組：沒加 --db 就不需要 psycopg2，
    # fetch / runs 也不必載入 sahi 與偵測管線
    from inference.evaluate import load_airports, score_feature
    from inference.geo_utils import detections_to_geojson
    from inference.pipeline import DEFAULT_WEIGHTS, AirportDetector

    out.mkdir(parents=True, exist_ok=True)
    detector = AirportDetector(weights=weights or DEFAULT_WEIGHTS, confidence=confidence,
                               bands=_parse_bands(bands), stretch=stretch, device=device)
    model_name = Path(detector.weights).parent.parent.name
    airports = load_airports() if score else []

    summary = []
    for tif in tifs:
        t0 = time.perf_counter()
        result = detector.detect(tif)
        fc = detections_to_geojson(result.detections, result.georef,
                                   source=tif.name, model=model_name)
        for f in fc["features"]:
            if airports:
                f["properties"].update(score_feature(f, airports))
        elapsed = time.perf_counter() - t0

        stem = out / tif.stem
        stem.with_suffix(".geojson").write_text(
            json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")

        if make_map:
            from inference.webmap import build_map, save_map
            lon = sum(p[0] for p in result.footprint()) / 4
            lat = sum(p[1] for p in result.footprint()) / 4
            save_map(build_map(fc, airports=airports, center=(lon, lat), zoom=13),
                     stem.with_suffix(".html"))

        run_id = None
        if db:
            from inference import postgis
            with postgis.connect() as cur:
                postgis.init_schema(cur)
                run_id = postgis.insert_run(
                    cur, tif.name, model_name, result.footprint(), fc,
                    params={"conf": confidence, "bands": bands, "stretch": stretch})

        hits = sum(bool(f["properties"].get("hit")) for f in fc["features"])
        summary.append((tif.name, result.raw_count, len(fc["features"]), hits, elapsed, run_id))

    typer.echo(f"\n{'file':<24}{'raw':>5}{'boxes':>7}{'hit':>5}{'sec':>7}  run_id")
    for name, raw, boxes, hits, sec, run_id in summary:
        typer.echo(f"{name:<24}{raw:>5}{boxes:>7}{hits if score else '-':>5}"
                   f"{sec:>7.1f}  {run_id or '-'}")
    typer.echo(f"\n輸出 -> {out}")


@app.command()
def fetch(
    lon: float = typer.Argument(..., help="中心經度"),
    lat: float = typer.Argument(..., help="中心緯度"),
    name: str = typer.Option(..., "--name", "-n", help="輸出檔名（不含副檔名）"),
    width: float = typer.Option(12_000, help="實地寬度（公尺）"),
    height: float = typer.Option(8_000, help="實地高度（公尺）"),
    out: Path = typer.Option(Path("data") / "s2cloudless", "--out", "-o"),
):
    """從 EOX Sentinel-2 cloudless 抓一塊範圍，存成 10m 的 GeoTIFF。"""
    from inference.geotiff_io import write_geotiff
    from inference.tiles import bbox_around, fetch_mosaic, resample_to_ground

    rgb, georef = fetch_mosaic(bbox_around(lon, lat, width, height), 14, "s2cloudless")
    rgb, georef = resample_to_ground(rgb, georef, lat)
    path = write_geotiff(out / f"{name}.tif", rgb, georef)
    typer.echo(f"{path}  {rgb.shape[1]}x{rgb.shape[0]} px")


@app.command("db-init")
def db_init():
    """建表並灌入 OSM 機場圍籬。"""
    from inference import postgis
    from inference.evaluate import load_airports

    with postgis.connect() as cur:
        postgis.init_schema(cur)
        n = postgis.upsert_airports(cur, load_airports())
    typer.echo(f"schema OK，airports {n} 筆")


@app.command()
def runs(limit: int = typer.Option(20, help="顯示最近幾筆")):
    """列出資料庫裡最近的偵測紀錄與命中情況。"""
    from inference import postgis

    with postgis.connect() as cur:
        cur.execute("""
            SELECT r.id, r.source, r.created_at,
                   count(d.id) AS boxes,
                   count(d.id) FILTER (WHERE EXISTS (
                       SELECT 1 FROM airports a
                       WHERE ST_Area(ST_Intersection(d.geom, a.geom)::geography)
                             >= 0.5 * ST_Area(d.geom::geography))) AS hits
            FROM runs r LEFT JOIN detections d ON d.run_id = r.id
            GROUP BY r.id ORDER BY r.id DESC LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    typer.echo(f"{'id':>4}  {'source':<34}{'boxes':>6}{'hit':>5}  created")
    for rid, source, created, boxes, hits in rows:
        typer.echo(f"{rid:>4}  {source:<34}{boxes:>6}{hits:>5}  {created.astimezone():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    app()
