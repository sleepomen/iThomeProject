"""
Day 27: 第一次拿`沒看過的場景`測模型。

13 張 Sentinel-2 cloudless 場景（8 座機場 + 5 個困難負樣本），
每張跑 Day 23 的管線、用 OSM 圍籬打分數、寫進 Day 26 的 PostGIS，
最後輸出總表與一張縮圖總覽。

--bands 決定要不要模仿訓練資料的長相，Day 28 會拿來做對照實驗。
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import postgis  # noqa: E402
from inference.evaluate import load_airports, score_feature  # noqa: E402
from inference.geo_utils import detections_to_geojson, lonlat_to_pixel  # noqa: E402
from inference.pipeline import AirportDetector  # noqa: E402

SCENES = ROOT / "data" / "s2cloudless"
OUT = ROOT / "runs" / "predict" / "day27_multi"
MODEL = "airport_obb_mixed"

BLUE, GREEN, RED = (219, 152, 52), (113, 204, 46), (60, 76, 231)


def draw_scene(result, fc, airports, title):
    img = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
    for ap in airports:
        polys = ap["geom"].geoms if ap["geom"].geom_type == "MultiPolygon" else [ap["geom"]]
        for poly in polys:
            ring = np.asarray(poly.exterior.coords)
            col, row = lonlat_to_pixel(result.georef, ring[:, 0], ring[:, 1])
            pts = np.stack([col, row], -1).round().astype(np.int32)
            cv2.polylines(img, [pts], True, BLUE, 2, cv2.LINE_AA)
    for det, f in zip(result.detections, fc["features"]):
        color = GREEN if f["properties"]["hit"] else RED
        cv2.polylines(img, [np.asarray(det.points, np.int32)], True, color, 3, cv2.LINE_AA)
    img = cv2.resize(img, (600, 400), interpolation=cv2.INTER_AREA)
    bar = np.zeros((34, 600, 3), np.uint8)
    cv2.putText(bar, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def contact_sheet(tiles, cols=4):
    blank = np.full_like(tiles[0], 255)
    tiles = tiles + [blank] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def run(detector, tag, airports, write_db=True, scenes=SCENES):
    manifest = json.loads((scenes / "manifest.json").read_text(encoding="utf-8"))
    rows, tiles, all_features = [], [], []

    for sc in manifest:
        tif = scenes / sc["file"]
        result = detector.detect(tif)
        fc = detections_to_geojson(result.detections, result.georef,
                                   source=tif.name, model=MODEL, tag=tag)
        for f in fc["features"]:
            f["properties"].update(score_feature(f, airports))
        all_features += fc["features"]

        hits = sum(f["properties"]["hit"] for f in fc["features"])
        fps = len(fc["features"]) - hits
        expected = sc["airports"]
        found = any(f["properties"]["hit"] and f["properties"]["iata"] in expected
                    for f in fc["features"])
        rows.append({"scene": sc["scene"], "desc": sc["desc"], "airports": expected,
                     "raw": result.raw_count, "boxes": len(fc["features"]),
                     "hit": hits, "fp": fps, "found": found if expected else None,
                     "best_inside": max((f["properties"]["inside_frac"]
                                         for f in fc["features"]), default=0.0)})

        in_view = [a for a in airports if a["iata"] in expected]
        label = "found" if found else ("MISS" if expected else "neg")
        tiles.append(draw_scene(result, fc, in_view,
                                f"{sc['scene']}  {label}  box={len(fc['features'])} fp={fps}"))

        if write_db:
            with postgis.connect() as cur:
                postgis.delete_runs(cur, f"{tif.name}#{tag}", MODEL)
                postgis.insert_run(cur, f"{tif.name}#{tag}", MODEL, result.footprint(), fc,
                                   params={"bands": detector.bands, "stretch": detector.stretch,
                                           "tag": tag})
    return rows, tiles, all_features


def print_table(rows):
    print(f"{'scene':<18}{'機場':<7}{'raw':>5}{'boxes':>7}{'hit':>5}{'fp':>5}"
          f"{'best_inside':>13}  結果")
    for r in rows:
        verdict = ("找到" if r["found"] else "漏判") if r["airports"] else \
            ("乾淨" if r["boxes"] == 0 else "誤判")
        print(f"{r['scene']:<18}{','.join(r['airports']) or '-':<7}{r['raw']:>5}{r['boxes']:>7}"
              f"{r['hit']:>5}{r['fp']:>5}{r['best_inside']:>13.1%}  {verdict}")

    pos = [r for r in rows if r["airports"]]
    neg = [r for r in rows if not r["airports"]]
    total_boxes = sum(r["boxes"] for r in rows)
    total_hit = sum(r["hit"] for r in rows)
    print(f"\n機場召回率  {sum(r['found'] for r in pos)}/{len(pos)}")
    print(f"框的精確率  {total_hit}/{total_boxes}"
          f" = {total_hit / total_boxes:.1%}" if total_boxes else "框的精確率  沒有任何框")
    print(f"負樣本誤判  {sum(r['boxes'] > 0 for r in neg)}/{len(neg)} 張有框，"
          f"共 {sum(r['boxes'] for r in neg)} 個")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rgb")
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    airports = load_airports()
    # EOX 的 tif 只有 3 個波段（R, G, B），(0, 1, 2) 在這裡就是物理正確的真彩色
    detector = AirportDetector(bands=(0, 1, 2), stretch="max", confidence=args.confidence)
    rows, tiles, feats = run(detector, args.tag, airports, write_db=not args.no_db)

    print_table(rows)
    cv2.imwrite(str(OUT / f"day27_{args.tag}_sheet.jpg"), contact_sheet(tiles),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    (OUT / f"day27_{args.tag}.json").write_text(
        json.dumps({"rows": rows, "features": feats}, ensure_ascii=False, indent=2,
                   default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
