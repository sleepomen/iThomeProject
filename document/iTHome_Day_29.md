Day 28 的結論是：模型要修，得從資料重新來過。這個系列剩下的時間不夠重訓，但可以先把`重訓之後還會用到的東西`整理好。今天要把 Day 23 到 Day 28 散在各個腳本裡的步驟，包成一個命令列工具。

## 為什麼需要 CLI
先看一下 `Tests/` 資料夾現在長怎樣：

```
test_day23_pipeline.py
test_day24_detect_geo.py
test_day25_map.py
test_day26_postgis.py
test_day27_multi_scene.py
test_day28_error_analysis.py
```

每一支都是「那天想驗證什麼就寫什麼」，路徑寫死在檔案開頭：

```python
TIF = ROOT / "data" / "raw" / "som_san_air.tif"
OUT = ROOT / "runs" / "predict" / "day24_geo"
```

想對新的圖跑一次偵測，要先打開檔案改路徑，然後還得記得：要輸出 GeoJSON、要用 OSM 打分數、要畫地圖、要寫資料庫，`這四件事散在四支不同的腳本裡`。

CLI 的價值不在省打字，而是`把「一次偵測」該產出的東西固定下來`，每次都一樣，不會漏。

## 目標
```
python -m inference.cli fetch 121.55 25.07 --name tsa
python -m inference.cli detect data/s2cloudless/tsa.tif --map --db
python -m inference.cli runs
python -m inference.cli db-init
```

四個指令：

| 指令 | 做什麼 | 對應哪天 |
|---|---|---|
| `fetch` | 從 EOX 抓一塊範圍，存成 10 m 的 GeoTIFF | Day 27 |
| `detect` | 偵測 + GeoJSON + OSM 評分 + 地圖 + 資料庫 | Day 23 ~ 26 |
| `db-init` | 建表、灌入 OSM 機場圍籬 | Day 26 |
| `runs` | 列出資料庫裡最近的偵測紀錄 | Day 26 |

## 選 typer
Python 寫 CLI 有三個常見的選擇：

- `argparse`：標準庫，Day 27 的 `fetch_s2_scenes.py` 就是用它，但參數一多就很囉嗦
- `click`：用 decorator 定義參數
- `typer`：建在 click 之上，`用 type hint 定義參數`

環境裡 typer 已經在了（Day 8 下載資料集用的 roboflow 套件依賴它），選它。它的寫法是把函式簽名直接變成 CLI：

```python
@app.command()
def detect(
    tifs: list[Path] = typer.Argument(..., exists=True, dir_okay=False,
                                      help="一或多張 GeoTIFF"),
    out: Path = typer.Option(DEFAULT_OUT, "--out", "-o", help="輸出資料夾"),
    confidence: float = typer.Option(0.25, "--conf", min=0.0, max=1.0),
    bands: str = typer.Option("0,1,2", help="拿哪三個波段當 RGB"),
    score: bool = typer.Option(True, help="用 OSM 機場圍籬替每個框打分數"),
    make_map: bool = typer.Option(False, "--map", help="輸出 folium 互動地圖"),
    db: bool = typer.Option(False, "--db", help="寫入 PostGIS"),
    ...
):
```

幾個 typer 幫忙處理掉的事：

- `list[Path]` 自動變成「可以接多個檔案」
- `exists=True` 會在執行前先檢查檔案存不存在，不存在直接報錯，不會跑到一半才炸
- `min=0.0, max=1.0` 會擋掉 `--conf 25` 這種手殘
- `bool` 選項自動產生 `--score / --no-score` 一對開關

```
 Usage: python -m inference.cli detect [OPTIONS] TIFS...

┌─ Arguments ─────────────────────────────────────────────────┐
│ *    tifs      TIFS...  一或多張 GeoTIFF [required]         │
└─────────────────────────────────────────────────────────────┘
┌─ Options ───────────────────────────────────────────────────┐
│ --out      -o           PATH          輸出資料夾            │
│ --conf                  FLOAT RANGE   [default: 0.25]       │
│                         [0.0<=x<=1.0]                       │
│ --bands                 TEXT          拿哪三個波段當 RGB    │
│ --score    --no-score                 用 OSM 機場圍籬打分數 │
│ --map                                 輸出 folium 互動地圖  │
│ --db                                  寫入 PostGIS          │
└─────────────────────────────────────────────────────────────┘
```

`--help` 的說明直接來自 `help=` 參數跟函式的 docstring，寫程式的同時文件就寫好了。

## detect：一個指令做完四件事
核心邏輯其實就是把前幾天的函式依序呼叫：

```python
for tif in tifs:
    result = detector.detect(tif)                          # Day 23
    fc = detections_to_geojson(result.detections,          # Day 24
                               result.georef, ...)
    for f in fc["features"]:
        f["properties"].update(score_feature(f, airports)) # Day 25

    stem.with_suffix(".geojson").write_text(json.dumps(fc))

    if make_map:
        save_map(build_map(fc, airports=airports, ...),    # Day 25
                 stem.with_suffix(".html"))
    if db:
        with postgis.connect() as cur:                     # Day 26
            postgis.init_schema(cur)
            run_id = postgis.insert_run(cur, tif.name, ...)
```

能這麼短，是因為前幾天一直在做同一件事：`把邏輯從 Tests/ 搬回 inference/`。Day 23 把讀圖和畫框搬出來，Day 24 把座標轉換搬出來，Day 25 把評分和地圖搬出來。CLI 只是最後一層薄薄的膠水。

今天搬家的是 `resample_to_ground`。它原本寫在 Day 27 的 `fetch_s2_scenes.py` 裡，但 `fetch` 指令也要用，所以搬進 `inference/tiles.py`，讓下載腳本跟 CLI 共用同一份。`同一段邏輯只能有一個家`，不然改了一邊忘了另一邊，就會出現兩個版本的「10 m」。

### 只 import 用得到的東西
```python
@app.command()
def detect(...):
    from inference.evaluate import load_airports, score_feature
    from inference.pipeline import AirportDetector
    ...
    if db:
        from inference import postgis
```

import 寫在函式裡面，而不是檔案開頭。原本以為這樣能讓 `--help` 快很多，實際量了一下：

```
import inference.pipeline        0.12 s
import torch, ultralytics        0.88 s
```

`import inference.pipeline` 只要 0.12 秒，因為 torch 根本不會在這時候載入。Day 23 寫 `AirportDetector` 的時候就把模型做成延遲載入（第一次呼叫 `.model` 才建立），sahi 也是到那時才去 import torch。省下的時間很少。

但 import 寫在函式裡還是有用，只是理由不一樣：`沒加 --db 就不需要 psycopg2`。如果哪天換一台沒裝 PostgreSQL 驅動的電腦，`detect` 還是能跑；`fetch` 跟 `runs` 也不需要偵測相關的套件。每個指令只依賴它自己用得到的東西。

## runs：從資料庫看歷史紀錄
```python
@app.command()
def runs(limit: int = typer.Option(20)):
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
```

`FILTER (WHERE ...)` 是 PostgreSQL 的聚合過濾語法，一條 SQL 同時算出「總共幾個框」跟「命中幾個」，不用跑兩次查詢。命中的定義跟 Day 25 一樣：框有一半以上面積在機場圍籬內。

```
  id  source                          boxes  hit  created
  41  neg_changhua.tif#matched            4    0  2026-10-23 23:43
  40  neg_mailiao.tif#matched             4    0  2026-10-23 23:43
  39  neg_taichungport.tif#matched        4    0  2026-10-23 23:43
```

### 時區的小坑
第一次跑的時候，時間顯示的是 `15:43`，但我明明是晚上 11 點多在跑。

`created_at` 是 `timestamptz`，PostgreSQL 內部存 UTC，取出來的時候換算成`連線的時區`。Docker 容器的預設時區是 UTC，所以拿到的是 UTC 時間。psycopg2 回傳的是帶時區資訊的 `datetime`，在 Python 端轉成本機時區就好：

```python
f"{created.astimezone():%Y-%m-%d %H:%M}"
```

`.astimezone()` 不帶參數的意思是「轉成本機時區」。

## Windows 終端機的編碼
Day 25 跑 OSM 下載腳本的時候其實遇過一次：

```
CYI  RCKU  Chiayi Airport  (�x��)
```

`(軍用)` 變成了亂碼。Windows 終端機預設的編碼是 `cp950`（Big5），Python 輸出 UTF-8 的中文就會亂掉。

最簡單的解法是設環境變數：

```
set PYTHONIOENCODING=utf-8
```

typer 的 `--help` 用了 rich 來畫框線，rich 自己會處理終端機編碼，所以 CLI 的說明文字沒問題，但我們 `typer.echo` 印的表格還是要靠這個環境變數。

## 跑一次看看
```
python -m inference.cli db-init
schema OK，airports 13 筆
```

```
python -m inference.cli fetch 120.7305 22.0382 --name hcn
data\s2cloudless\hcn.tif  1201x801 px
```

`detect` 跟完整的端到端流程，留到明天的演示一次跑完。

## 小結
今天把前六天的成果包成了 `python -m inference.cli`：

1. 四個指令 `fetch / detect / db-init / runs`，對應 Day 23 ~ Day 28 的功能
2. 用 `typer` 從 type hint 生成參數和說明文件，檔案存在性、數值範圍都在執行前檢查
3. CLI 本身很薄，因為前幾天一直在`把邏輯從 Tests/ 搬回 inference/`，今天再把 `resample_to_ground` 搬進 `tiles.py`
4. import 寫在函式內，原本以為能加速 `--help`，量了才發現 torch 本來就是延遲載入；真正的好處是`每個指令只依賴自己用得到的套件`
5. 兩個 Windows 小坑：PostgreSQL 的 `timestamptz` 預設以 UTC 顯示，終端機的 `cp950` 編碼會讓中文變亂碼

明天是這個系列的最後一個實作日。要用今天的 CLI，從一個`系列裡從沒出現過的機場`開始，一路跑完 抓圖 → 偵測 → 評分 → 地圖 → 資料庫，那我們明天見。
