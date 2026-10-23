Day 3 我們用 docker-compose 開了一個 PostGIS 容器，然後從 Day 8 開始訓練模型，它就一直躺在那裡沒人理。今天終於要把它接上：偵測結果寫進資料庫，空間運算交給 SQL。結果接上之後最大的收穫反而不是資料庫，而是`容器裡有一份完整的 PROJ`，Day 24 那支手寫的 UTM 逆投影終於有答案可以對了。

## 先把容器叫起來
```
docker compose up -d
level=warning msg="the attribute `version` is obsolete, it will be ignored"
 Container ithome_project Running
```

Docker Compose v2 已經不看 `version:` 這個欄位了，順手從 `docker-compose.yml` 刪掉。

Python 端要連 PostgreSQL 得用 `psycopg2`。經歷過 rasterio 跟 pyproj 被 Smart App Control 擋掉，我對任何帶 `.pyd` 的套件都有點緊張，`psycopg2-binary` 裡面也有一包編譯好的 `libpq`：

```
pip install psycopg2-binary
>>> import psycopg2
psycopg2 2.9.13 (dt dec pq3 ext lo64)
```

沒被擋。為什麼它可以、rasterio 不行，我沒有深究（可能是簽章，也可能是 Smart App Control 的信譽判定），反正能用就先用。

連上去看一下版本：

```
POSTGIS="3.3.4" PGSQL="150" GEOS="3.9.0-CAPI-1.16.2" PROJ="7.2.1"
```

看到最後那個 `PROJ="7.2.1"` 的時候我愣了一下。

## 被擋掉的 PROJ，在容器裡好好的
Day 24 我們會自己寫 UTM 逆投影，就是因為 pyproj 被 Smart App Control 擋掉。pyproj 底層就是 PROJ，而 PostGIS 的 `ST_Transform` 底層`也是 PROJ`。

差別在於 PostGIS 跑在 Docker 的 Linux 容器裡，`Windows 的程式碼完整性檢查管不到容器裡的 .so 檔`。

這代表 Day 24 那支 Snyder 級數解終於有了「標準答案」可以對。Day 24 驗證疊了三層（往返、數值積分、公開常數），但說到底都是我自己寫的程式在互相驗證。現在可以拿業界標準的 PROJ 來比了。

### 測試設計
```python
def _proj_lonlat(cur, xs, ys, epsg):
    cur.execute("""
        SELECT ST_X(g), ST_Y(g) FROM (
            SELECT ST_Transform(ST_SetSRID(ST_MakePoint(x, y), %s), 4326) AS g
            FROM unnest(%s::float8[], %s::float8[]) AS t(x, y)
        ) s
    """, (epsg, xs.tolist(), ys.tolist()))
    return np.array(cur.fetchall())
```

`unnest` 把兩個陣列攤成一張表，一次查詢就能轉 500 個點，不用跑 500 次 round trip。

測三種範圍：

1. **影像範圍內**：桃園那張 GeoTIFF 上隨機 500 個像素，這是實際會用到的區域
2. **分帶內**：中央經線 ±3°、赤道到 60°N，UTM 設計上的有效範圍
3. **分帶外**：±3° 到 ±6°，UTM 本來就不該這樣用，看看會壞多少

第 2、3 組的測試點是`先用 PROJ 從經緯度正投影到 UTM`，再拿我們的逆投影轉回來比，整個過程測試點都不經過我們的程式。

### 結果
```
[1] geo_utils 對 PROJ
  影像範圍內（桃園）       500 點  最大   0.0602 mm  中位數 0.0600 mm
  分帶內 ±3°, 0~60°N      500 點  最大   0.3597 mm  中位數 0.0795 mm
  分帶外 ±3~6°, 0~60°N    500 點  最大  46.0634 mm  中位數 4.0228 mm
  Web Mercator 全球        500 點  最大 0.000004 mm
```

在桃園這張圖上，我們跟 PROJ 差 `0.06 mm`。Sentinel-2 一個像素是 10 公尺，0.06 mm 是一個像素的`十六萬分之一`。

### 一個差點寫錯的結論
我第一次跑這個測試的時候，第 2 組不是照經緯度抽樣，而是在固定的東距範圍（170 ~ 830 km）裡抽，結果最大誤差跑出 `31.8 mm`。我差點就要在文章裡寫「Day 24 說誤差小於 1 mm 是錯的」。

後來把誤差最大的點一個個印出來：

```
E=  500km N= 2770km  lat= 25.05 dlon= +0.00  err=   0.0597 mm
E=  170km N= 2770km  lat= 25.01 dlon= -3.27  err=   0.3520 mm
E=  170km N= 5500km  lat= 49.56 dlon= -4.56  err=   6.7305 mm
E=  170km N= 6650km  lat= 59.86 dlon= -5.90  err=  41.0312 mm
```

看 `dlon` 那一欄就懂了。UTM 的分帶是用`經度`劃的（每帶 6°），不是用東距。高緯度的經線比較擠，同樣是東距 170 km，在 25°N 是中央經線外 3.27°，到了 60°N 已經是 5.90°，`早就跑出分帶了`。

改成照經緯度抽樣之後，分帶內最大誤差 0.36 mm，Day 24 的說法成立。分帶外會壞到 46 mm，這也正是 UTM 要切成 60 個分帶的原因：橫麥卡托級數展開的誤差`隨著離中央經線的距離快速發散`。

這次的教訓跟 Day 23 一樣：`測試的抽樣方式本身也會出錯`，數字很難看的時候先懷疑量法。

## 建表
三張表：

```sql
CREATE TABLE airports (
    iata      text PRIMARY KEY,
    name_en   text,
    geom      geometry(MultiPolygon, 4326) NOT NULL
);

CREATE TABLE runs (
    id          serial PRIMARY KEY,
    source      text NOT NULL,
    model       text NOT NULL,
    params      jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    footprint   geometry(Polygon, 4326) NOT NULL
);

CREATE TABLE detections (
    id      serial PRIMARY KEY,
    run_id  integer NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    score   real NOT NULL,
    props   jsonb,
    geom    geometry(Polygon, 4326) NOT NULL
);
CREATE INDEX detections_geom_idx ON detections USING GIST (geom);
```

幾個設計決定：

- **`runs.footprint`**：影像四個角的經緯度。沒有它就只能問「框到了什麼」，問不了「這張圖裡`明明有`哪座機場卻沒框到」，也就是漏判
- **`props jsonb`**：長寬比、碎片數這些屬性每天都在變，塞進 jsonb，不用每加一個欄位就改 schema
- **`ON DELETE CASCADE`**：同一張圖重跑的時候，刪掉舊的 run，底下的框跟著消失
- **GIST 索引**：空間查詢的 B-tree，沒有它每個 `ST_Intersects` 都是全表掃描

`PipelineResult` 加了一個 `footprint()` 方法，把影像四角換成經緯度：

```python
def footprint(self):
    h, w = self.image.shape[:2]
    lon, lat = pixel_to_lonlat(self.georef, [0, w, w, 0], [0, 0, h, h])
    return [[float(a), float(b)] for a, b in zip(lon, lat)]
```

### 寫入：讓資料庫自己解析 GeoJSON
幾何怎麼傳進 SQL？可以在 Python 組 WKT 字串 `POLYGON((121.24 25.09, ...))`，但 WKT 是 `x y`（空白分隔）、GeoJSON 是 `[x, y]`，手動轉換很容易把經緯度順序搞反。

直接把 GeoJSON 文字丟給 `ST_GeomFromGeoJSON`，讓資料庫解析：

```python
execute_values(cur, """
    INSERT INTO detections (run_id, score, ..., geom) VALUES %s
""", rows, template="(%s, %s, ..., ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))")
```

`execute_values` 會把多筆資料合成一條 INSERT，比逐筆插入快很多。

## 用 SQL 重算 Day 25 的指標
Day 25 用 shapely 算的 `inside_frac` 跟 `coverage`，換成 SQL：

```sql
SELECT d.id, d.score, a.iata,
       ST_Area(ST_Intersection(d.geom, a.geom)::geography)
           / ST_Area(d.geom::geography)  AS inside_frac,
       ST_Area(ST_Intersection(d.geom, a.geom)::geography)
           / ST_Area(a.geom::geography)  AS coverage
FROM detections d
LEFT JOIN LATERAL (
    SELECT a.* FROM airports a
    ORDER BY ST_Area(ST_Intersection(d.geom, a.geom)::geography) DESC,
             d.geom <-> a.geom
    LIMIT 1
) a ON true
WHERE d.run_id = %s
```

兩個重點：

**`::geography`**。`geometry(4326)` 的單位是`度`，直接 `ST_Area` 會吐出「平方度」這種沒有意義的數字。轉成 `geography` 之後 PostGIS 會在橢球面上算，單位是平方公尺。這跟 Day 25 在 Python 裡先搬到 UTM 再算是同一件事，只是 PostGIS 幫你做了。

**`LATERAL JOIN`**。每個框只配一座機場：重疊面積最大的那座；完全沒碰到就配最近的（`<->` 是 PostGIS 的 KNN 距離運算子，會用到 GIST 索引）。

```
[3] SQL 重算 Day 25 的指標（括號內是 shapely 的結果）
  det#1 score=0.507 -> TPE
    inside_frac 45.84%  (45.84%)
    coverage    17.68%  (17.68%)
    box_km2     4.320  (4.320)
```

`小數點後兩位完全一致`。一邊是 shapely + 我們手寫的 UTM 投影，一邊是 PostGIS 的 geography（橢球面計算），兩條完全不同的路得到同一個答案。

### Day 24 的數字也重算一次
```sql
SELECT ST_Contains(d.geom, p),
       ST_Distance(d.geom::geography, p::geography),
       ST_Distance(ST_Centroid(d.geom)::geography, p::geography)
FROM detections d, ST_SetSRID(ST_MakePoint(121.2328, 25.0767), 4326) AS p
```

```
ARP 在框內? False  到框邊 907 m  到框中心 1,455 m
```

跟 Day 24 用 haversine 算的一模一樣。

### 最近的機場（KNN）
```sql
SELECT a.iata, ST_Centroid(d.geom)::geography <-> a.geom::geography
FROM detections d, airports a
ORDER BY ST_Centroid(d.geom) <-> a.geom LIMIT 3
```

```
TPE  49 m
TSA  31,327 m
HSZ  39,741 m
```

框的中心點離桃園機場圍籬 `49 公尺`，也就是框的中心`在圍籬外面`。這跟 Day 25 看到的「框有一半在機場外」對得上。

### 漏判查詢
最後一個是 Day 27 會用到的查詢：影像範圍內有哪些機場沒被任何框碰到？

```sql
SELECT a.iata, a.name_en
FROM runs r
JOIN airports a ON ST_Within(a.geom, r.footprint)
WHERE r.id = %s
  AND NOT EXISTS (
      SELECT 1 FROM detections d
      WHERE d.run_id = r.id AND ST_Intersects(d.geom, a.geom))
```

```
[4] 這張圖裡有哪些機場沒被框到
   沒有
```

桃園機場有被碰到（雖然只碰到一半），所以不算漏判。這張圖裡也只有這一座機場，`一張圖、一座機場、一個框`，這個查詢在這裡沒什麼用。

這正好點出目前最大的問題：`我們從頭到尾只有一張圖`。

## 小結
今天把 PostGIS 接上了：

1. `psycopg2-binary` 沒有被 Smart App Control 擋
2. PostGIS 容器裡有完整的 `PROJ 7.2.1`，拿來跟 Day 24 的手寫投影對答案：桃園範圍差 `0.06 mm`，分帶內最大 0.36 mm
3. 差點因為`抽樣方式錯誤`寫出「Day 24 是錯的」，印出誤差最大的點才發現測試點跑出了分帶
4. 三張表 `airports / runs / detections`，用 `::geography` 算面積，SQL 跟 shapely 的結果完全一致

但寫查詢寫到最後發現，資料庫準備好要回答「哪些機場沒被框到」「誤判都在哪裡」這類問題，`我們卻只有一張圖可以問`。

原本計畫明天要用 Copernicus Data Space 下載更多 Sentinel-2 影像，但要註冊帳號、處理 OAuth、下載整個 100 km 見方的 tile 再自己裁。今天在 Day 25 的底圖清單裡其實已經看到一個更簡單的來源，明天就用它來做`這個系列第一次真正的泛化測試`，那我們明天見。
