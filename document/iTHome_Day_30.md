今天是這個系列最後一個實作日。昨天把 Day 23 到 Day 28 包成了 `python -m inference.cli`，今天就用它從頭到尾跑一次：挑一座`這個系列從來沒出現過的機場`，從抓衛星圖開始，一路到地圖和資料庫。

## 挑一座新機場
Day 27 測了 8 座機場，台灣本島有 IATA 代碼的機場還剩幾座沒用過。我挑了`恆春機場（HCN）`：

- 在台灣最南端，跟訓練用的桃園直線距離約 340 公里
- 周圍有海岸、山坡、農田，是 Day 27 確認過模型會誤判的地形
- 跑道只有一條，比桃園小很多，尺度也不一樣

OSM 上有它的圍籬多邊形，Day 25 抓的 `tw_airports.geojson` 裡面就有：

```
HCN 恆春機場 [120.730483, 22.038185]
```

## 第零步：資料庫
PostGIS 容器要先開著：

```
docker compose up -d
 Container ithome_project Running
```

建表、灌入 OSM 機場圍籬（重複執行沒關係，`CREATE TABLE IF NOT EXISTS` 加上 `ON CONFLICT DO UPDATE`）：

```
python -m inference.cli db-init
schema OK，airports 13 筆
```

## 第一步：抓圖
```
python -m inference.cli fetch 120.7305 22.0382 --name hcn
data\s2cloudless\hcn.tif  1201x801 px
```

這一行背後發生的事：

1. `bbox_around` 以恆春機場為中心算出 12 km x 8 km 的經緯度範圍（Day 27）
2. `fetch_mosaic` 算出要哪些第 14 層的圖磚，下載、拼接、裁切（Day 25）
3. `resample_to_ground` 從 8.7 m/px 縮到 10 m/px，仿射矩陣跟著改（Day 27）
4. `write_geotiff` 組 GeoKeyDirectory，寫成 EPSG:3857 的 GeoTIFF（Day 27）

## 第二步：偵測、評分、地圖、資料庫
把恆春跟原始的桃園 tif 一起丟進去：

```
python -m inference.cli detect data/s2cloudless/hcn.tif data/raw/som_san_air.tif \
    --map --db -o runs/predict/day30_demo
```

```
file                      raw  boxes  hit    sec  run_id
hcn.tif                     7      5    1    2.6  42
som_san_air.tif             2      1    0    0.6  43

輸出 -> runs\predict\day30_demo

real    0m4.149s
```

兩張圖總共 `4.1 秒`。第一張的 2.6 秒裡包含載入模型、把權重搬上 GPU，第二張只要 0.6 秒。

這一行背後發生的事：

1. 讀 GeoTIFF、選波段、拉伸、CLAHE（Day 18、Day 23）
2. SAHI 切片推論（Day 18 ~ 19）
3. 幾何合併碎框（Day 21）
4. 像素轉經緯度、輸出 GeoJSON（Day 24）
5. 用 OSM 機場圍籬打分數（Day 25）
6. 輸出 folium 互動地圖（Day 25）
7. 寫入 PostGIS（Day 26）

輸出資料夾：

```
hcn.geojson           6,581 bytes
hcn.html             86,235 bytes
som_san_air.geojson   1,372 bytes
som_san_air.html     72,219 bytes
```

HTML 比 Day 25 的大，因為 CLI 版會把 13 座機場的 OSM 圍籬都放進地圖裡，拖到哪都看得到。

### 先驗一下：桃園的結果有沒有變
`som_san_air.tif` 是 Day 24 跑過的圖，CLI 的結果應該要一模一樣：

```
CLI 輸出   score 0.5067  center [121.228112, 25.089122]  inside_frac 0.4584
Day 24     score 0.5067  center [121.228112, 25.089122]
```

`分數、中心座標到小數點後六位都一樣`，Day 25 的 inside_frac 45.84% 也一樣。CLI 只是重新包裝，沒有改變任何行為。這種「重構前後輸出不變」的檢查，Day 23 換掉 rasterio 的時候也做過一次，這次又派上用場。

## 第三步：看結果
```
python -m inference.cli runs --limit 2

  id  source            boxes  hit  created
  43  som_san_air.tif       1    0  2026-10-23 23:45
  42  hcn.tif               5    1  2026-10-23 23:45
```

恆春 5 個框、命中 1 個。打開 GeoJSON 看每個框的屬性：

```
score=0.701  hit=False  在機場內  0.0%  3.45 km²  距機場   303 m
score=0.519  hit=False  在機場內  0.0%  0.23 km²  距機場 1,691 m
score=0.382  hit=False  在機場內  0.0%  1.86 km²  距機場 1,729 m
score=0.322  hit=True   在機場內 93.5%  0.25 km²  距機場     0 m
score=0.288  hit=False  在機場內  0.0%  0.15 km²  距機場 2,203 m
```

畫到 Esri 的高解析底圖上：

![day30 hcn](runs/predict/day30_demo/day30_hcn.jpg)

`藍色`是 OSM 的恆春機場圍籬，`綠色`是命中的框，`紅色`是誤判。

- **綠框（0.32）**：在跑道北段上，93.5% 落在圍籬內。模型`確實在一座從沒看過的機場上找到了跑道`，雖然只框到一小段
- **最大的紅框（0.70）**：機場東邊的山坡，3.45 km²。這是`分數最高`的框，離機場圍籬只有 303 公尺，但裡面一點機場都沒有
- **西北的紅框（0.38）**：一半在海上、一半在海岸
- **北邊兩個小紅框**：農田和山坡上的空地

跟 Day 28 的分析完全一致：

- 分數最高的是誤判（0.701），命中的那個只有 0.322，`信心分數跟正確與否無關`
- 命中的框只框到機場的一小塊，跟訓練標註「只框機場碎片」的習慣一樣
- 誤判落在大片的、邊界明顯的區域上

互動地圖 `runs/predict/day30_demo/hcn.html` 打開可以點每個框看屬性，切到 Sentinel-2 cloudless 底圖就是模型實際看到的畫面。

## 資料庫裡累積了什麼
從 Day 26 開始，每次偵測都寫進 PostGIS。到今天為止：

```sql
SELECT r.params->>'tag' AS tag, count(*) AS boxes,
       count(*) FILTER (WHERE hit) AS hits
FROM runs r JOIN detections d ON d.run_id = r.id
GROUP BY tag
```

```
實驗          框   命中    精確率
matched       27      4    14.8%
rgb           80     10    12.5%

資料庫裡共 29 次偵測、114 個框
```

`rgb` 是 Day 27 的原始測試，`matched` 是 Day 28 的色彩對齊實驗。這些數字不用再翻 log 或重跑腳本，一條 SQL 就撈得出來。

如果之後重新訓練了模型，只要對同樣的 13 張圖跑一次 `detect --db`，就能在資料庫裡直接跟今天的結果比較。`這條管線最大的價值，是讓「下一個模型」有一把固定的尺可以量`。

## 回頭看這條管線
把今天跑過的流程畫成一條線：

```
經緯度
  │  fetch：圖磚 → 拼接 → 重採樣 → GeoTIFF
  ▼
GeoTIFF (EPSG:3857 或 UTM)
  │  detect：波段 → 拉伸 → CLAHE → SAHI → 碎框合併
  ▼
Detection（像素座標）
  │  pixel_to_lonlat（UTM 級數解 / Web Mercator 封閉解）
  ▼
GeoJSON（WGS84）
  ├──► OSM 圍籬評分（inside_frac / coverage）
  ├──► folium 地圖
  └──► PostGIS（runs / detections / airports）
```

每一層都有對應的驗證：

| 層 | 驗證方式 | 哪天 |
|---|---|---|
| 讀 GeoTIFF | 跟 rasterio 逐位元比對 | Day 23 |
| UTM 投影 | 往返、數值積分、公開常數、PROJ | Day 24、26 |
| Web Mercator | PROJ，誤差 0.000004 mm | Day 26 |
| 寫 GeoTIFF | 讀回來的中心對上 OSM 形心 | Day 27 |
| 面積指標 | shapely 跟 PostGIS geography 結果一致 | Day 26 |
| CLI | 跟 Day 24 輸出逐位比對 | 今天 |

工程這一側，每一層都能信任。

模型那一側，Day 27 和 Day 28 已經說得很清楚了：`它記住了一座機場的幾個碎片，沒有學會什麼是機場`。

## 小結
今天用 CLI 跑完了整條管線：

1. `db-init` → `fetch` → `detect --map --db` → `runs`，四個指令從經緯度跑到資料庫
2. 兩張圖 `4.1 秒`，CLI 對桃園的輸出跟 Day 24 `逐位一致`
3. 恆春機場：5 個框、命中 1 個。模型在一座從沒看過的機場上找到了一段跑道，但分數最高的框（0.701）是山坡上的誤判
4. PostGIS 累積了 29 次偵測、114 個框，Day 27 和 Day 28 的實驗結果一條 SQL 就能比較

這條管線現在能做的事是：`給任何一塊地方的經緯度，4 秒內告訴你模型認為機場在哪，以及它錯得多離譜`。後半句目前比前半句更有用。

明天是最後一天。要把 Day 1 畫的系統架構圖拿出來，一格一格對照這 30 天做了什麼、沒做什麼、哪些做了但發現是錯的，那我們明天見。
