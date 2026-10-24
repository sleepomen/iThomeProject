最後一天。Day 1 畫了一張系統架構圖，說好最後要拿出來對照，今天就一格一格看：哪些做完了、哪些刻意沒做、哪些做了但後來發現是錯的。

## Day 1 的架構圖
先把它原封不動貼回來：

```
[ 資料源 ]
   |-- TASA 開源衛星影像
   v
[ AI 模型與空間處理 ]
   |-- YOLO + PyTorch (影像物件偵測 -> 輸出像素座標)
   |-- Rasterio (像素座標轉換 -> 輸出真實世界經緯度 WGS84)
   v
[ 後端與資料庫 ]
   |-- PostgreSQL + PostGIS (儲存空間幾何資料)
   |-- FastAPI (封裝 AI 模型與資料庫，提供 RESTful API)
   v
[ 前端 Data Board ]
   |-- React (UI 介面與狀態管理)
   |-- Leaflet.js + OpenStreetMap (底圖渲染與空間資料視覺化)

以上環境皆運行於 [ Docker ] 容器中
```

## 逐格對照
| Day 1 的規劃 | 最後的樣子 | 狀態 |
|---|---|---|
| TASA 開源衛星影像 | Day 2 發現要申請，改用 Sentinel-2；Day 27 再改用 EOX Sentinel-2 cloudless 圖磚，免帳號 | 換掉了 |
| YOLO + PyTorch | YOLOv8 → OBB 傾斜框 → SAHI 切片推論 → 混合資料集（Day 10 ~ 22） | 做完 |
| Rasterio 像素轉經緯度 | 被 Smart App Control 擋掉，改用 `tifffile` 讀檔、自己寫 UTM 級數解（Day 23 ~ 24） | 換掉了 |
| PostgreSQL + PostGIS | `runs / detections / airports` 三張表，OSM 機場圍籬當地面真相（Day 26） | 做完 |
| FastAPI | 改成 `python -m inference.cli`（Day 29） | 刻意不做 |
| React | 沒有 | 刻意不做 |
| Leaflet.js + OpenStreetMap | `folium`（Leaflet 的 Python 包裝）輸出互動地圖（Day 25） | 做完 |
| Docker | 只有 PostGIS 跑在容器裡，Python 在本機跑 GPU | 部分 |

另外 Day 1 也寫了想做「人口活動分析、異常飛行物偵測、災害影響範圍預測」，這三個`一個都沒碰到`。Day 8 就發現 Sentinel-2 的 10 m 解析度下一架客機只有 3 ~ 4 個像素，目標只能縮小到「機場」這種大物件。

### 換掉的兩格
兩次都不是自己想換，是被環境逼的：

- **TASA → Sentinel-2**：衛星影像要申請（Day 2）
- **Rasterio → tifffile + 手寫投影**：Windows 的 Smart App Control 擋掉 GDAL 和 PROJ 的 DLL（Day 23、24）

但第二次換掉反而是好事。自己寫投影逼著我去驗證，往返、數值積分、公開常數一路驗下來，Day 26 有了容器裡的 PROJ 可以對答案，桃園範圍內差 `0.06 mm`。如果一開始 rasterio 就能用，我大概永遠不會知道 `transform * (col, row)` 背後在算什麼。

### 刻意不做的兩格
FastAPI 跟 React 是 Day 22 的時候決定砍掉的。剩下的天數只夠二選一：開一個新的技術棧，或是把已經有的東西驗證到能信任。我選了後者。

現在回頭看是對的。如果 Day 23 直接去寫 API，就不會發現訓練集洩漏；如果 Day 24 直接把框丟上前端，就不會把框換算成公尺，也就不會發現 ARP 根本不在框裡。`一個漂亮的前端會把錯的結果包裝得很像對的`。

而且 CLI 已經把介面定好了：`detect` 吃 GeoTIFF、吐 GeoJSON、寫進 PostGIS。之後要包 FastAPI，基本上就是把 typer 的 decorator 換成 `@app.post`。

## 做了但發現是錯的
這是我覺得這個系列最有價值的部分。

| 哪天 | 當時以為 | 後來發現 | 哪天發現 |
|---|---|---|---|
| Day 10 ~ 22 | mAP 在進步 | train / valid / test 全是同一座機場，量的是記憶 | Day 23 |
| Day 18 ~ 22 | 用波段 0, 1, 2 當 RGB | 波段 0 是 B1 海岸氣膠，不是紅光 | Day 23 |
| Day 21 ~ 22 | 把框從長寬比 2.52 推到 5.05 是進步 | 類別是 `airport` 不是 `runway`，標註中位數長寬比只有 1.86 | Day 24 |
| Day 22 | 框準確落在跑道上 | 框只有 45.8% 在機場圍籬內，ARP 離框 907 m | Day 24 ~ 25 |
| Day 17 ~ 22 | 問題在模型和後處理 | 問題在標註：每個標註只框到機場的 1% ~ 9% | Day 28 |

這五列有一個共通點：`每一次都是換了一把新的尺才發現的`。

- mAP 量不出洩漏，要回頭看訓練圖的來源才看得到
- 像素量不出偏移，要換成經緯度、換成公尺才看得到
- 一張圖量不出泛化，要換 13 張沒看過的圖才看得到
- 模型輸出量不出標註的問題，要把標註本身對位回地圖才看得到

## 已知債：模型泛化
Day 27 拿 13 張沒看過的圖測（8 座機場、5 個困難負樣本）：

```
機場召回     5 / 8
框的精確率   10 / 80 = 12.5%
負樣本誤判   5 / 5
```

Day 28 查完原因，三個懷疑只有一個成立：

| 懷疑 | 結果 |
|---|---|
| 門檻太低 | 不成立，誤判框的分數中位數 0.421，比命中框的 0.350 還高 |
| 色彩跟訓練圖不同 | 部分成立，直方圖對齊後框從 80 個降到 27 個，精確率只到 14.8% |
| 標註有問題 | 成立，10 張來源圖裡 8 張對位回原圖、縮放比都是 1.000，標註只框機場碎片 |

Day 30 的恆春機場也一樣：5 個框命中 1 個，分數最高的 0.701 是山坡。

這筆債系列內沒有還，但修法和工具都準備好了：

1. 用 OSM 的機場圍籬自動產生標註（`fetch_osm_airports.py`）
2. 用 EOX 圖磚抓任何地方的真彩色影像（`inference.cli fetch`）
3. 刻意放入港口、工業區、河床這類負樣本（Day 27 那 5 張就是起點）
4. 用台灣以外的機場訓練、台灣的機場測試，從根本避開洩漏

重訓之後對同樣 13 張圖跑一次 `detect --db`，就能在 PostGIS 裡跟這次的 `rgb` 和 `matched` 直接比。

## 最後的架構圖
把實際做出來的東西重新畫一次：

```
[ 資料源 ]
   |-- EOX Sentinel-2 cloudless 圖磚 (inference/tiles.py)
   |-- OSM 機場圍籬 (preprocessing/fetch_osm_airports.py)
   v
[ AI 模型與空間處理 ]
   |-- tifffile 讀 GeoTIFF，波段選擇 + 拉伸 + CLAHE
   |-- YOLOv8-OBB + SAHI 切片推論 + 幾何合併碎框
   |-- 手寫 UTM / Web Mercator 投影 (inference/geo_utils.py)
   |-- OSM 圍籬評分 inside_frac / coverage (inference/evaluate.py)
   v
[ 資料庫 ]                                   ← Docker
   |-- PostgreSQL + PostGIS (runs / detections / airports)
   v
[ 介面 ]
   |-- CLI：fetch / detect / db-init / runs (inference/cli.py)
   |-- folium 互動地圖 (inference/webmap.py)
```

跟 Day 1 比，`中間兩層更厚，最外面一層更薄`。原本規劃的重心是「做出一個看得到的系統」，最後的重心變成「做出一個能量出自己錯在哪的系統」。

## 31 天的數字
```
文章        31 篇
訓練版本    baseline → v2 → OBB → tiled → mixed
資料庫      29 次偵測、114 個框
推論速度    兩張圖 4.1 秒
投影誤差    0.06 mm（對 PROJ）
泛化        召回 5/8、精確率 12.5%
```

## 寫在最後
Day 1 我說這可能會變成一個 DevLog，「都在 Fine Tuning Model、Debug 還有串接來串接去的苦海中」，結果真的是這樣，只是苦海的位置跟想像的不一樣。

我以為難的是模型，花了六天調 OBB、SAHI、NMM、混合資料集。結果真正卡住的是 Day 8 那幾張標註，是 Day 7 自己寫的那句「Garbage in, Garbage out」。三十天前就寫下來的道理，要繞一大圈、換了好幾把尺才真的看懂。

如果只能留一件事給之後的自己，大概是：`結果看起來很好的時候，先換一把尺量量看`。

謝謝一路看到這裡的大家，這次的鐵人賽就到這裡，我們下次見。
