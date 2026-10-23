從 Day 10 第一次訓練到現在，我們的模型只看過`一張圖`。Day 23 發現訓練集的 10 張來源圖全是桃園同一個場景，Day 26 寫好的資料庫查詢也只有一張圖可以問。今天要做的事情很簡單，也是這個系列早就該做的事：`拿模型沒看過的地方去測它`。

## 不用 Copernicus 的理由
原本的計畫是用 Copernicus Data Space 下載 Sentinel-2 原始影像。但實際評估了一下流程：

1. 註冊帳號，處理 OAuth token
2. 搜尋每座機場上空雲量低的日期
3. 下載整個 100 km x 100 km 的 L2A tile（一個好幾百 MB）
4. 自己裁切、組波段、拉伸

為了測 8 座機場，要下載好幾 GB 的資料。

Day 25 做地圖的時候，我放了一個底圖叫 `Sentinel-2 cloudless`，是 EOX 這家公司把一整年的 Sentinel-2 影像`去雲之後拼成的全球鑲嵌圖`：

| | Copernicus 原檔 | EOX Sentinel-2 cloudless |
|---|---|---|
| 感測器 | Sentinel-2 | Sentinel-2 |
| 解析度 | 10 m | 10 m |
| 波段 | 13 個 | RGB（B4/B3/B2）3 個 |
| 雲 | 看運氣 | 已經去掉了 |
| 帳號 | 要 | 不用 |
| 格式 | 整個 tile | 圖磚，要多少抓多少 |

感測器一樣、解析度一樣，而且 Day 25 寫好的 `fetch_mosaic` 本來就能抓它的圖磚。代價是`只有 RGB 三個波段`，沒有我們訓練時用的 B1。這件事等一下會直接撞上。

授權部分，s2cloudless 2020 版是 `CC BY-NC-SA 4.0`，非商業使用要標註來源：「Sentinel-2 cloudless by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2020)」。個人部落格文章沒問題。

## 場景設計
測試集要能回答兩個問題：

- 有機場的時候，`找不找得到`？（召回）
- 沒有機場的時候，`會不會亂框`？（誤判）

所以分兩類：

**正樣本（8 張）**：中心對準 OSM 機場圍籬的形心

```python
POSITIVE = ["TPE", "TSA", "RMQ", "CYI", "TNN", "KHH", "HUN", "TTT"]
```

桃園（TPE）一定要放，它是`對照組`：同一座機場、不同的渲染方式，模型還認不認得？

**負樣本（5 張）**：沒有機場，但有`長條狀的人造物`，專門用來騙模型

```python
NEGATIVE = {
    "neg_banqiao":      (121.455, 25.012, "板橋／新莊市區"),
    "neg_taipeiport":   (121.375, 25.150, "台北港"),
    "neg_taichungport": (120.505, 24.275, "台中港"),
    "neg_mailiao":      (120.195, 23.785, "麥寮六輕工業區"),
    "neg_changhua":     (120.450, 23.950, "彰化農田"),
}
```

港口有長長的碼頭，工業區有大片矩形廠房，河流有筆直的堤防，這些在 10 m 解析度下都很像跑道。農田放進來是當`簡單題`，如果連農田都會框那問題就大了。

每張大小 12 km x 8 km，跟原本的 `som_san_air.tif`（13.5 x 9.2 km）差不多。

## 尺度：8.73 m 不是 10 m
Day 25 算過，第 14 層在台灣的實地解析度是：

```
第 14 層實地解析度 8.73 m/px
```

模型是在 10 m/px 的影像上訓練的。直接拿 8.73 m 的圖去測，同一座機場會`大 15%`。YOLO 對尺度有一定的容忍度，但能對齊就對齊，不要讓變因混在一起。

縮小影像很簡單，`麻煩的是仿射矩陣要跟著改`：

```python
def resample_to_ground(rgb, georef, lat, target=10.0):
    factor = georef.transform[0, 0] * np.cos(np.radians(lat)) / target
    h, w = rgb.shape[:2]
    new_w, new_h = round(w * factor), round(h * factor)
    out = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    t = georef.transform.copy()
    t[0, 0] *= w / new_w          # 像素變大了，每個像素代表的距離跟著變大
    t[1, 1] *= h / new_h
    return out, GeoRef(transform=t, epsg=georef.epsg)
```

左上角的原點不變，只有像素大小要乘上縮放比。縮小用 `INTER_AREA`，它會把多個像素平均起來，比 `INTER_LINEAR` 不容易產生鋸齒。

## 寫 GeoTIFF：自己組 GeoKey
抓下來的影像要存成 GeoTIFF，這樣 Day 23 的管線不用改就能吃。

Day 23 只寫了讀取，今天補上寫入。tifffile 寫圖沒問題，但 GeoTIFF 的地理資訊要自己用 `extratags` 塞進去：

```python
keys = [1, 1, 0, 3,                  # 版本 1.1.0，後面接 3 把鑰匙
        1024, 0, 1, 1,               # GTModelType = 投影座標系
        1025, 0, 1, 1,               # GTRasterType = PixelIsArea
        3072, 0, 1, 3857]            # ProjectedCSType = EPSG:3857

extratags = [
    (33550, "d", 3, (res_x, res_y, 0.0), False),          # ModelPixelScale
    (33922, "d", 6, (0, 0, 0, x0, y0, 0), False),         # ModelTiepoint
    (34735, "H", len(keys), keys, False),                 # GeoKeyDirectory
]
tifffile.imwrite(path, rgb, photometric="rgb", extratags=extratags)
```

GeoKeyDirectory 是一個`扁平的整數陣列`，每 4 個一組：(鑰匙代碼, 存放位置, 個數, 值)。第一組是標頭，最後一個 3 表示後面有 3 把鑰匙。

Day 23 提過 GeoTIFF 有兩種存仿射矩陣的方式，這裡用 `PixelScale + Tiepoint` 那種，也是 GDAL 預設的寫法。

寫完讀回來驗證：

```
(801, 1201, 3) uint8  epsg=3857
影像中心 -> (121.2327, 25.0790)
```

跟 OSM 桃園機場圍籬的形心 `(121.2327, 25.0790)` 完全一致。

13 個場景全部抓下來：

```
tpe                1201x801  機場 ['TPE']  Taiwan Taoyuan International Airport
tsa                1200x801  機場 ['TSA']  Taipei Songshan Airport
rmq                1201x801  機場 ['RMQ']  Taichung Airport
...
neg_mailiao        1201x801  機場 無  麥寮六輕工業區
neg_changhua       1201x801  機場 無  彰化農田

13 個場景 -> data/s2cloudless
```

## 跑模型
管線完全沒改，只有一個參數要想清楚：`波段`。

```python
detector = AirportDetector(bands=(0, 1, 2), stretch="max")
```

Day 23 講過，`(0, 1, 2)` 在原始的 158 波段堆疊裡是 B1/B2/B3，物理上不正確，但模型是在這個組合上訓練的。EOX 的 tif 只有三個波段 R/G/B，所以 `(0, 1, 2)` 在這裡就是`物理正確的真彩色`。

也就是說，今天模型看到的是`它從來沒看過的色彩`。這是我們能拿到的最接近「真實世界」的測試。

每張圖跑完，用 Day 25 的 OSM 圍籬打分數、寫進 Day 26 的 PostGIS。

## 結果
```
scene             機場     raw  boxes  hit   fp  best_inside  結果
tpe               TPE       13      8    3    5       100.0%  找到
tsa               TSA        6      4    0    4         0.0%  漏判
rmq               RMQ       23      9    4    5       100.0%  找到
cyi               CYI       13      7    1    6        59.2%  找到
tnn               TNN       18      9    1    8       100.0%  找到
khh               KHH       14      9    0    9        40.1%  漏判
hun               HUN       10      7    1    6       100.0%  找到
ttt               TTT       15      6    0    6        16.0%  漏判
neg_banqiao       -          5      4    0    4         0.0%  誤判
neg_taipeiport    -          5      5    0    5         0.0%  誤判
neg_taichungport  -         14      4    0    4         0.0%  誤判
neg_mailiao       -          7      5    0    5         0.0%  誤判
neg_changhua      -          9      3    0    3         0.0%  誤判

機場召回率  5/8
框的精確率  10/80 = 12.5%
負樣本誤判  5/5 張有框，共 21 個
```

![day27 sheet](runs/predict/day27_multi/day27_rgb_sheet.jpg)

`藍色`是 OSM 機場圍籬，`綠色`是命中的框（過半面積在機場內），`紅色`是誤判。

### 數字怎麼讀
**召回 5/8 看起來還行，但很寬鬆**。「找到」的定義是`至少有一個框`過半落在機場內。台中（RMQ）9 個框裡有 4 個命中，但另外 5 個框在周圍的農田和河床上。

**精確率 12.5% 才是真相**。模型畫了 80 個框，只有 10 個在機場上。`每 8 個框只有 1 個是對的`。

**負樣本全軍覆沒**。5 張沒有機場的圖，每一張都有框，連最簡單的彰化農田都框了 3 個。

### 模型在框什麼
把紅框一個個看過去，有一個很明顯的共同點：`長條形、邊界筆直的東西`。

- **台北港、台中港**：碼頭和防波堤，又長又直
- **麥寮**：工業區的矩形廠房和儲槽區
- **松山（TSA）**：機場本身沒框到，框都打在`淡水河和基隆河的河岸`上
- **台東（TTT）**：一個巨大的框蓋住卑南溪的河床
- **彰化農田**：大片規則的矩形田區

模型學到的不是「機場」，比較像是`「有直線邊界的大塊區域」`。

### 桃園的對照組
TPE 是模型在原始 tif 上看過的那座機場。換成 EOX 渲染之後：

- Day 24 的原始 tif：1 個框，45.8% 在機場內，沒命中
- 今天的 EOX：8 個框，3 個命中（分別有 62%、94%、100% 在圍籬內）、5 個誤判

同一座機場換了色彩，框的數量跟位置`全部不一樣`。這不太像一個「學會辨識機場」的模型會有的行為。

### 資料庫
這次每張圖都寫進了 PostGIS，Day 26 寫好的漏判查詢終於有用了：

```sql
SELECT r.source, a.iata
FROM runs r JOIN airports a ON ST_Within(a.geom, r.footprint)
WHERE r.source LIKE '%#rgb'
  AND NOT EXISTS (
    SELECT 1 FROM detections d
    WHERE d.run_id = r.id AND ST_Intersects(d.geom, a.geom))
```

```
tsa.tif#rgb  TSA
```

松山機場是唯一一座`連碰都沒碰到`的機場。KHH 和 TTT 都有框擦到圍籬，只是沒有過半，所以 SQL 層面不算完全漏判。這就是 `ST_Intersects`（有碰到）跟 50% 面積門檻（命中）的差別，兩個問題都有用。

## 小結
今天是這個系列第一次真正的泛化測試：

1. 用 EOX `Sentinel-2 cloudless` 代替 Copernicus，同感測器、同解析度、不用帳號
2. 第 14 層在台灣是 8.73 m/px，`重採樣到 10 m` 對齊訓練尺度，仿射矩陣的像素大小要跟著改
3. 補上 `write_geotiff`，自己組 GeoKeyDirectory，讓圖磚影像能直接餵進 Day 23 的管線
4. 13 個場景的結果：召回 5/8、精確率 `12.5%`、負樣本 `5/5 全部誤判`

這個結果不意外。Day 23 就已經知道訓練集只有一個場景，今天只是用數字把它說清楚：`模型沒有學會什麼是機場`。

但「沒學會」有很多種可能。是門檻設太低？是色彩不一樣？還是訓練資料本身就教錯了？明天要做誤判分析，把這三個可能一個一個拆開來驗，那我們明天見。
