昨天我們把「感受野領域偏移」的理論拆解完畢，結論很簡單一句話：`推論時怎麼切，訓練時就怎麼切`。今天我們就要把這句話變成扣，用 `Shapely` 幾何運算把訓練集切碎、把被切斷的 Polygon 重新映射，然後重新訓練一次模型，看看那條被砍掉半截的跑道到底救不救得回來。

## 為什麼不能直接用 OpenCV 切
最直覺的做法是 `img[y:y+320, x:x+320]` 一行解決，但這只切了圖片，沒有切標註。標註是一組 `(x, y)` 正規化座標組成的封閉多邊形，當切片邊界橫過跑道時會發生三件事：

1. 一個多邊形可能被切成`兩塊甚至三塊`（想像切片框剛好卡在跑道的轉角）。
2. 切完之後的形狀`不再是原本的多邊形`，需要沿著切片邊界產生新的閉合邊。
3. 座標系從`全局`變成`局部`，全部要重新正規化。

這三件事在幾何學上就是一次`布林交集（Boolean Intersection）`，所以我們請 `Shapely` 上場。

## 核心：切片視窗與跑道輪廓做交集
整套邏輯的心臟只有這幾行：

```python
window = box(ox, oy, ox + TILE_SIZE, oy + TILE_SIZE)

for cls_id, poly in polygons:
    if not poly.intersects(window):
        continue
    for part in geom_to_parts(poly.intersection(window)):
        ...
```

`box()` 把切片網格變成一個正方形 Polygon，`poly.intersection(window)` 直接吐出裁切後的新幾何形狀，`Shapely` 會自動幫我們沿著邊界補上新的閉合邊，完全不用自己算線段交點。

### 陷阱一：intersection 回傳的型別不固定
`intersection()` 的回傳值可能是 `Polygon`、`MultiPolygon`，甚至是 `GeometryCollection`（當交集退化成一條線或一個點時）。如果不做型別判斷直接抓 `.exterior`，程式馬上炸給你看：

```python
def geom_to_parts(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty]
    return []
```

### 陷阱二：手繪標註幾乎都是無效多邊形
標註人員用滑鼠沿著跑道邊緣拖出來的輪廓，常常會`自相交（self-intersection）`，這種多邊形在 Shapely 眼中是 invalid，做交集會直接丟例外。解法是幾何界的萬用修復術 `buffer(0)`：

```python
poly = Polygon(pts)
if not poly.is_valid:
    poly = make_valid(poly).buffer(0)
```

`buffer(0)` 會把多邊形拆解重組成合法的形狀，代價是可能變成 MultiPolygon，但這正好被上面的 `geom_to_parts()` 接住。

### 陷阱三：碎片標註是毒藥
切片邊界如果只掃過跑道的一個角，會產生一個面積只有原本 1% 的小三角形。這種標註餵給模型等於在教它「這一小塊灰色三角形就是機場」，是純粹的噪音。所以我們設了雙重門檻：

```python
if part.area < MIN_ABS_AREA or part.area < poly.area * MIN_AREA_RATIO:
    stats["frag_drop"] += 1
    continue
```

絕對面積下限 64 px²，相對面積下限 5%，兩個條件任一不過就丟。

### 座標重映射
最後把全局座標平移成局部座標，再正規化：

```python
pts[:, 0] = np.clip(pts[:, 0] - ox, 0, tile) / tile
pts[:, 1] = np.clip(pts[:, 1] - oy, 0, tile) / tile
```

這裡的 `clip` 是保險，因為 `simplify()` 化簡多邊形時有極小的機率會讓點稍微跑出邊界外。另外要記得 Shapely 的 `exterior.coords` 會`重複收尾點`（第一點 == 最後一點），寫進 YOLO 標註前要用 `[:-1]` 砍掉。

## 背景失衡的採樣策略
Day 19 提到的問題在這裡兌現：切完之後大部分切片是純海面、純市區、純雲層。實作上用一個很簡單的分流：

```python
if lines:
    stats["pos"] += 1              # 有標註碎片，100% 保留
elif rng.random() < NEG_KEEP_RATIO: # 純背景，只留 15%
    stats["neg_kept"] += 1
else:
    continue                        # 其餘直接丟棄
```

跑完的統計數字：

```
[train] 原圖  18 -> 切片   96 (正樣本 85, 背景保留 11, 背景丟棄 66) | 標註碎片 保留 93 / 濾除 25
[valid] 原圖   2 -> 切片   12 (正樣本 12, 背景保留  0, 背景丟棄  6) | 標註碎片 保留 16 / 濾除  4
[test ] 原圖   2 -> 切片   12 (正樣本  9, 背景保留  3, 背景丟棄  6) | 標註碎片 保留  9 / 濾除  4
```

18 張原圖膨脹成 96 張切片，其中 66 張純背景被丟掉。原本的 18 個完整多邊形，被切成了 93 個碎片標註，另外有 25 個太小的碎片被門檻擋下來。

## 切完的標註長怎樣
理論歸理論，切完的標註到底有沒有對齊還是要用眼睛驗。我寫了個小腳本把切片畫出來，`綠色`是重映射後的 Polygon，`紅色`是 YOLO 內部 `cv2.minAreaRect()` 會算出來的 OBB：

![tiled label check](runs/tiled_check/tiled_labels_check.jpg)

左上角跟右上角那兩張最能說明問題，跑道明顯被切片邊界砍斷，綠色多邊形沿著切片邊界收成一條直邊，紅色 OBB 也正確地只框住殘餘的那一段。這就是我們要的效果，模型從今天開始就要學著看`半截跑道`。

## 重新訓練
資料集有了，訓練腳本幾乎沒變，只多加了一個參數：

```python
model.train(
    data=str(project_root / "find-airport-tiled" / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,
    device="0",
    name="airport_obb_tiled",
    fliplr=0.5,
    flipud=0.5,
    degrees=90.0,   # 切片後跑道方向更零碎，直接讓模型看遍所有旋轉角
)
```

`degrees=90.0` 是刻意加的。整圖訓練時跑道方向被機場的整體佈局綁住，但切片之後只剩一截柏油路，方向可以是任意角度，所以乾脆讓 augmentation 把所有旋轉角都掃過一遍。

## 交叉驗證：領域偏移到底有多嚴重
訓練完之後我做了一件更有意思的事，把`兩個模型`丟到`兩個驗證集`上互相交叉評估，四種組合：

| 模型 | 驗證集 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| baseline（整圖訓練） | 原始整圖 valid | 0.853 | 0.500 | 0.495 | 0.346 |
| baseline（整圖訓練） | 切片 valid | **0.000** | **0.000** | **0.000** | **0.000** |
| tiled（切片訓練） | 原始整圖 valid | 0.189 | 0.500 | 0.099 | 0.069 |
| tiled（切片訓練） | 切片 valid | 0.541 | 0.250 | 0.205 | 0.131 |

第二列的`三個 0` 是今天最震撼的數字。那個在整圖上 precision 高達 0.853、看起來表現優異的 baseline 模型，丟到切片驗證集上是`完全歸零`，一個都認不出來。這不是「表現變差」，是`徹底失明`。Day 19 講的 Domain Gap 不是理論上的隱憂，而是一道實測出來的斷崖。

反過來看第三列也一樣有意思，切片訓練的模型回到整圖上 mAP50 只剩 0.099，它把`全局視野`給忘掉了。兩個模型各自只在自己的領域裡work，這是一個非常乾淨的 domain gap 對照組。

## 拉回真實衛星圖
數字歸數字，最後還是要回到 Day 18 那張讓我們卡關的 GeoTIFF。同一張圖、同一套 CLAHE 前處理、同樣 0.25 信心門檻，差別只在權重跟 SAHI 切片尺寸（切片訓練的模型要用 320 才對得上訓練時的視野）：

![day20 compare](runs/predict/day20_compare/day20_side_by_side.jpg)

```
[baseline_slice640] 偵測到 1 個目標 conf=[0.266]
[tiled_slice320]    偵測到 6 個目標 conf=[0.559, 0.354, 0.325, 0.316, 0.295, 0.273]
```

左邊是 Day 18 的老問題，`一個框、0.27 信心度、只包住右上半截`，跑道左下那一大段完全被無視。

右邊的切片模型，`覆蓋範圍終於沿著整條跑道從右上延伸到左下`，最高信心度也從 0.27 拉到 0.56，中央那個 0.56 的框正好壓在跑道主體上。半截跑道的問題，解決了。

## 但是我們換來了新問題
把右圖放大看就會發現，代價很明顯：

1. **框變成碎片**：原本應該是一條完整跑道，現在被拆成 6 個獨立的框，因為切片訓練讓模型學會了「認出跑道的一小段」，但它`不知道這些片段屬於同一條跑道`。
2. **幾何精準度退化**：Day 17 那種緊貼跑道邊緣的漂亮 OBB 不見了，現在的框歪歪斜斜，有幾個甚至跨到旁邊的市區跟海岸線上去。
3. **NMS 沒把碎片縫起來**：SAHI 預設的 NMS 只會抑制`高度重疊`的框，這些沿著跑道排開的碎片彼此重疊率很低，所以一個都沒被合併。

換句話說，我們用`召回率`換掉了`精確度`。模型從「看得見一半」進步到「整條都看得見」，但也從「框得很準」退化成「框得很碎」。

## 小結
今天用 Shapely 的布林交集把訓練集切碎，把 18 張整圖變成 96 張帶著殘缺標註的切片，過程中踩了三個地雷：`intersection 回傳型別不固定`、`手繪標註自相交要靠 buffer(0) 修復`、`小碎片標註是純噪音必須設門檻濾掉`。

交叉驗證跑出來的 `0.000` 證明了 Domain Gap 是真實存在的斷崖，而切片訓練確實把那條被砍半的跑道救了回來，信心度從 0.27 拉到 0.56。

但新的問題也很清楚，一條跑道被拆成六個歪斜的碎框。明天我們要處理這個爛攤子，方向有兩個：一是把 SAHI 的後處理從 `NMS` 換成 `NMM（Non-Maximum Merging）`，讓沿線排開的碎片能被縫合成一條；二是做`混合資料集`，把整圖跟切片一起丟進去訓練，讓模型同時保有全局視野與局部細節，那我們明天見。
