今天的計畫很單純，把 Day 18 到 Day 22 散在各個測試腳本裡的步驟`串成一條管線`，讓一個函式吃 GeoTIFF 就能吐出偵測結果。結果一動手就撞牆，而且撞了三次，最後一次撞出來的東西讓我開始懷疑前面幾天的數字。

## 第一撞：作業系統把 GIS 套件擋掉了
開工第一件事是 import rasterio，然後就炸了：

```
ImportError: DLL load failed while importing _base: 應用程式控制原則已封鎖此檔案。
```

上個工作階段還跑得好好的。翻 Windows 的 Code Integrity 事件記錄：

```
Code Integrity determined that a process (python.exe) attempted to load
...\.venv\Lib\site-packages\rasterio.libs\jpeg62-a54a03145fbeae6eca2bf1276664dfc4.dll
that did not meet the Enterprise signing level requirements or violated code integrity policy
```

`Smart App Control`（智慧型應用程式控制）進入強制模式，把 rasterio 綁的`未簽署 GDAL DLL` 全擋了。被點名的有 `jpeg62.dll` 跟 `proj_9.dll`。

先確認災情範圍：

| 套件 | 狀態 |
|---|---|
| numpy / cv2 / shapely / torch / ultralytics / sahi / PIL | 正常 |
| rasterio | **擋掉** |
| pyproj | 沒裝（但它也綁 PROJ，八成會一起中彈） |

只有 rasterio 掛掉，但它正好是整條管線的入口。

### 解法：這個專案根本不需要 GDAL
關掉 Smart App Control 可以馬上解決，但那是`不可逆`的，關掉之後除非重灌 Windows 否則無法再打開。為了一個套件把整台機器的安全機制永久關掉，划不來。

回頭想，這個專案對 GIS 的需求其實只有三件事：

1. 像素資料
2. 仿射矩陣
3. 座標系統代碼

而 `tifffile` 是`純 Python` 的，沒有任何原生 DLL，三件事全給得出來：

```
shape (923, 1349, 158) float32
ProjectedCSTypeGeoKey: 32651
ModelTransformation: [[10.0, 0.0, 0.0, 315080.0], [0.0, -10.0, 0.0, 2779850.0], ...]
```

rasterio 給的東西它一樣不少，而且順便省掉 23 MB 的 GDAL。

### 仿射矩陣有兩種存法
寫讀取層的時候踩到一個 GeoTIFF 規格的細節，`地理定位資訊有兩種存法`，都得處理：

```python
if "ModelTransformation" in meta:
    # 一個完整的 4x4 矩陣，可以表達旋轉
    m = np.asarray(meta["ModelTransformation"]).reshape(4, 4)
    transform = np.array([[m[0, 0], m[0, 1], m[0, 3]],
                          [m[1, 0], m[1, 1], m[1, 3]]])
elif "ModelPixelScale" in meta and "ModelTiepoint" in meta:
    # 縮放 + 一個對位點，只能表達平移縮放
    sx, sy = meta["ModelPixelScale"][:2]
    i, j, x, y = tie[0], tie[1], tie[3], tie[4]
    transform = np.array([[sx, 0.0,  x - i * sx],
                          [0.0, -sy, y + j * sy]])
```

多數正射校正過的影像走第二種，我們手上這張走第一種。第二種要特別注意 `-sy`，因為`影像的 row 往下增加，投影座標的 y 往上增加`，符號相反。

rasterio 把這兩種都藏在 `src.transform` 後面，自己寫才會看到。

## 第二撞：158 個波段裡只有 14 個有東西
換成 tifffile 之後順手把波段統計印出來，這才發現 Day 18 一直沒人問的問題：`這 158 個波段到底是什麼`。

```
0-12    3336 3235 3282 3110 3437 3880 3985 3879 4071 4668 3327 2730  297
13      3641
14-127  幾乎全是 0
128-157 ~50 / ~2 交錯
```

前 14 個是真正的 Sentinel-2 光譜波段（數值 300 到 4700，典型的反射率乘以 10000），中間`114 個波段是純 0 的填充層`，最後 30 個是數值範圍完全不同的衍生圖層。

換句話說，`這個檔案有 91% 是空的`。

### 一個藏了五天的 bug
Day 18 寫的前處理是這樣：

```python
img = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
```

rasterio 的波段編號從 1 開始，所以 `[1, 2, 3]` 取的是`索引 0, 1, 2`，也就是 Sentinel-2 的 `B1 / B2 / B3`。

但真彩色應該是 `B4 紅 / B3 綠 / B2 藍`，對應`索引 3, 2, 1`。

我們一直在把 **B1（海岸氣膠波段）當成紅色通道**。而 B1 的原生解析度是 `60m`，其他可見光波段是 `10m`，重採樣到 10m 網格之後它就是一團糊。

把兩種渲染擺在一起：

![day23 band compare](runs/predict/day23_bands/day23_band_compare.jpg)

左邊是我們用了五天的版本，`整張蒙著一層霧`。右邊是真彩色，跑道、道路、建築物全都清楚得多。

看到這裡我以為今天要收一個大成果，修好波段順序，模型應該會看得更清楚。

## 第三撞：修好之後，結果變爛了
管線寫完之後我跑了三組對照，其他參數完全相同（混合模型、切 320、conf 0.25、CLAHE、Day 21 幾何合併）：

```
[A. bands 0,1,2 + max 拉伸]         原始  2 框 -> 合併後  1 框
   碎片x2 conf=0.507 角度=142.0 長寬比=5.05

[B. bands 3,2,1 + max 拉伸]         原始 13 框 -> 合併後  7 框
[C. bands 3,2,1 + 百分位數拉伸]      原始 21 框 -> 合併後 13 框
```

![day23 pipeline](runs/predict/day23_pipeline/day23_side_by_side.jpg)

左邊那張霧濛濛的圖，`一個框、正中跑道、長寬比 5.05`。右邊兩張清晰漂亮的圖，`框滿天飛`，城市街區、海岸線、雲層邊緣全都中招。

我第一個反應是門檻沒跟著調，圖變亮了模型當然更容易開火。掃了一輪信心度：

| conf | A 原始/合併 | A 最佳長寬比 | C 原始/合併 | C 最佳長寬比 |
|---|---|---|---|---|
| 0.25 | 2 / 1 | **5.05** | 21 / 13 | 2.83 |
| 0.35 | 1 / 1 | **4.71** | 13 / 9 | 2.65 |
| 0.45 | 1 / 1 | **4.71** | 3 / 3 | 1.56 |
| 0.55 | 0 / 0 | - | 2 / 2 | 1.49 |
| 0.65 | 0 / 0 | - | 0 / 0 | - |

調門檻救不回來。C 把框數壓到 3 的時候，`沒有任何一個框的形狀像跑道`（最佳長寬比只有 1.56），而且中心點散落各處。

A 的乾淨不是因為它對，是因為`曝光不足讓模型半盲`，它只對整張圖最強的那個訊號開火，而那個訊號剛好是跑道。`欠曝當成了信心度過濾器`。

## 第四撞：為什麼模型這麼脆弱
到這裡我想不通。模型再弱，換個色彩渲染也不該從「一個完美的框」崩成「十三個亂框」。除非它`根本沒有在學跑道的通用特徵`。

於是我回去看訓練資料。Roboflow 的檔名會把來源圖跟增強版本編在一起，數一數唯一來源：

```
train 唯一來源圖: 6
valid 唯一來源圖: 2
test  唯一來源圖: 2
```

`18 張訓練圖是 6 張原圖乘以 3 種 Roboflow 增強`。然後我把這 10 張唯一來源圖畫出來：

![unique sources](runs/predict/day23_pipeline/unique_sources.jpg)

十張全是`同一個場景`。同一條海岸線、同一座機場、同一團雲，只是裁切位置跟旋轉角度不同。

而且它們跟我們一路在測試的那張 GeoTIFF `是同一個地方`。

這代表三件事，一件比一件糟：

1. **train / valid / test 互相洩漏**。三個 split 共用同一批像素，驗證集量到的是`記憶`不是`泛化`。
2. **Day 20 跟 Day 22 的交叉驗證表要重新解讀**。那些 mAP 數字不是在量「模型認不認得機場」，是在量「模型記不記得這一座機場」。
3. **模型對渲染方式極度敏感是必然的**。它學到的是「這張圖在這個色彩渲染下長這樣」，色彩一動，記憶就對不上了。

Day 19 我們花了一整天講 Domain Gap，講感受野不匹配。現在看起來，真正的 domain gap 從第一天就在了，而且大得多，`訓練集只有一個 domain，裡面只有一座機場`。

## 管線的預設值要選哪個
知道真相之後，預設值反而好決定了。管線的預設是：

```python
bands = SENTINEL2_FIRST_THREE   # (0, 1, 2)
stretch = "max"
```

刻意`對齊訓練資料的長相`，而不是物理正確的真彩色。程式碼裡把理由寫清楚：

```python
# 預設值刻意對齊訓練資料的長相而不是物理正確的真彩色。
# 換成 (3, 2, 1) + 百分位數拉伸，人眼看起來清楚得多，
# 但模型會在城市紋理與海岸線上狂開火（Day 23 實測 1 框 -> 13 框）。
```

這是一個`已知的債`，不是一個正確的設計。等訓練資料修好了，這兩個預設值都該換掉。

## 管線長什麼樣
最後把東西串起來。以前這些邏輯寫死在 `Tests/test_nmm_postprocess.py` 的 `load_image()` 裡，誰要用都得 import 一支測試檔案，正式邏輯躺在測試裡本身就是個味道。現在是：

```python
detector = AirportDetector()          # 權重、切片尺寸、門檻、波段都綁在這
result = detector.detect("data/raw/som_san_air.tif")

for d in result.detections:
    print(d.score, d.angle, d.aspect, d.center)
```

幾個設計上的取捨：

- **模型延遲載入**。建立 `AirportDetector` 很便宜，只有真的要推論時才把權重搬上 GPU，這樣拿它當設定容器不會有負擔。
- **回傳 dataclass 不是 dict**。`Detection` 有 `aspect` 跟 `center` 兩個算出來的屬性，Day 21 那版用 dict 傳來傳去，欄位名打錯不會有人告訴你。
- **`PipelineResult` 同時帶回前處理後的影像與 `georef`**。影像給畫圖用，`georef` 是明天要接經緯度的入口。

### 驗收：數字要一模一樣
換掉讀圖套件最怕的就是悄悄改變行為，所以我把 Day 21 跟 Day 22 的腳本原封不動再跑一次：

```
[before] SAHI 預設後處理輸出 6 個碎框
[after ] 幾何合併後 4 個框
   members=[0, 2, 4]  conf=0.559  角度=145.6  長=566.0  寬=224.9  長寬比=2.52
```

`跟 Day 21 當天的輸出一個字都不差`。tifffile 對這個專案來說是 rasterio 的等價替代品，這下可以放心了。

## 小結
今天本來只想把管線串起來，結果串出四件事：

1. Smart App Control 擋掉 rasterio 的 GDAL DLL，改用純 Python 的 `tifffile`，順便發現 GeoTIFF 的仿射矩陣`有兩種存法`
2. 158 個波段裡`只有 14 個有內容`，其餘 91% 是空的
3. 我們從 Day 18 就一直把 `B1 海岸氣膠波段當紅色通道`，修好之後畫面清楚多了，但`偵測結果從 1 個框崩成 13 個框`
4. 追查原因時發現`訓練集的 10 張唯一來源圖全是同一個場景`，train / valid / test 互相洩漏

第四點才是今天真正的收穫，雖然它是個壞消息。前面幾天所有的指標、所有的交叉驗證表，量到的都是`模型對一座機場的記憶`。

明天照原訂計畫要接`像素座標到經緯度`，把 Day 15 那支從來沒跑起來過的 `geo_utils.py` 接上管線。不過 pyproj 大概也會被 Smart App Control 擋掉，所以明天可能得自己動手算 UTM 逆投影，那我們明天見。
