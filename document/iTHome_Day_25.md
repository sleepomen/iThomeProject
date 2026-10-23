Day 24 我們讓每個框都帶著經緯度出來，也發現那個框`沒有命中機場基準點（ARP）`，差了 907 公尺。但那張圖的底圖是我們自己的 Sentinel-2 影像加一張手畫的經緯網格，說「框壓在儲槽區上」其實是用眼睛猜的。今天要把 GeoJSON 丟上`真的地圖`，看看那個框到底壓在什麼東西上面。

## 為什麼要上地圖
GeoJSON 本身只是一串數字：

```json
"coordinates": [[[121.2490, 25.0991], [121.2433, 25.1056], [121.2072, 25.0792], ...]]
```

人看不懂，但地圖看得懂。今天的目標有兩個輸出：

1. **互動地圖**：用 `folium` 做一個 HTML 檔，可以拖、可以縮放、可以點框看屬性、可以切換底圖
2. **靜態圖**：文章要放圖，總不能叫大家都去開 HTML，所以要能在 Python 裡把底圖抓下來、把框畫上去

## folium：Python 包一層的 Leaflet
`folium` 是 Leaflet.js 的 Python 包裝，寫 Python、吐 HTML。它是純 Python 套件，沒有任何 C extension，這次 Smart App Control 總算沒來找麻煩：

```
pip install folium
Successfully installed folium-0.20.0
```

建一張地圖、加底圖、加 GeoJSON，幾行就好：

```python
m = folium.Map(location=[25.0767, 121.2328], zoom_start=13, tiles=None)
folium.TileLayer(tiles=ESRI_URL, attr="Esri", name="Esri 高解析衛星").add_to(m)
folium.GeoJson(feature, style_function=lambda _: {"color": "#e74c3c"}).add_to(m)
folium.LayerControl().add_to(m)
m.save("day25_map.html")
```

這裡有個坑要先講：`folium 的座標順序是 (lat, lon)`。

GeoJSON 是 `[lon, lat]`（RFC 7946 規定 x 在前），folium 的 `location` 卻是 `[lat, lon]`（Leaflet 的習慣）。直接把 GeoJSON 物件丟給 `folium.GeoJson` 它會自己處理，但手動放點的時候一定要反過來。Day 24 剛被 GeoJSON 的繞行方向坑過，今天又遇到一個順序問題，`GIS 的世界裡，順序永遠是第一個要確認的東西`。

我放了三種底圖，右上角可以切換：

| 底圖 | 解析度 | 用途 |
|---|---|---|
| Sentinel-2 cloudless（EOX） | 10 m | 跟我們的模型`同一個感測器`，看模型「看到」的世界 |
| Esri World Imagery | 約 0.5 m | 給人眼對照，看清楚地面到底是什麼 |
| OpenStreetMap | 向量圖 | 看道路、地名 |

第一個特別重要，晚點會再用到。

## 靜態圖：自己拼圖磚
互動地圖的底圖是瀏覽器即時下載的，Python 這邊拿不到。要出靜態圖就得自己把底圖抓下來，這就要先搞懂`圖磚（tile）`是怎麼運作的。

網路地圖的規則其實只有三條：

1. 投影一律是 `Web Mercator（EPSG:3857）`
2. 第 z 層把整個世界切成 `2^z x 2^z` 張 256 px 的方塊
3. 方塊編號 (x, y) 從左上角（180°W, 85.05°N）開始數

所以「某個經緯度在第 z 層的哪一張圖磚」是可以直接算的：

```python
def _global_px(lon, lat, zoom):
    x, y = lonlat_to_mercator(lon, lat)
    res = resolution(zoom)                 # 第 zoom 層一個像素幾公尺
    return (x + MERCATOR_HALF) / res, (MERCATOR_HALF - y) / res
```

除以 256 取整數就是圖磚編號，餘數就是在圖磚裡的位置。

### Web Mercator 比 UTM 好算太多了
Day 24 為了 UTM 逆投影寫了一大串級數展開，Web Mercator 則是`正逆兩個方向都有封閉解`：

```python
def lonlat_to_mercator(lon, lat):
    x = A * np.radians(lon)
    y = A * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y

def mercator_to_lonlat(x, y):
    lon = np.degrees(x / A)
    lat = np.degrees(2 * np.arctan(np.exp(y / A)) - np.pi / 2)
    return lon, lat
```

原因是 Web Mercator 偷懶，`把地球當成正球體`（半徑直接用 WGS84 的長半軸）。這也是它被測繪界嫌棄的原因，但對畫網路地圖來說夠用了。

我把這兩個函式加進 `geo_utils.py`，讓 `pixel_to_lonlat` 看 EPSG 代碼自動分派：

```python
def proj_to_lonlat(x, y, epsg: int):
    if epsg == WEB_MERCATOR:
        return mercator_to_lonlat(x, y)
    return utm_to_lonlat(x, y, zone_from_epsg(epsg))
```

這樣做的好處是：`拼好的底圖也有一個 GeoRef`，跟 Day 24 的 UTM GeoTIFF 用法一模一樣。要把框畫到底圖上，只要：

```
GeoJSON 經緯度 -> lonlat_to_pixel(底圖的 georef) -> 底圖像素 -> cv2.polylines
```

Day 24 寫的 `lonlat_to_pixel` 不用改一行就能用在新的投影上，這是當初把 GeoRef 抽成獨立物件的回報。

### 尺度要算對
第 z 層的解析度是赤道上的值，到了台灣要乘上 `cos(緯度)`：

```python
def ground_resolution(zoom, lat):
    return resolution(zoom) * np.cos(np.radians(lat))
```

```
靜態圖第 14 層，實地解析度 8.65 m/px
```

Web Mercator 會把高緯度拉大，台灣在 25°N，平面上的 1 px 只有赤道的 0.91 倍長。這個數字 Day 27 會再用到。

最後補一個本機快取，抓過的圖磚存在 `data/tiles_cache/`。圖磚伺服器是別人的，`同一張不要抓兩次`。

## 換一把尺：從一個點到一個多邊形
Day 24 的判準是「ARP 在不在框裡」，一個點只能回答是或否，回答不了「框到的東西有多少是機場」。

Day 24 也發現，資料集的類別是 `airport` 不是 `runway`。既然要框整座機場，就需要`整座機場的範圍`當地面真相。

OpenStreetMap 正好有：`aeroway=aerodrome` 這個標籤畫的就是機場的圍籬範圍。用 Overpass API 把台灣本島有 IATA 代碼的機場全抓下來：

```python
QUERY = """
[out:json][timeout:90];
(
  way["aeroway"="aerodrome"]["iata"](21.8,119.9,25.4,122.1);
  relation["aeroway"="aerodrome"]["iata"](21.8,119.9,25.4,122.1);
);
out geom;
"""
```

```
CYI  RCKU  Chiayi Airport
HUN  RCYU  Hualien Airport
KHH  RCKH  Kaohsiung International Airport
RMQ  RCMQ  Taichung Airport
TPE  RCTP  Taiwan Taoyuan International Airport
TSA  RCSS  Taipei Songshan Airport
...
寫出 data/reference/tw_airports.geojson  13 座機場
```

有一個小坑：桃園機場在 OSM 上是 `relation` 不是 `way`。relation 的外環被拆成好幾段 way，要先 `linemerge` 接起來再 `polygonize` 才拼得回完整的多邊形：

```python
outer = [LineString(...) for m in el["members"] if m["role"] == "outer"]
shell = unary_union(list(polygonize(linemerge(outer))))
```

### 兩個新指標
有了多邊形就能問兩個更有用的問題：

- **inside_frac**：框的面積有幾成落在機場圍籬內，衡量`框得準不準`
- **coverage**：機場的面積有幾成被框到，衡量`框得全不全`

面積一定要在`公尺`的平面上算。經緯度的一度在東西向和南北向長度不同，直接拿度數算面積，在台灣這個緯度東西向會被高估約 10%。所以先用 Day 24 寫好的 `lonlat_to_utm` 把多邊形搬到 UTM 平面：

```python
def to_metric(geom, zone):
    return transform(partial(_utm_xy, zone=zone), geom)

inter = box_m.intersection(airport_m).area
inside_frac = inter / box_m.area
coverage = inter / airport_m.area
```

框要有過半面積落在機場內才算`命中`，這個 50% 的門檻是我定的，之後的評估都用這個標準。

## 結果
```
OSM 桃園機場圍籬 relation/19784610，中心 25.0790N 121.2327E

框 score=0.5067  配對機場 TPE
  框面積              4.32 km²
  框落在機場內的比例  45.8%
  機場被框到的比例    17.7%
  命中 (>= 50%)?      False
```

畫出來：

![day25 side by side](runs/predict/day25_map/day25_side_by_side.jpg)

左邊是 Esri 的高解析影像，右邊是 Sentinel-2 cloudless，`藍色虛線`是 OSM 的機場圍籬，`紅色`是模型的框（沒命中所以是紅的），`青色十字`是 ARP。

有了真的底圖，Day 24 的描述要修正一下：

- 框`沒有`壓在儲槽區上。那一圈圓形油槽在框的西邊，框本身壓在`機場圍籬外西北側的農地`上
- 框的東南長邊大約沿著北跑道（05L/23R）走，所以有 45.8% 的面積還是在圍籬內
- 整座機場 11.2 km²，框只蓋到 17.7%，`航廈、南跑道、停機坪全部不在框裡`

Day 24 用經緯網格看是「偏了 1.4 公里」，換成多邊形之後更精確的說法是：`框有一半在機場外，而且只框到機場的北邊一條`。

### 互動地圖
`runs/predict/day25_map/day25_map.html` 用瀏覽器打開就能玩。點紅框會跳出屬性表，包括分數、碎片數、長寬比、inside_frac，還有一個 Google Maps 連結：

```
score         0.5067
n_fragments   2
aspect        5.05
iata          TPE
inside_frac   0.4584
coverage      0.1768
hit           False
```

整個 HTML 只有 15.8 KB，因為底圖不在檔案裡，是開啟時才從圖磚伺服器下載。

## 小結
今天把 Day 24 的 GeoJSON 放上了真的地圖：

1. `folium` 做互動地圖，記得它的座標是 `(lat, lon)`，跟 GeoJSON 相反
2. 搞懂圖磚的規則之後，自己拼底圖出靜態圖。`Web Mercator` 正逆投影都有封閉解，加進 `geo_utils` 之後，`lonlat_to_pixel` 不用改就能用
3. 用 OSM 的機場圍籬換掉 Day 24 的單一 ARP 點，多了 `inside_frac` 和 `coverage` 兩個指標

結論是那個框`一半在機場外、只蓋到機場的 17.7%`，用 50% 的門檻算是沒命中。

今天還順手做了兩件明天會用到的事：OSM 機場圍籬是很好的`空間資料表`，而偵測結果也已經是乾淨的 GeoJSON。明天要把 Day 3 開好、從 Day 8 之後就沒動過的 `PostGIS` 接上來，讓資料庫幫我們做這些空間運算，那我們明天見。
