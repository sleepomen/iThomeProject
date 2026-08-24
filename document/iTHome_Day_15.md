在昨天說完`仿射變換`之後我們就要開始實作了，今天目標是透過一個小腳本來實現，把 YOLO 輸出的中心點 (x,y) 轉換為我們看得懂的 GPS 座標。

## 套件安裝
我們必須先安裝這個套件才能進能下一步的動作:
```
pip install rasterio pyproj
```

`rasterio` 負責讀取 GeoTIFF 衛星圖像，並進行 Affine 矩陣運算。
`pyproj` 負責處理不同 EPSG 座標之間的轉換。

## 扣
在根目錄下的 `inference` 資料夾新增 `geo_utils.py` 的檔案，專門用來做處理空間計算

```
import rasterio
from pyproj import Transformer

def pixel_to_latlon(tif_path: str, pixel_x: float, pixel_y: float) -> tuple:
    """
    將衛星圖上的像素座標轉換為 Google Maps 支援的 WGS84 經緯度。
    """
    with rasterio.open(tif_path) as src:
        # 1. 取得這張圖片的空間轉換矩陣與原始座標系統
        affine_transform = src.transform
        source_crs = src.crs
        
        # 2. 透過矩陣相乘 將像素轉換為原始投影座標 
        geo_x, geo_y = affine_transform * (pixel_x, pixel_y)
        
        # 3. 設定目標座標系為 EPSG:4326
        target_crs = "EPSG:4326"
        
        # 4. 建立轉換器 並將投影座標轉換為經緯度
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        lon, lat = transformer.transform(geo_x, geo_y)
        
        return lat, lon

if __name__ == "__main__":
    test_tif = "test_image_967268.tif" 

    test_px, test_py = 300, 450 
    
    try:
        lat, lon = pixel_to_latlon(test_tif, test_px, test_py)
        print(f"Finsih")
        print(f"lat: {lat}, lon: {lon}")
        print(f"Google Maps Link: https://www.google.com/maps/search/?api=1&query={lat},{lon}")
    except Exception as e:
        print(f"error: {e}")
```

## 扣解釋
### 矩陣運算
在這行 `affine_transform * (pixel_x, pixel_y)` 他直接在底層跑完了我們昨天講的線性代數公式。

### 動態 CRS 捕捉
腳本會自動讀取 `src.crs`，不管下載的衛星圖是在哪個 UTM Zone 他都能夠直接辨別，然後轉換成 Google Maps 通用的 `EPSG:4326`。

### 導航
直接把算出的經緯度加入 Google Maps 的 API 網址格式裡。

## 小結
今天讓 rasterio 套件能夠辨識輸出的影像的地理真實位置，那明天我們將會將它自動化，讓 YOLO 每次的輸出都能附帶真實世界經緯度位置，方便我們做後續的操作，那我們明天見。

