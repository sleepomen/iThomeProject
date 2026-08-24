import rasterio
from pyproj import Transformer

def pixel_to_latlon(tif_path: str, pixel_x: float, pixel_y: float) -> tuple:
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
    test_tif = "data/raw/som_san_air.tif" 

    test_px, test_py = 300, 450 
    
    try:
        lat, lon = pixel_to_latlon(test_tif, test_px, test_py)
        print(f"Finsih")
        print(f"lat: {lat}, lon: {lon}")
        print(f"Google Maps Link: https://www.google.com/maps/search/?api=1&query={lat},{lon}")
    except Exception as e:
        print(f"error: {e}")