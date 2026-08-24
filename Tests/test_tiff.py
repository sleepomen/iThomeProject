import rasterio
from rasterio.plot import show
import numpy as np

file_path = r'images\som_san_air.tif'

with rasterio.open(file_path) as src:
    print("=== 衛星影像資訊 ===")
    print(f"影像大小: {src.width} x {src.height} 像素")
    print(f"波段數量: {src.count}")

    img_data = src.read([1, 2, 3]).astype(float)

    for i in range(3):
        band = img_data[i]

        valid_pixels = band[band > 0]
        
        if len(valid_pixels) > 0:
            p_low, p_high = np.percentile(valid_pixels, (2, 90))

            img_data[i] = (band - p_low) / (p_high - p_low)

    img_data = np.clip(img_data, 0, 1)

    print("正在開啟影像視窗...")
    show(img_data, transform=src.transform)