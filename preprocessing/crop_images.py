import os
import rasterio
from rasterio.windows import Window
import numpy as np
import cv2

file_path = r'data\raw\som_san_air.tif' 
output_dir = r'dataset\images'       
tile_size = 640                    

os.makedirs(output_dir, exist_ok=True)

with rasterio.open(file_path) as src:
    img_width, img_height = src.width, src.height
    tile_count = 0
    stride = 160
    for y in range(0, img_height - tile_size + 1,stride):
        for x in range(0, img_width - tile_size +1,stride):
            
            #視窗範圍
            window = Window(col_off=x, row_off=y, width=tile_size, height=tile_size)

            img_data = src.read([1, 2, 3], window=window).astype(float)
            
            #動態直方圖拉伸
            for i in range(3):
                band = img_data[i]
                valid_pixels = band[band > 0] 
                if len(valid_pixels) > 0:
                    p_low, p_high = np.percentile(valid_pixels, (2, 90))
                    if p_high > p_low:
                        img_data[i] = (band - p_low) / (p_high - p_low)

            img_data = np.clip(img_data, 0, 1)
            img_data = (img_data * 255).astype(np.uint8)
            
            #transpose 轉置
            img_data = np.transpose(img_data, (1, 2, 0))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            filename = f"tile_x{x}_y{y}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), img_data)
            tile_count += 1

    print('Finish')