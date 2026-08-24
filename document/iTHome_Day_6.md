昨天我們用 Python 跟 Rasterio 處理了從 Sentinel-2 抓下來的資料，讓我們可以清楚的知道這張圖的地理資訊，今天我們要把影像變成一個模型看得懂的東西，處理成電腦能讀取的數據。

## 為甚麼影像需要切分
衛星遙測影像常常具備極廣的涵蓋範圍跟龐大的資料量，單一張的影像陣列數量常常達到數千或是數萬等級，壓縮後的檔案會非常巨大，在做深度學習的時候這種超大尺度的影像輸入至 GPU 進行卷積神經網路訓練的時候會讓顯卡的記憶體崩潰，導致運算程序強制終止。

為了解決龐大資料量計算的問題，在前期資料處理階段會做 Sliding Window 裁切技術，透過設定固定步幅與視窗大小，將原本龐大的影像資料切分成局部區塊，例如常見的 640 x 640，把他轉成標準的 JPG 格式，讓巨量資料控制在硬體可負荷的範圍，也能符合神經網路的批次訓練 (Batch Training) 時的輸入規範。

## 模型怎麼看圖片
針對記憶體的限制常常會有一個不太現實的想法，為甚麼不把大圖直接強制縮放到 640 x 640，還要花時間切成幾百張小圖?

因為這關乎到卷積神經網路如何進行 `特徵提取 (Feature Extraction)` 的本質，用 YOLO 架構為例，他預設的輸入尺寸為 640x 640/512 x 512，在原始影像數據中我們需要辨識的範圍相當小，可能僅僅 15 x 15 像素。

### 如採用全圖縮放
高達數十倍的採樣會導致影像中的高頻空間資訊在壓縮過程完全流失，假設影像中需要提取船舶的特徵，它會很容易被壓成不足 1像素的雜訊，會導致模型在特徵映射時無法把目標跟背景區分開來，產生嚴重的漏檢。

### 如採用滑動視窗切分
它的好處是在不改變空間解析度的前提下擷取部分局部特徵，這代表每個像素的物理距離維持不變，因此像船舶這種小部分特徵能夠被偵測，同時保留原始尺寸跟細節，確保 YOLO 能夠捕捉到小目標的語意資訊，讓預測準確率提升。

## 扣
我們會用到 Rasterio 裡面的 Window 功能，它可以讓我們只讀取圖中某一個小區塊，會搭配 OpenCV 來存圖。

```
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

    for y in range(0, img_height, tile_size):
        for x in range(0, img_width, tile_size):
            
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

```
首先我們指定路徑後就開始操作，用巢狀迴圈遍歷整張圖，指定這次的切分範圍，讀取前三個波段並轉為浮點數，使用昨天的動態直方圖拉伸並且濾除邊界的 NO Data，確保數值在0至1間後轉為一般影像的 0~255，因為 OpenCV 是讀(H,W,C)但 Rasterio 讀(B,H,W)，所以要做一下 transpose 轉置，顏色同樣，做完以上操作後標記原始座標，自動儲存在指定路徑的資料夾。

## 小結
今天把昨天處理好的 .tif 資料做切分處理，讓我們可以在未來套用這樣的公式做資料處理，在資料前處理的部分也是到了一個階段，明天預計會進入`人工標註 (Data Annotation)`，那我們就明天見。