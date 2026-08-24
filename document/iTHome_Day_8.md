來到第八天，前兩天都在做資料準備，其實原本沒有明確的想要偵測甚麼，因為影像中最為明顯的就是機場了，所以我打算用機場作為模型訓練依據。

## Problems
但先來講講遇到的問題，其實如果這張不是抓機場就會有點麻煩，因為我發現 Sentinel-2 有個限制，那就是它的空間解析度只有 10m/pixel，這代表圖中一個像素在真實世界大約是 10m x 10m，一台客機大約是 30~45m，在 640 x 640 的圖像中大約只佔了 3~4 個像素點，更不用說比客機還小的東西了，對於肉眼來說根本就是白點，也完全做不了標記工程。

所以說未來如果確認要偵測的物體必須用到商用的高解析度衛星，如果持續用免費的 Sentinel-2 恐怕只能偵測較大的物件，所以這些就是選擇先用機場代替的原因，後續需要偵測更小的物體或是確認方向後我們再決定要怎麼操作。

再來就是昨天做影像切分的時候我們輸出的照片只有六張，為了增加我們的資料量，我改為重疊滑動視窗 (Overlapping Sliding Window)，將滑動步輻 (Stride) 改為 320 像素:

```
# 加上步長 Stride=160，讓視窗互相重疊 75%，產生更多訓練資料
stride = 320 
for y in range(0, img_height - tile_size + 1, stride):
    for x in range(0, img_width - tile_size + 1, stride):
```

透過重疊切割，原本的六張圖瞬間多出幾張，且跑道會分散在不同的影像中，增加了一些多樣性。

## Roboflow
這是一個免費的標註軟體，切下來我將介紹如何操作。

### 專案建立
首先點選 `Project`，project type 選擇 `Object Detection`，命名好你的專案後就可以開始上船你已經切分好的資料集。

### 手動標註
點擊 `Data Augmentation`，使用手動拉框 (Bounding Box) 工具，將目標物框起來並標上類別。

### 資料增強
標註完之後進入 Generate Version 階段。

1.Train/Test Split 設定為 `Train 80% / Vaild 10% / Test 10%`
2.Augmentation 加入 Filp 、 Rotate

## 從網站抓資料
點選 `How to Upload Custom Weights` 後找到 Setting，點選 `API KEY`，選擇 `Pubilshable API KEY`。

### 扣
```
import os
from dotenv import load_dotenv
from roboflow import Roboflow

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)

workspace_id = "ian-liu-s-workspace"
project_id = "find-airport"

project = rf.workspace(workspace_id).project(project_id)
version = project.version(1)
dataset = version.download("yolov8")

print("Finish")
```

執行之後系統就會自動在電腦下載並壓縮成一個標準的 YOLOv8 資料集，裡面的座標數值也進行了正規化。

## 小結
總之今天我們實做了標註資料以及抓取至電腦端，不過更值得注意的是未來我們的資料取得方式以及操作軟體，畢竟資料的品質是影響模型好壞的關鍵因素，所以在未來會繼續尋找好的軟體跟 Data Soures，那我們明天見。
