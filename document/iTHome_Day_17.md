我們終於要開始時做`遙測視覺（Remote Sensing Computer Vision`領域中最核心、也是最具挑戰性的技術升級，在昨天的理論篇中，我們探討了「邊界框如何抽象化代表真實世界物件」
今天我們要讓 AI 「角度感知」，並且在最新世代的硬體上徹底釋放算力。

## 傳統水平框 (HBB) 的幾何
在我們過去幾天的實作中，我們使用的都是最經典的`「水平邊界框 (Horizontal Bounding Box, HBB)」`，在傳統 YOLO 模型中，一個物件被定義為 B = (x_c, y_c, w, h)，代表中心點座標與寬高，這在日常生活中非常管用——辨識路上的汽車、行人都沒問題，因為這些物件在相機視角中大多是直立或水平的，但在遙測衛星圖的世界裡可能不能這樣做，因為機場跑道通常是為了迎合當地盛行風向而建造的，所以在正北朝上的衛星圖中，跑道幾乎都是傾斜的（例如 45 度角），當模型試圖用一個「不允許旋轉的正方形」去包住一條「細長的斜向跑道」時，會發生`背景雜訊比失衡 (Background Noise Imbalance)`，根據幾何學計算，當一個長寬比為 10:1 的物體以 45 度角傾斜時，包覆它的最小水平方框內部，會有超過 80% 的面積是背景（草地、農田、民宅），只有不到 20% 是真正的跑道，這導致模型在訓練時陷入一種狀況，它分不清到底「灰色的柏油路徑」是機場，還是「旁邊那大片的綠色草地」才是機場，這正是傳統物件偵測應用於遙測時，`假陽性（False Positives）`極高的根本原因，同時也會導致昨天的中心點座標定位發生嚴重偏移。

## 旋轉邊界框 (OBB)
為了解決這個幾何缺陷，學界提出了 `OBB (Oriented Bounding Box)` 的概念，它在原本的參數中加入了一個角度變數：OBB = (x_c, y_c, w, h, \theta)。這個 `\theta（Theta）`允許框框自由旋轉，完美平行於跑道的走向，將背景雜訊降至最低，在 YOLOv8-OBB 的底層，損失函數（Loss Function）的計算也變得極度複雜，傳統的 IoU (交集區間比) 無法直接計算旋轉框的重疊率，因此YOLOv8 採用了 `ProbIoU (Probabilistic IoU)` 的技術，將兩個旋轉的矩形視為`「二維高斯分佈（2D Gaussian Distributions）」`，透過計算兩個分佈的`巴氏距離（Bhattacharyya Distance）`來評估框框的精準度。

## 標註資料的奇蹟 
從 Polygon 到 OBB訓練 OBB 模型最大的門檻，通常是`「資料集格式」`，如果當初標註人員只畫了水平方框（只給 5 個數字），那這組資料就永遠無法訓練出旋轉模型，還好我們在 find-airport-1 資料集的 labels/train 中發現了高階標註格式：Plaintext0 0.690625 0.265625 0.7 0.271875 0.7078125 0.2734375 0.7078125 ... (略)
這是一長串由無數個 (x, y) 組成的`多邊形分割 (Polygon Segmentation) `點雲，這些點描出跑道不規則的輪廓，Ultralytics YOLO 的框架極度現代化，當我們在訓練時呼叫 yolov8n-obb.pt 模型，系統內部會觸發 OpenCV 的 `cv2.minAreaRect() `演算法。它會在背景自動計算出能將這些多邊形緊密包圍的「最小外接旋轉矩形」，並自動轉換為模型需要的 (x, y, w, h, \theta) 格式。

## OBB 訓練腳本實作與效能分析
當底層驅動的鴻溝被填平後，程式碼的實作反而簡單，我們在 training/train_obb.py 中寫下了這段扣：

```
from pathlib import Path
from ultralytics import YOLO

def train_obb_model():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    yaml_path = project_root / "find-airport-1" / "data.yaml"

    model = YOLO("yolov8n-obb.pt") 

    results = model.train(
        data=str(yaml_path),
        epochs=50,             # OBB 學習角度較費時，設定 50 epochs 讓其收斂
        imgsz=640,
        batch=16,
        device="0",            
        project=str(project_root / "runs" / "train"),
        name="airport_obb_v1"
    )

if __name__ == "__main__":
    train_obb_model()
```

TensorBoard 吐出了令人瞠目結舌的訓練結果，訓練時間 (Training Time) 50 個 Epoch 僅耗時 0.006 小時 (約 21 秒)，在過去使用 CPU 或舊款 GPU 時，這可能需要數個小時，推論速度 (Inference Speed)為驚人的 2.6ms，這代表模型每秒可處理近 400 張高解析度衛星圖片，完全具備處理現代衛星星系（Satellite Constellations）每日產生的 TB 級資料串流的實力，準確度 (Precision) 達到 0.854。這證明了 OBB 模型確實剔除了背景草地的雜訊，精準捕捉到了機場跑道的核心幾何特徵。

## 小結
我們今天成功誕生了能夠畫出「傾斜框」的進階視覺模型，但我們遇到一個問題，OpenCV 根本無法直接讀取龐大的 GeoTIFF 衛星原圖。
明天我們將結合 Day 15 用到的 rasterio ，寫出一套能無縫解析多光譜影像、並直接餵給 OBB 模型的腳本，那我們明天見。