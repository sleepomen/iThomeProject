昨天我們完成了推論視覺化，今天我們回到理論：當模型畫出了框，我們該如何客觀評估它到底學得好不好？

## 核心評估指標拆解
評估目標偵測模型，不能只看簡單的正確率，必須看以下三大指標：Precision（精準率）： 模型「說是機場的框」裡面，有多少真的是機場？$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$Recall（召回率）： 全圖真正的機場裡，模型抓到了多少？$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$mAP@0.5（平均平均精準度）： IoU 門檻設為 $0.5$ 時，Precision-Recall 曲線下的面積（最核心的綜合指標）。

## 診斷過擬合（Overfitting）現象
回顧我們之前產出的 `results.png` 與 `confusion_matrix.png`：

訓練集 Loss 一直降，但驗證集 Loss 上升/震盪： 代表模型開始「背答案」，只記住訓練集那幾張圖的細節。

False Positives (FP=11) 過高： 模型極度敏感，把許多類似跑道的地景（如高速公路）誤判為機場。

這就是標準的 Overfitting。

## 三種過擬合防禦方法
在遙測影像小資料集下，解決過擬合有三個黃金招式：

`資料增強（Data Augmentation）`：
利用旋轉（Random Rotation）、翻轉（Flip）、Mosaic 拼貼，讓模型每次看到的機場角度與背景都不同。

`加入負樣本（Negative Samples）`：
在訓練集中加入「沒有機場」的衛星圖（只有農田、大海、城市），訓練模型學會「說不」，降低 FP 誤報率。

`早停機制（Early Stopping）`：
當驗證集 Loss 連續數個 Epoch 不再下降時，自動停止訓練，防止模型過度擬合。

## 小結
今天我們搞懂了評估模型的 key metrics，也找到了 Day 10 模型的過擬合病因。

明天我們將進入實作篇：`調整超參數（Hyperparameter Tuning）`與增強設定，重新訓練出更高精準度、低誤報率的 YOLOv8 機場偵測模型，那我們明天見。