經過前幾天的資料處理，我們今天終於到正式開始模型的訓練，今天的內容會以第一版訓練出來的模型進行數據跟影相的分析。

那當然了，每次開始前總會有些東西需要先處理好。

## 釋放顯卡算力
在實作之前我們遇到硬體限制的問題，5070 Ti 用全新的 `Blackwell` 架構，算力為 `sm_120`，而標準版 PyTorch (CUDA 12.4) 僅支援至上一代的 `sm_90`，導致執行時噴出一堆 error。

所以要先砍掉預設舊版套件，改裝支援 50 系列的 pytorch nightily build，成功掛載之後我們的 batch size 就可以開大一點，增加訓練收斂的速度。

## 執行腳本的扣
```
import os
from pathlib import Path
from ultralytics import YOLO

def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    yaml_path = project_root / "find-airport-1" / "data.yaml"

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=50,                  # 訓練 50 輪
        imgsz=640,                  # 圖片尺寸 640x640
        batch=16,                   # 5070 Ti 16GB 顯存直接開到 16
        workers=4,                  # 資料載入執行緒
        name="sentinel_airport_v1",  # 輸出專案名稱
        device=0,                   # 使用 GPU 0
        project=str(project_root / "runs" / "detect") # 統一將訓練結果輸出至根目錄的 runs/
    )

    print("Finish")

if __name__ == "__main__":
    main()
```

## 自動化推論腳本的扣
訓練完成後，我們撰寫推論腳本進行驗證，為了避免 YOLO 因為重複執行而自動遞增資料夾名稱，我們加入了動態搜尋最新修改時間（st_mtime）的防錯機制：

```
from pathlib import Path
from ultralytics import YOLO

def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    detect_dir = project_root / "runs" / "detect"
    best_pts = list(detect_dir.glob("**/weights/best.pt"))

    if not best_pts:
        raise FileNotFoundError(f"best.pt not found")

    latest_best_pt = max(best_pts, key=lambda p: p.stat().st_mtime)

    print(f"{latest_best_pt.relative_to(project_root)}")
    model = YOLO(str(latest_best_pt))

    test_source = project_root / "find-airport-1" / "test" / "images"

    if not test_source.exists():
        raise FileNotFoundError(f"Folder not found：{test_source}")

    results = model.predict(
        source=str(test_source),
        conf=0.25,
        save=True,
        project=str(project_root / "runs" / "predict"),
        name="airport_test_result"
    )

    output_dir = project_root / "runs" / "predict" / "airport_test_result"
    print(f"Finsih")

if __name__ == "__main__":
    main()
```

## 第一次訓練結果跟分析
訓練完後打開自動新增的 `run\` 資料夾，就可以看到模型訓練結果跟許多數據。

### 學習曲線/指標判斷 
#### 訓練集 Loss
`train/box_loss` 跟 `train/cls_loss` 呈現穩定的下滑趨勢，證明神經網路確實在學習影像目標物的特徵。

#### 驗證集 Loss / mAP
`val\cls_loss` 在地 30 輪出現震盪，最終的 `mAP@0.5` 維持在約 0.013 ~ 0.020 的低位。

#### 小結
可以看到訊連結果似乎不是那麼好，我猜是因為資料集的數量問題導致過擬合現象，雖然之前有擴充資料集的數量，但驗證集的數量不是那麼多，導致驗證指標波動較大。

## 實際推論圖像觀察
看過測試集衛星圖後，我觀察到了幾個現象:

### 成功偵測
模型確實準確定位出了桃園機場的跑道區域，並給出了 0.59 與 0.38 的置信度分數。

### 重疊框現象
在圖像中看到目標物區域出現兩個高度重疊的藍色邊界框，這個重疊框的出現，印證了神經網路在推論時多個網格同時發起預測的特性，這也關於到明天其中一個內容，`NMS (非極大值抑制)。

## 小結
今天我們完成了環境修復、模組化訓練與自動化推論的流程，同時也遇到了極小資料集帶來的 overfitting。

既然模型會輸出多個重疊框，我們該如何透過演算法將其優化為單一精準框？明天我們將來講講NMS與置信度門檻調校，那我們明天見。

