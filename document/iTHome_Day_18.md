YOLO 底層吃的是 OpenCV，而 OpenCV 在這方面就是個「俗人」，它的守備範圍僅限於標準 8-bit 的 JPG 或 PNG。遇到帶有空間座標、16-bit 深度、甚至塞了十幾個光譜波段的 GeoTIFF 原檔，它根本不知道從何讀起，只能兩眼一抹黑當場罷工。為了解決這個硬體與套件的代溝，我們必須請出 Day 15 裝好的 GIS 神器 rasterio，親手把衛星圖「降維打擊」成 OpenCV 看得懂的形狀。

## 影像降維與 16-bit 壓縮
要繞過 YOLO 的輸入層，我們要把遙測影像拆解並重組，用 rasterio 抽出 RGB 三個波段後原本的矩陣形狀是 (3, H, W)，必須先轉置成 OpenCV 的 (H, W, 3) 格式，陷阱在於位元壓縮，真實衛星圖是 16-bit（數值範圍 0~65535），我們需要把它等比例壓回螢幕能顯示的 8-bit（0~255），如果在除以最大值的時候沒有轉成 np.uint8，後面的所有色彩空間轉換會整個亂掉。

## 巨幅影像與 SAHI 
測試腳本但跑出來的結果是一片空白，真實的空拍圖隨便就破萬像素，YOLO 預設的 640x640 視野會把整張地圖進行壓縮，一條原本幾百像素寬的跑道被擠壓成不到 1 個像素的灰色雜點，特徵完全被消失，這時候必須導入期他套件解決：`SAHI (Slicing Aided Hyper Inference)`。

傳統 YOLO 的直出推論會將整張巨幅影像強行 Resize 至 640x640，這導致微小目標的特徵保留度趨近於零，幾乎完全糊掉，但好處是具備全局視野、沒有物件被截斷的風險，且後處理成本極低，SAHI 切片輔助推論採取了完全不同的邏輯，它維持原圖解析度，將原圖裁切成數百張 640x640 的小圖，從而 100% 保留了原始像素細節，但是這個套件也帶來了新的挑戰，我們必須精準設定重疊率（Overlap）以防止目標在邊界被切斷，最後還得仰賴 NMS 或 NMM 演算法將所有切片產生的預測框重新縫合，這也增加了後處理的運算成本。

## CLAHE 局部打光
掛上 SAHI 切片後終端機跑了很久，但結果還是空白，後來發現一個問題，右下角的雲層反光太亮了，當 rasterio 在做全局 16-bit 壓縮時，因為雲層數值頂天，導致其他背光的工業區跟跑道被相對被壓縮，對比度會低到根本分不出來其他附近的物件，更用不到邊緣梯度的卷積神經網路，這讓我們直接加上同為影像處理的套件：`CLAHE (限制對比度自適應直方圖均衡化)`，我們不能直接在 RGB 通道打光，那會讓整張地圖的顏色很不協調，正確的做法是把圖片轉入 LAB 色彩空間，把顏色（A, B）跟亮度（L）剝離開來，我們只針對 L 通道套用 CLAHE 演算法，把圖片切成 8x8 的小網格進行局部直方圖拉伸，最後再縫合回 RGB。這個方法讓我們需要的目標物終於出現了。

## 實作扣

```
Pythonimport cv2
import numpy as np
import rasterio
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def run_hardcore_inference():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "runs" / "train" / "airport_obb_v1-4" / "weights" / "best.pt"
    test_source = project_root / "data" / "raw" / "som_san_air.tif"
    output_dir = project_root / "runs" / "predict" / "sahi_result"

    with rasterio.open(str(test_source)) as src:
        img_array = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
        if img_array.dtype == np.uint16 or img_array.max() > 255:
            img_array = (img_array / img_array.max() * 255).astype(np.uint8)

        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced_lab = cv2.merge((clahe.apply(l), a, b))
        img_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=0.25,  # 嚴格濾除幾何幻覺
        device="cuda:0"
    )

    result = get_sliced_prediction(
        img_bgr,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    result.export_visuals(export_dir=str(output_dir), file_name="sahi_final")
    print(f"Finish")
if __name__ == "__main__":
    run_hardcore_inference()
```

## 小結
剛加上 CLAHE 時我把`信心門檻（Confidence Threshold）`降到了 0.05，結果產出的圖片非常詭異，AI 把雲層邊緣、長條形廠房全部硬猜成目標物，畫出了幾十個重疊的紅色畸形框，這就是 SAHI 切片帶來的副作用，在極低門檻下，`NMS（非極大值抑制）演算法`處理旋轉角度的縫合邏輯會徹底崩潰，最後把門檻拉回 0.25，我們需要的圖片才出現，很好的沿著跑道的邊緣輸出預測框，沒有加進其他的物件，OBB 演算法在這時候生效了，但如果把圖片放大，會發現這框只包住了右上半截，左下半段卻漏掉了，模型在訓練時看的是「擁有航廈與完整跑道的整座機場」，現在推論時被 SAHI 切掉，局部切片裡只剩下一小截灰色的柏油路，特徵不足導致 AI 不敢斷定那是機場，最終被 0.25 的門檻清除，要解決這種大型物件被截斷的問題，單靠推論端已經不夠了，我們必須更加深入，在訓練階段就開始做一些處理，讓 AI 從一開始就習慣看「破碎的特徵」，明天我們準備進入自動化切片訓練資料集，那我們明天見。