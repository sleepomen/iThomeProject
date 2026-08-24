大家好我是來自北區的一名高中生，可以叫我Vessel，目前二升三準備特殊選材，這是我第一次參加 iTHome 鐵人賽，之所以選擇 Software Development 是因為我平常喜歡做全端專案，其實我是個社會組，再跨領域這塊可以說是非常之大，這次的參賽就當作是嘗試，希望能夠順利完賽。

之所以主題訂下 AI Model with GIS 是因為之前有做過相關主題的 Project ，不過那時的 with AI 部分只有串接 LLM ，所以這次我加入了 YOLO/Pytorch 作為 AI 的部分，最後的系統會是以 Web 的 方式呈現。

先簡單說明一下在之後我希望達到的效果，首先我會希望以 TASA 的開源衛星影像資料做為資料集，訓練完模型後做分析，想提供更多地理資訊上的細節或是產出更多值得應用的層面，例如人口資料活動分析或是異常飛行物偵測以及預測自然/人為災害影響程度與範圍，前端介面會希望有許多 Data Board 來呈現資料，同時有資料分析的功能，。

接下來是系統架構，我選擇以 FasAPI/Python 做為後端，前端使用 Reacr + Leaflet.js/OpenSreetMap ，資料庫選擇 PostgreSQL 以及它的擴充套件 PostGIS 讓他分析地理資料，AI Model 會選擇YOLO/Pytorch，最後用 Docker 作為運行環境，架構圖會長得像以下:
[ 資料源 ]
   |
   |-- TASA 開源衛星影像
   |
   v

[ AI 模型與空間處理 ]
   |
   |-- YOLO + PyTorch (影像物件偵測 -> 輸出像素座標)
   |
   v
   |
   |-- Rasterio (像素座標轉換 -> 輸出真實世界經緯度 WGS84)
   |
   v

[ 後端與資料庫 ]
   |
   |-- PostgreSQL + PostGIS (儲存空間幾何資料)
   |      ^
   |      | (讀寫資料)
   |      v
   |-- FastAPI (封裝 AI 模型與資料庫，提供 RESTful API)
   |
   v
   | (JSON / GeoJSON 資料流)
   |
   v

[ 前端 Data Board ]
   |
   |-- React (UI 介面與狀態管理)
   |
   |-- Leaflet.js + OpenStreetMap (底圖渲染與空間資料視覺化)


==============================================================
以上環境皆運行於 [ Docker ] 容器中

基本上在明天就會開始基礎環境建設，也歡迎大家指點我的不足，未來可能會變成一個 DevLog ，可能都在 Fine Tuning Model 、 Debug 還有串接來串接去的苦海中，就請大家當作看一個高中生與 LLM 雞同鴨講的做 Project 以及學技術心得，不過想要呈現的方式跟最後的效果甚至應用層面都有可能變動，就讓我們期待明天的結果。