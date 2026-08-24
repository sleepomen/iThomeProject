既然前兩天確定好系統架構以及資料來源後，今天要來正式建立 Docker 環境了。

## WHAT and WHY is DOCKER
簡單來說它是一個**容器**，可以把整個 Project 包在 container 裡面，同時把全部需要的套件一起下載，固定版本環境，在協作的時候也比較不會有我電腦能跑為何你跑不了的問題，而且關閉頁面她還是會繼續運作，主要還是說一次下載套件太香了，還同時給你灌到虛擬環境裡面，真的方便。

## PostGIS
我們就順便講講為甚麼要用 PostGIS 好了，首先 PostgreSQL 只能儲存字串或數字，讓後端無法真正處理所謂的地理資料，但 PostGIS 不一樣，他可以對所謂的地理資料做運算，例如計算經緯度兩地的直線距離，常見的函式例如 ST_DWithin ，同時他也是 pgSQL 的擴充套件，生態可以說是非常好。

## What's going on ?
說說我今天在用的時候遇到的一點小插曲，因為之前做開發都是在我的筆電上，現在是用桌機，剛打開 Docker 發現運行不了，我以為是虛擬化沒開，但我看了一下工作管理員明明就是開的阿，然後想起來之前換筆電的時後也有遇到這種狀況，就必須要去設定一下，開一下 Hyper-V 、虛擬機平台跟支援 W 子系統 Linux 版，然後再下載 WSL 就可以了。

順便說個好笑的，~~我在命名的時候打成 dokcer-compose.yml~~，所以 powershell 報 not found ，害我想很久還有甚麼東西沒用到，~~結果是單純打錯字~~。

## 扣
```
version: '3.8'

services:
  db:
    image: postgis/postgis:15-3.3
    container_name: ithome_project
    restart: always
    environment:
      POSTGRES_USER: vessel
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: gis_data
    ports:
      - "5432:5432"
    volumes:
      - gis_db_data:/var/lib/postgresql/data

volumes:
  gis_db_data:

```

## 小結
總之最後 docker-compose up -d 下去終於看到 start 了，我們的專案終於有了一小步進展，~~怎麼感覺超慢，會不會太 chill~~，總之明天會開始學一下 SNAP 跟研究一下 Sentinel-2 那些資料集的部分，~~反正系統架起來會很快，在資料處理跟模型訓練的部分就稍微多一點 part 吧~~。