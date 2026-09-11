"""
Day 23: 純 Python 的 GeoTIFF 讀取層，取代 rasterio。

Windows 的 Smart App Control 會擋掉 rasterio 綁的未簽署 GDAL DLL
（jpeg62.dll、proj_9.dll），整個套件無法 import。但這個專案對 GIS 的需求
其實只有三件事：`像素資料`、`仿射矩陣`、`座標系統代碼`，
tifffile 三件都給得出來，而且是純 Python 沒有任何原生 DLL。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

# Sentinel-2 堆疊順序是 B1, B2, B3, B4, ...
#
# TRUE_COLOR (3, 2, 1) = B4 紅 / B3 綠 / B2 藍，`物理上正確`的真彩色。
# FIRST_THREE (0, 1, 2) = B1 / B2 / B3，索引 0 的 B1 是海岸氣膠波段，
#   原生解析度 60m，重採樣到 10m 網格之後糊成一片，塞進紅色通道會讓整張圖蒙霧。
#
# 但我們的模型是在 FIRST_THREE 的渲染結果上訓練的，改成物理正確的組合反而
# 會讓它離開訓練分布（見 Day 23）。所以`正確`與`該用`在這裡不是同一件事。
SENTINEL2_TRUE_COLOR = (3, 2, 1)
SENTINEL2_FIRST_THREE = (0, 1, 2)


@dataclass(frozen=True)
class GeoRef:
    """像素座標到投影座標的映射，以及該投影的 EPSG 代碼。"""
    transform: np.ndarray   # 2x3 仿射矩陣：[x, y]^T = transform @ [col, row, 1]^T
    epsg: int | None

    def pixel_to_proj(self, col, row):
        """像素 (col, row) -> 投影座標 (x, y)，支援純量與陣列。"""
        pts = np.stack([np.asarray(col, dtype=np.float64),
                        np.asarray(row, dtype=np.float64),
                        np.ones_like(np.asarray(col, dtype=np.float64))], axis=-1)
        out = pts @ self.transform.T
        return out[..., 0], out[..., 1]


def _read_georef(tf: tifffile.TiffFile) -> GeoRef:
    """
    GeoTIFF 有`兩種`存仿射矩陣的方式，都得處理：

      1. ModelTransformation —— 一個完整的 4x4 矩陣（可以表達旋轉）
      2. ModelPixelScale + ModelTiepoint —— 縮放 + 一個對位點（只能表達平移縮放）

    多數正射校正過的影像走第二種，但我們手上這張走第一種。
    """
    meta = tf.geotiff_metadata or {}

    if "ModelTransformation" in meta:
        m = np.asarray(meta["ModelTransformation"], dtype=np.float64).reshape(4, 4)
        transform = np.array([[m[0, 0], m[0, 1], m[0, 3]],
                              [m[1, 0], m[1, 1], m[1, 3]]])
    elif "ModelPixelScale" in meta and "ModelTiepoint" in meta:
        sx, sy = np.asarray(meta["ModelPixelScale"], dtype=np.float64)[:2]
        tie = np.asarray(meta["ModelTiepoint"], dtype=np.float64).reshape(-1, 6)[0]
        i, j, x, y = tie[0], tie[1], tie[3], tie[4]
        # y 方向是負的：影像的 row 往下增加，投影座標的 y 往上增加
        transform = np.array([[sx, 0.0, x - i * sx],
                              [0.0, -sy, y + j * sy]])
    else:
        raise ValueError("GeoTIFF 缺少地理定位標籤（ModelTransformation / ModelPixelScale）")

    epsg = meta.get("ProjectedCSTypeGeoKey") or meta.get("GeographicTypeGeoKey")
    return GeoRef(transform=transform, epsg=int(epsg) if epsg else None)


def read_geotiff(path: str | Path) -> tuple[np.ndarray, GeoRef]:
    """讀出原始波段堆疊 (H, W, N) 與地理定位資訊。"""
    with tifffile.TiffFile(str(path)) as tf:
        arr = tf.asarray()
        georef = _read_georef(tf)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr, georef


def live_bands(stack: np.ndarray) -> list[int]:
    """找出真的有內容的波段（標準差為 0 的是空的填充層）。"""
    flat = stack.reshape(-1, stack.shape[-1])
    return np.where(flat.std(axis=0) > 0)[0].tolist()


def pick_rgb_bands(stack: np.ndarray, bands: tuple[int, int, int] | None) -> tuple[int, int, int]:
    """沒指定波段時，依堆疊的波段數量猜一組合理的。"""
    if bands is not None:
        return bands
    n = stack.shape[-1]
    if n == 1:
        return (0, 0, 0)
    if n == 3:
        return (0, 1, 2)              # 一般 RGB GeoTIFF
    return SENTINEL2_TRUE_COLOR       # 多波段，當成 Sentinel-2 堆疊


def to_uint8_rgb(stack: np.ndarray,
                 bands: tuple[int, int, int] | None = None,
                 stretch: str = "percentile",
                 percentiles: tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    """
    把指定的三個波段轉成 8-bit RGB。

    stretch="max"        —— 除以全域最大值（Day 18 的做法）
    stretch="percentile" —— 用 2% / 98% 百分位數拉伸

    百分位數才是對的。這張圖上有大片`雲`，雲是整張圖最亮的東西，
    除以全域最大值等於讓雲決定整張圖的曝光，地面全被壓暗。
    """
    idx = list(pick_rgb_bands(stack, bands))
    img = stack[..., idx].astype(np.float32)

    if stretch == "max":
        hi = float(img.max()) or 1.0
        img = img / hi * 255.0
    elif stretch == "percentile":
        lo, hi = np.percentile(img, percentiles)
        if hi <= lo:
            hi = lo + 1.0
        img = (img - lo) / (hi - lo) * 255.0
    else:
        raise ValueError(f"未知的 stretch 模式: {stretch}")

    return np.clip(img, 0, 255).astype(np.uint8)
