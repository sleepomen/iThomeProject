"""
Day 23: 端到端機場偵測管線。

把 Day 18 到 Day 22 散在各個 Tests/ 腳本裡的步驟串成一條：

    GeoTIFF -> 波段選擇 -> 動態範圍拉伸 -> CLAHE -> SAHI 切片推論
            -> Day 21 幾何合併 -> Detection 清單

之前這些邏輯寫死在 `Tests/test_nmm_postprocess.py` 的 `load_image()` 裡，
正式邏輯躺在測試檔案中，誰要用都得 import 一支 test，這裡把它搬回來。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from inference.geotiff_io import (
    SENTINEL2_FIRST_THREE, GeoRef, read_geotiff, to_uint8_rgb,
)
from inference.merge_fragments import merge_predictions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs/train/airport_obb_mixed/weights/best.pt"


@dataclass
class Detection:
    """一個偵測結果。points 是 OBB 的四個角點（像素座標）。"""
    points: np.ndarray
    score: float
    n_fragments: int
    angle: float
    length: float
    width: float

    @property
    def aspect(self) -> float:
        return self.length / max(self.width, 1e-6)

    @property
    def center(self) -> tuple[float, float]:
        c = self.points.mean(axis=0)
        return float(c[0]), float(c[1])


@dataclass
class PipelineResult:
    image: np.ndarray                 # 前處理後的 RGB，畫圖與除錯用
    georef: GeoRef
    detections: list[Detection] = field(default_factory=list)
    raw_count: int = 0                # 幾何合併`之前`的框數


def enhance(rgb: np.ndarray, clip_limit: float = 3.5,
            tile_grid: int = 8) -> np.ndarray:
    """
    Day 18 的 CLAHE 局部打光，只作用在 LAB 的 L（亮度）通道。

    直接對 RGB 三通道各做一次 CLAHE 會把色彩關係打亂，
    轉到 LAB 只動亮度，色度 a/b 保持不變。
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


class AirportDetector:
    """把權重與推論參數綁在一起，模型只在第一次呼叫時載入。"""

    def __init__(self,
                 weights: str | Path = DEFAULT_WEIGHTS,
                 slice_size: int = 320,
                 overlap: float = 0.2,
                 confidence: float = 0.25,
                 device: str = "cuda:0",
                 bands: tuple[int, int, int] | None = SENTINEL2_FIRST_THREE,
                 stretch: str = "max"):
        self.weights = Path(weights)
        self.slice_size = slice_size
        self.overlap = overlap
        self.confidence = confidence
        self.device = device
        # 預設值刻意`對齊訓練資料的長相`而不是物理正確的真彩色。
        # 換成 (3, 2, 1) + 百分位數拉伸，人眼看起來清楚得多，
        # 但模型會在城市紋理與海岸線上狂開火（Day 23 實測 1 框 -> 13 框）。
        self.bands = bands
        self.stretch = stretch
        self._model = None

    @property
    def model(self):
        # 延遲載入：建立 detector 很便宜，真的要推論才把權重搬上 GPU
        if self._model is None:
            if not self.weights.exists():
                raise FileNotFoundError(f"找不到權重: {self.weights}")
            self._model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(self.weights),
                confidence_threshold=self.confidence,
                device=self.device,
            )
        return self._model

    def preprocess(self, tif_path: str | Path) -> tuple[np.ndarray, GeoRef]:
        stack, georef = read_geotiff(tif_path)
        rgb = to_uint8_rgb(stack, bands=self.bands, stretch=self.stretch)
        return enhance(rgb), georef

    def detect(self, tif_path: str | Path) -> PipelineResult:
        img, georef = self.preprocess(tif_path)

        result = get_sliced_prediction(
            img, self.model,
            slice_height=self.slice_size, slice_width=self.slice_size,
            overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap,
            verbose=0,
        )
        preds = result.object_prediction_list

        detections = [
            Detection(points=m["points"], score=m["score"],
                      n_fragments=len(m["members"]), angle=m["angle"],
                      length=m["length"], width=m["width"])
            for m in merge_predictions(preds)
        ]
        return PipelineResult(image=img, georef=georef,
                              detections=detections, raw_count=len(preds))
