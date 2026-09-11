"""Day 23: 把畫框的邏輯從 Tests/ 搬出來，讓管線與測試共用同一套。"""
from __future__ import annotations

import cv2
import numpy as np

RED = (60, 60, 235)      # 合併前的原始碎框
GREEN = (90, 220, 90)    # 由多個碎片合併而成
ORANGE = (40, 150, 245)  # 單一框（沒有被合併，或被共線性檢查剔除）


def draw_boxes(img_bgr: np.ndarray, boxes, title: str | None = None) -> np.ndarray:
    """boxes 是 (points, color, label) 的序列。回傳畫好的 BGR 影像。"""
    canvas = img_bgr.copy()
    for pts, color, label in boxes:
        pts = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, color, 3, cv2.LINE_AA)
        x, y = pts[0][0]
        cv2.putText(canvas, label, (int(x), max(int(y) - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if title is None:
        return canvas
    bar = np.zeros((44, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def detections_to_boxes(detections):
    """把 PipelineResult.detections 轉成 draw_boxes 吃的格式。"""
    boxes = []
    for d in detections:
        multi = d.n_fragments > 1
        label = f"merged x{d.n_fragments} {d.score:.2f}" if multi else f"{d.score:.2f}"
        boxes.append((d.points, GREEN if multi else ORANGE, label))
    return boxes


def hstack_panels(panels: list[np.ndarray], gap: int = 8) -> np.ndarray:
    """把多張等高的圖橫向拼起來，中間留白線。"""
    if not panels:
        raise ValueError("沒有可拼接的面板")
    sep = np.full((panels[0].shape[0], gap, 3), 255, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    return out
