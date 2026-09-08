"""
Day 21: 把 SAHI 切片推論吐出來的「同一條跑道被拆成好幾塊」碎框縫回一條。

SAHI 內建的 NMS / NMM / GREEDYNMM 全都以「重疊率」(IOU / IOS) 為判準，
但沿著跑道排開的碎片彼此重疊率是 0，所以內建後處理一個都合併不了。

這裡改用三個幾何判準：
  1. 間距 (gap)   —— 碎片邊界距離夠近才可能屬於同一個目標
  2. 共線 (axis)  —— 用 PCA 對群內所有角點擬合主軸
  3. 穩健剔除     —— 垂直偏移離群的碎片踢出去再重新擬合

最後沿著主軸取投影範圍，重建一個貼著跑道方向的 OBB。
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

# ---------------- 參數 ----------------
GAP_PX = 70.0          # 碎片邊界距離門檻，超過就當成不同目標
OUTLIER_K = 3.0        # 垂直偏移 > K x 中位數 才算離群
OUTLIER_MIN = 60.0     # 垂直偏移絕對下限，避免中位數太小時誤殺
MAX_ITER = 5           # 穩健擬合的最大迭代次數


def obb_polygon(pred) -> Polygon:
    """SAHI 把 OBB 的四個角點塞在 segmentation 裡，撈出來還原成 Polygon。"""
    seg = pred.mask.segmentation[0]
    return Polygon(np.asarray(seg, dtype=np.float64).reshape(-1, 2)).buffer(0)


def cluster_by_gap(polys: list[Polygon], gap: float = GAP_PX) -> list[list[int]]:
    """用 union-find 把「邊界距離 <= gap」的碎片連成同一群（連通分量）。"""
    parent = list(range(len(polys)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].distance(polys[j]) <= gap:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(polys)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]


def _corner_points(polys: list[Polygon], members: list[int]) -> np.ndarray:
    """把群內所有碎片的角點疊成一個 (N, 2) 點雲。"""
    return np.vstack([np.asarray(polys[i].exterior.coords[:-1]) for i in members])


def _fit_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """對點雲做 PCA，回傳 (質心, 主軸單位向量, 法向量)。"""
    mu = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mu, full_matrices=False)
    axis = vt[0]
    normal = np.array([-axis[1], axis[0]])
    return mu, axis, normal


def fit_axis_robust(polys: list[Polygon], members: list[int]) -> tuple:
    """
    擬合主軸，並把垂直偏移離群的碎片剔除後重新擬合。

    跑道是細長物件，真正屬於它的碎片一定貼在同一條線上；
    偏離主軸太遠的多半是誤判（市區、海岸線），順便被這一步濾掉。

    回傳 (保留的成員, 質心, 主軸, 法向量, 被剔除的成員)。
    """
    kept = list(members)
    dropped: list[int] = []

    for _ in range(MAX_ITER):
        mu, axis, normal = _fit_axis(_corner_points(polys, kept))
        if len(kept) <= 2:
            break

        offsets = {
            i: abs((np.asarray(polys[i].centroid.coords[0]) - mu) @ normal)
            for i in kept
        }
        limit = max(OUTLIER_K * float(np.median(list(offsets.values()))), OUTLIER_MIN)
        outliers = [i for i in kept if offsets[i] > limit]
        if not outliers:
            break
        dropped.extend(outliers)
        kept = [i for i in kept if i not in outliers]

    mu, axis, normal = _fit_axis(_corner_points(polys, kept))
    return kept, mu, axis, normal, dropped


def axis_rect(polys: list[Polygon], members: list[int],
              mu: np.ndarray, axis: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """沿主軸取投影範圍，重建一個貼齊跑道方向的 OBB（4 個角點）。"""
    pts = _corner_points(polys, members) - mu
    t, w = pts @ axis, pts @ normal
    corners = [
        mu + axis * a + normal * b
        for a, b in ((t.min(), w.min()), (t.max(), w.min()),
                     (t.max(), w.max()), (t.min(), w.max()))
    ]
    return np.asarray(corners)


def merge_predictions(preds, gap: float = GAP_PX) -> list[dict]:
    """
    主流程：碎框 -> 分群 -> 穩健擬合主軸 -> 重建 OBB。

    每個結果是 {"points", "score", "members", "dropped", "angle", "length", "width"}。
    被剔除的離群碎片會各自以原本的框回到結果裡，不會憑空消失。
    """
    polys = [obb_polygon(p) for p in preds]
    scores = [float(p.score.value) for p in preds]
    merged: list[dict] = []

    def single(idx: int) -> dict:
        pts = np.asarray(polys[idx].exterior.coords[:-1])
        mu, axis, normal = _fit_axis(pts)
        proj_t, proj_w = (pts - mu) @ axis, (pts - mu) @ normal
        return {
            "points": pts, "score": scores[idx], "members": [idx], "dropped": [],
            "angle": float(np.degrees(np.arctan2(axis[1], axis[0])) % 180),
            "length": float(np.ptp(proj_t)), "width": float(np.ptp(proj_w)),
        }

    for group in cluster_by_gap(polys, gap):
        if len(group) == 1:
            merged.append(single(group[0]))
            continue

        kept, mu, axis, normal, dropped = fit_axis_robust(polys, group)
        pts = axis_rect(polys, kept, mu, axis, normal)
        proj_t = (_corner_points(polys, kept) - mu) @ axis
        proj_w = (_corner_points(polys, kept) - mu) @ normal
        merged.append({
            "points": pts,
            "score": max(scores[i] for i in kept),
            "members": kept, "dropped": dropped,
            "angle": float(np.degrees(np.arctan2(axis[1], axis[0])) % 180),
            "length": float(np.ptp(proj_t)), "width": float(np.ptp(proj_w)),
        })
        # 離群碎片保留原框，交給使用者自己判斷是不是誤判
        merged.extend(single(i) for i in dropped)

    return sorted(merged, key=lambda m: -m["score"])
