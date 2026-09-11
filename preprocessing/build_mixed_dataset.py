"""
Day 22: 把「整圖」與「切片」兩個資料集合併成一個混合資料集。

Day 20 的交叉驗證證明了兩件事：
  - 整圖訓練的模型在切片上是 0.000（看不見半截跑道）
  - 切片訓練的模型在整圖上 mAP50 只剩 0.099（忘了全局視野）

兩個模型各自只在自己的領域裡work，所以這裡把兩份資料餵給同一個模型，
讓它同時看過「一整座機場」與「一截柏油路」。

關鍵不是把檔案倒在一起就好，而是`比例`。
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHOLE_ROOT = ROOT / "Find-airport-1"
TILED_ROOT = ROOT / "find-airport-tiled"
DST_ROOT = ROOT / "find-airport-mixed"

SPLITS = ("train", "valid", "test")
BALANCE_SPLIT = "train"   # 只有訓練集需要重取樣，驗證/測試集要保持原樣才有可比性


def count_instances(label_dir: Path) -> tuple[int, int]:
    """回傳 (檔案數, 標註實例數)。"""
    files = sorted(label_dir.glob("*.txt"))
    total = sum(
        len([l for l in f.read_text().strip().splitlines() if l.strip()])
        for f in files
    )
    return len(files), total


def copy_split(src_root: Path, split: str, prefix: str, repeat: int = 1) -> int:
    """把一個 split 複製過去，repeat > 1 時做過取樣（複製多份，靠 augmentation 拉開差異）。"""
    src_img = src_root / split / "images"
    src_lbl = src_root / split / "labels"
    dst_img = DST_ROOT / split / "images"
    dst_lbl = DST_ROOT / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    written = 0
    for img_path in sorted(src_img.glob("*.jpg")):
        lbl_path = src_lbl / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        for k in range(repeat):
            # 過取樣的副本用 _rep{k} 區隔，避免檔名撞在一起
            suffix = "" if k == 0 else f"_rep{k}"
            name = f"{prefix}_{img_path.stem}{suffix}"
            shutil.copy2(img_path, dst_img / f"{name}.jpg")
            shutil.copy2(lbl_path, dst_lbl / f"{name}.txt")
            written += 1
    return written


def write_yaml() -> Path:
    yaml_path = DST_ROOT / "data.yaml"
    yaml_path.write_text(
        "names:\n- airport\n"
        "nc: 1\n"
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
    )
    return yaml_path


def main():
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

    # 用「標註實例數」而不是「圖片張數」來決定過取樣倍率。
    # 一張整圖裡可能有好幾條跑道，一張切片可能是純背景，
    # 拿張數來平衡會被背景切片灌水。
    _, whole_inst = count_instances(WHOLE_ROOT / BALANCE_SPLIT / "labels")
    _, tiled_inst = count_instances(TILED_ROOT / BALANCE_SPLIT / "labels")
    repeat = max(1, round(tiled_inst / max(whole_inst, 1)))
    print(f"train 實例數: 整圖 {whole_inst} / 切片 {tiled_inst} "
          f"-> 整圖過取樣 x{repeat}")

    for split in SPLITS:
        r = repeat if split == BALANCE_SPLIT else 1
        n_whole = copy_split(WHOLE_ROOT, split, "whole", repeat=r)
        n_tiled = copy_split(TILED_ROOT, split, "tile", repeat=1)

        _, inst = count_instances(DST_ROOT / split / "labels")
        print(f"[{split:5s}] 整圖 {n_whole:4d} + 切片 {n_tiled:4d} "
              f"= {n_whole + n_tiled:4d} 張 / {inst:4d} 實例")

    yaml_path = write_yaml()
    print(f"\ndata.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
