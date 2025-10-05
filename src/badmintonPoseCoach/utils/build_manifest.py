from pathlib import Path
import json
import random


def build_manifest(
    root_dir: str = "data/Badminton_Strorke_Video",
    out_path: str = "data/Badminton_Strorke_Video/manifest.json",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Tạo file manifest.json từ dataset dạng 'folder = class'.
    Ví dụ:
        data/raw/forehand_clear/*.mp4
        data/raw/backhand_drop/*.mp4
    """
    root = Path(root_dir)
    assert root.exists(), f"Dataset folder not found: {root}"

    random.seed(seed)
    splits = {"train": [], "val": [], "test": []}

    # --- Duyệt từng class ---
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        videos = sorted(cls_dir.glob("*.mp4"))
        if shuffle:
            random.shuffle(videos)
        n = len(videos)
        if n == 0:
            continue

        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        splits["train"] += [{"path": str(v), "label": label} for v in videos[:n_train]]
        splits["val"] += [{"path": str(v), "label": label} for v in videos[n_train:n_train + n_val]]
        splits["test"] += [{"path": str(v), "label": label} for v in videos[n_train + n_val:]]

    # --- Ghi file ---
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] manifest saved to {out}")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} samples")


if __name__ == "__main__":
    build_manifest()
