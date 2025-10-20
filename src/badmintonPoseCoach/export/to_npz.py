# src/badmintonPoseCoach/export/to_npz.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import json


# -----------------------------
# Temporal interpolation helpers
# -----------------------------
def _interp_1d_time(arr: np.ndarray) -> np.ndarray:
    """
    Nội suy 1D theo thời gian cho vector arr (T,).
    - Không NaN -> trả về như cũ
    - Có NaN -> nội suy tuyến tính giữa các điểm valid; ngoài biên giữ giá trị biên
    - Tất cả NaN -> trả về zeros
    """
    out = arr.astype(np.float32).copy()
    T = out.shape[0]
    mask = ~np.isnan(out)
    if mask.any():
        if (~mask).any():
            xs = np.arange(T, dtype=np.float32)
            out[~mask] = np.interp(xs[~mask], xs[mask], out[mask])
    else:
        out[:] = 0.0
    return out


def impute_pose_and_bbox(
    kpts_tv3: np.ndarray,   # (T, V, 3) float32  [x, y, conf] in pixels
    bbox_t4:  np.ndarray,   # (T, 4)   float32  [x1, y1, x2, y2] in pixels
    W: int,
    H: int,
    max_nan_frame_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Impute NaN theo thời gian cho pose & bbox.
    - Nếu tỷ lệ frame có NaN > max_nan_frame_ratio -> ok=False (nên skip clip).
    - Ngược lại: nội suy theo thời gian và clamp vào [0,W-1]/[0,H-1]; conf clamp [0,1].
    """
    kpts = kpts_tv3.copy()
    bbox = bbox_t4.copy()

    # 1) kiểm tra tỷ lệ frame có NaN (kể cả bbox)
    any_nan_frame = np.any(np.isnan(kpts_tv3), axis=(1, 2)) | np.any(np.isnan(bbox_t4), axis=1)
    if any_nan_frame.mean() > max_nan_frame_ratio:
        return kpts_tv3, bbox_t4, False  # quá nhiều NaN -> bỏ clip

    # 2) impute pose (theo joint & channel)
    T, V, C = kpts.shape
    for v in range(V):
        for c in range(C):  # x, y, conf
            kpts[:, v, c] = _interp_1d_time(kpts[:, v, c])

    # 3) impute bbox từng kênh
    for c in range(4):
        bbox[:, c] = _interp_1d_time(bbox[:, c])

    # 4) clamp vào khung + sắp thứ tự bbox
    # pose
    kpts[..., 0] = np.clip(kpts[..., 0], 0.0, float(W - 1))
    kpts[..., 1] = np.clip(kpts[..., 1], 0.0, float(H - 1))
    kpts[..., 2] = np.clip(kpts[..., 2], 0.0, 1.0)

    # bbox
    bbox[:, 0] = np.clip(bbox[:, 0], 0.0, float(W - 1))
    bbox[:, 2] = np.clip(bbox[:, 2], 0.0, float(W - 1))
    bbox[:, 1] = np.clip(bbox[:, 1], 0.0, float(H - 1))
    bbox[:, 3] = np.clip(bbox[:, 3], 0.0, float(H - 1))
    x1 = np.minimum(bbox[:, 0], bbox[:, 2]); x2 = np.maximum(bbox[:, 0], bbox[:, 2])
    y1 = np.minimum(bbox[:, 1], bbox[:, 3]); y2 = np.maximum(bbox[:, 1], bbox[:, 3])
    bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3] = x1, y1, x2, y2

    return kpts, bbox, True


# -----------------------------
# Unified NPZ saver (pose-only)
# -----------------------------
def save_track_unified_npz_imputed(
    npz_path: str | Path,
    video_path: str | Path,
    meta: Dict[str, Any],          # {"fps","W","H","T_total"}
    track_obj: Dict[str, np.ndarray],   # {"t","kpt","bbox", ...}
    valid_idx: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    max_nan_frame_ratio: float = 0.5,
    to_npz: bool = True,
) -> Optional[str]:
    """
    Lưu 1 file NPZ duy nhất đủ thông tin cho mọi model (RNN/ST-GCN/Transformer):
      - kpts:   (T, V, 3) float32  [x, y, conf] pixel-space
      - bbox:   (T, 4)    float32  [x1, y1, x2, y2]
      - frames: (T,)      int32
      - meta_json: str    JSON (video_path, fps, W, H, track_id, label)

    Impute NaN theo thời gian; nếu > max_nan_frame_ratio frame có NaN -> return None (skip clip).
    """
    t  = track_obj["t"]
    kp = track_obj["kpt"]
    bb = track_obj["bbox"]

    if valid_idx is None:
        valid_idx = np.arange(len(t), dtype=np.int32)

    kp_sel = kp[valid_idx].astype(np.float32)
    bb_sel = bb[valid_idx].astype(np.float32)
    frames = t[valid_idx].astype(np.int32)

    kp_imp, bb_imp, ok = impute_pose_and_bbox(
        kp_sel, bb_sel, int(meta["W"]), int(meta["H"]),
        max_nan_frame_ratio=max_nan_frame_ratio
    )
    if not ok:
        return None

    rec = {
        "kpts":   kp_imp,                # (T, V, 3)
        "bbox":   bb_imp,                # (T, 4)
        "frames": frames,                # (T,)
        "meta_json": json.dumps({
            "video_path": str(video_path),
            "fps": int(meta.get("fps", 0)),
            "W": int(meta.get("W", 0)),
            "H": int(meta.get("H", 0)),
            "track_id": int(track_obj.get("track_id", -1)),
            "label": label
        }, ensure_ascii=False)
    }

    if not to_npz:
        return rec

    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_path), **rec)
    return str(npz_path)


# -----------------------------
# (Optional) Saver không impute
# -----------------------------
def save_track_unified_npz(
    npz_path: str | Path,
    video_path: str | Path,
    meta: Dict[str, Any],
    track_obj: Dict[str, np.ndarray],
    valid_idx: Optional[np.ndarray] = None,
    label: Optional[str] = None,
) -> str:
    """
    Phiên bản đơn giản KHÔNG impute NaN (giữ nguyên dữ liệu).
    Chủ yếu để debug so sánh trước/sau impute.
    """
    t  = track_obj["t"]
    kp = track_obj["kpt"]
    bb = track_obj["bbox"]
    if valid_idx is None:
        valid_idx = np.arange(len(t), dtype=np.int32)

    rec = {
        "kpts":   kp[valid_idx].astype(np.float32),
        "bbox":   bb[valid_idx].astype(np.float32),
        "frames": t[valid_idx].astype(np.int32),
        "meta_json": json.dumps({
            "video_path": str(video_path),
            "fps": int(meta.get("fps", 0)),
            "W": int(meta.get("W", 0)),
            "H": int(meta.get("H", 0)),
            "track_id": int(track_obj.get("track_id", -1)),
            "label": label
        }, ensure_ascii=False)
    }
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_path), **rec)
    return str(npz_path)
