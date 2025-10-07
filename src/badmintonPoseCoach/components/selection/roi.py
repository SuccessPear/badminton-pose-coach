from __future__ import annotations
import numpy as np
from typing import Tuple, Dict


def slice_by_valid_ratio(obj: Dict[str, np.ndarray],
                         T_total: int,
                         valid_ratio: Tuple[float, float] = (0.1, 0.9)) -> Tuple[np.ndarray, int, int]:
    """
    Lấy ra index frame hợp lệ theo valid_ratio (ví dụ 10–90%)
    obj: {"t": (T,), ...}
    """
    t = np.asarray(obj["t"], dtype=np.int32)
    t1 = int(valid_ratio[0] * T_total)
    t2 = int(valid_ratio[1] * T_total)
    mask = (t >= t1) & (t < t2)
    idx = np.where(mask)[0]
    return idx, t1, t2


def roi_rect(W: int, H: int,
             crop_left: float, crop_right: float,
             crop_top: float, crop_bottom: float) -> Tuple[int, int, int, int]:
    """
    Trả về (x1, y1, x2, y2) theo tỉ lệ cắt ROI
    """
    x1 = int(W * crop_left)
    x2 = int(W * crop_right)
    y1 = int(H * crop_top)
    y2 = int(H * crop_bottom)
    return x1, y1, x2, y2


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    """
    bbox: (T,4) dạng [x1,y1,x2,y2]
    return: (T,2) tâm bbox
    """
    cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
    cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
    return np.stack([cx, cy], axis=-1)


# def track_presence_in_roi(obj: Dict[str, np.ndarray],
#                           roi: Tuple[int, int, int, int],
#                           idx_valid: np.ndarray | None = None) -> float:
#     """
#     Tính tỷ lệ frame mà người đó nằm trong ROI (bbox-center-based).
#     """
#     if obj["bbox"].size == 0:
#         return 0.0
#     x1, y1, x2, y2 = roi
#     bb = obj["bbox"][idx_valid] if idx_valid is not None else obj["bbox"]
#     centers = bbox_center(bb)
#     cx, cy = centers[:, 0], centers[:, 1]
#     inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
#     return float(inside.mean())

def track_presence_in_roi(obj, roi, valid_idx):
    """
    Compute fraction of frames (within valid_idx) where bbox center lies inside ROI.
    """
    if valid_idx.size == 0: return 0.0
    b = obj["bbox"][valid_idx]
    cx = (b[:,0] + b[:,2]) * 0.5
    cy = (b[:,1] + b[:,3]) * 0.5
    x1,y1,x2,y2 = roi
    inside = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)
    return float(inside.mean())


def crop_frame(frame: np.ndarray,
               roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Cắt frame (numpy image) theo ROI (x1,y1,x2,y2)
    """
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]
