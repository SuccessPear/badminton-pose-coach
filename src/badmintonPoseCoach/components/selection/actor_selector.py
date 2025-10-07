from __future__ import annotations
from typing import Dict, Tuple, Any
import numpy as np

from badmintonPoseCoach.components.selection.roi import slice_by_valid_ratio, roi_rect, track_presence_in_roi
from badmintonPoseCoach.core.pose.motion import motion_energy

def select_actor_once_roi(
    tracks: Dict[int, Dict[str, np.ndarray]],
    meta: Dict[str, Any],
    params: Dict[str, Any] | Any,
) -> Tuple[int | None, Dict[str, Any], Tuple[int,int,int,int]]:
    """
    Chọn duy nhất 1 actor cho toàn bộ video dựa trên:
      - valid_ratio (bỏ 10% đầu/cuối)
      - ROI (crop_left/right/top/bottom)
      - presence_in_roi_min
      - motion_energy + idle filters

    params có thể là dict hoặc dataclass có các field sau:
      valid_ratio: (float,float)
      crop: (l,r,t,b)
      presence_in_roi_min: float
      kpt_thr: float
      ema_alpha: float
      idle_active_thr: float
      idle_mean_speed_thr: float
      win_len, win_stride (không dùng ở bản này: global)
    """
    # lấy tham số
    valid_ratio           = tuple(params.get("valid_ratio", (0.10, 0.90)))
    crop                  = tuple(params.get("crop", (0.15, 0.85, 0.30, 0.70)))
    presence_in_roi_min   = float(params.get("presence_in_roi_min", 0.5))
    kpt_thr               = float(params.get("kpt_thr", 0.25))
    ema_alpha             = float(params.get("ema_alpha", 0.6))
    idle_active_thr       = float(params.get("idle_active_thr", 0.06))
    idle_mean_speed_thr   = float(params.get("idle_mean_speed_thr", 0.004))

    W, H, T_total = meta["W"], meta["H"], meta["T_total"]
    roi = roi_rect(W, H, *crop)

    candidates = []
    valid_map: Dict[int, np.ndarray] = {}

    # lọc theo thời gian hợp lệ & cư trú trong ROI
    for tid, obj in tracks.items():
        idx_valid, _, _ = slice_by_valid_ratio(obj, T_total, valid_ratio)
        if idx_valid.size < 3:
            continue
        pres = track_presence_in_roi(obj, roi, idx_valid)
        if pres >= presence_in_roi_min:
            candidates.append(int(tid))
            valid_map[int(tid)] = idx_valid

    if not candidates:
        return None, {"reason": "no_candidate_in_roi"}, roi

    # chấm điểm global (không sliding window để giữ ổn định id)
    best_tid, best_score = None, -1e9
    details = {}
    for tid in candidates:
        obj = tracks[tid]
        idx = valid_map[tid]
        k_win = obj["kpt"][idx]  # (T,V,3)
        E, act, ms = motion_energy(k_win, kpt_thr=kpt_thr, ema_alpha=ema_alpha, joint_weights=None)
        # idle filter
        if (act < idle_active_thr) and (ms < idle_mean_speed_thr):
            continue
        score = E + 0.1 * act
        details[int(tid)] = {"energy": E, "active_ratio": act, "mean_speed": ms, "score": score}
        if score > best_score:
            best_score, best_tid = score, tid

    if best_tid is None:
        return None, {"reason": "no_track_survived_idle_filter"}, roi

    return int(best_tid), {"mode": "global", "details": details}, roi
