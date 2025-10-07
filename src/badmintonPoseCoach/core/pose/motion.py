import numpy as np
from badmintonPoseCoach.core.pose.normalize import hip_center_shoulder_scale
from badmintonPoseCoach.core.signals import ema_filter

# joint weights để tuỳ chọn từ constants
def motion_energy(kpts_tv3, kpt_thr=0.25, ema_alpha=0.6, joint_weights=None):
    """
    kpts_tv3: (T,V,3) in pixels or normalized (tự chịu trách nhiệm normalize trước nếu muốn invariance)
    return: energy, active_ratio, mean_speed
    """
    T = kpts_tv3.shape[0]
    if T <= 2: return 0.0, 0.0, 0.0
    k = hip_center_shoulder_scale(kpts_tv3)     # invariance theo kích thước
    k[..., :2] = ema_filter(k[..., :2], alpha=ema_alpha)
    xy, sc = k[..., :2], k[..., 2]
    vis = sc >= kpt_thr
    d = np.diff(xy, axis=0)                     # (T-1,V,2)
    speed = np.linalg.norm(d, axis=-1)          # (T-1,V)
    vis_mid = vis[1:] & vis[:-1]                # (T-1,V)
    w = np.ones(speed.shape[1], dtype=np.float32)
    if joint_weights is not None:
        for j, wt in joint_weights.items():
            if j < w.size: w[j] = wt
    frame_score = (speed * w[None, :] * vis_mid).sum(axis=1)
    med = np.median(frame_score)
    mad = np.median(np.abs(frame_score - med)) + 1e-8
    thr = max(med + 2.5 * mad, 0.02)
    active_ratio = float((frame_score > thr).mean())
    denom = max(1, vis_mid.sum())
    energy = float((speed * w[None, :] * vis_mid).sum() / denom)
    mean_speed = float(frame_score.mean()) if frame_score.size else 0.0
    return energy, active_ratio, mean_speed
