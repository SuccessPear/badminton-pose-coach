import numpy as np
from badmintonPoseCoach.constants import L_SH, R_SH, L_HIP, R_HIP

def hip_center_shoulder_scale(kpts_tv3):
    k = kpts_tv3.copy()
    xy = k[..., :2]
    hip_mid = (xy[:, L_HIP:L_HIP+1, :] + xy[:, R_HIP:R_HIP+1, :]) / 2.0
    xy = xy - hip_mid
    sh = np.linalg.norm(xy[:, L_SH, :] - xy[:, R_SH, :], axis=-1)
    s = np.median(sh[sh > 1e-3])
    if s > 0: xy = xy / s
    k[..., :2] = xy
    return k
