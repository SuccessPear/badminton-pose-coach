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

def normalize_pose_rnn(pose, W, H, method="skeleton"):
    # pose: (T,K,3)
    if method == "image":
        pose[...,0] /= W
        pose[...,1] /= H
    elif method == "skeleton":
        # pelvis = joint 11,12 in average
        pelvis = pose[:,[11,12],:2].mean(1, keepdims=True)
        pose[...,:2] -= pelvis
        # scale with the shoulder
        shoulder = pose[:,[5,6],:2].mean(1, keepdims=True)
        scale = (pose[:,5,:2]-pose[:,6,:2]).norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pose[...,:2] /= scale[:,None,:]
    return pose

def normalize_stgcn(self, kpts, W, H, mode="root"):
    """
    kpts: (T, V, 3) float32, pixel-space
    W, H: frame width, height
    mode: "image" | "root" | "person"
    """
    L_HIP, R_HIP = 11, 12
    k = kpts.copy().astype(np.float32)
    # 1. scale to [0,1]
    k[..., 0] /= float(W)
    k[..., 1] /= float(H)

    if mode == "image":
        return k
