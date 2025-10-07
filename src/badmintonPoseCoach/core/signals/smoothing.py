# src/badmintonPoseCoach/core/signals/smoothing.py
import numpy as np

def ema_filter(x: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Exponential Moving Average (EMA) smoothing along the time axis.
    x: np.ndarray of shape (T, ...) - temporal sequence (e.g. (T,V,2))
    alpha: smoothing factor in [0,1]
           higher alpha → more smoothing → slower response
    """
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for t in range(1, x.shape[0]):
        y[t] = alpha * y[t-1] + (1 - alpha) * x[t]
    return y


def median_filter(x: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Simple temporal median filter (window size k, odd number).
    Applies independently per coordinate dimension.
    """
    assert k >= 1 and k % 2 == 1, "window size k must be odd"
    T = x.shape[0]
    pad = k // 2
    out = np.empty_like(x, dtype=np.float32)
    for t in range(T):
        t1 = max(0, t - pad)
        t2 = min(T, t + pad + 1)
        out[t] = np.median(x[t1:t2], axis=0)
    return out


def smooth_keypoints(kpts_tv3: np.ndarray, method: str = "ema", **kwargs) -> np.ndarray:
    """
    Convenience wrapper for smoothing keypoints sequence.
    kpts_tv3: (T, V, 3)
    method: 'ema' or 'median'
    kwargs: parameters for filter (alpha or k)
    """
    k = kpts_tv3.copy()
    if method == "ema":
        k[..., :2] = ema_filter(k[..., :2], alpha=kwargs.get("alpha", 0.6))
    elif method == "median":
        k[..., :2] = median_filter(k[..., :2], k=kwargs.get("k", 3))
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    return k
