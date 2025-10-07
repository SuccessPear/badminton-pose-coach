import numpy as np

def interp_1d(arr):
    out = arr.astype(np.float32).copy()
    T = out.shape[0]
    m = ~np.isnan(out)
    if m.any():
        if (~m).any():
            x = np.arange(T, dtype=np.float32)
            out[~m] = np.interp(x[~m], x[m], out[m])
    else:
        out[:] = 0.0
    return out

def impute_sequence_timewise(arr_txk):  # arr: (T, K)
    out = arr_txk.copy()
    for k in range(out.shape[1]):
        out[:, k] = interp_1d(out[:, k])
    return out
