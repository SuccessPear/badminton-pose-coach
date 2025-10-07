# badminton_actor_select_v6_roi.py
# ------------------------------------------------------------------------------
# Pipeline (ROI hard-crop version):
# 1) YOLO-Pose + ByteTrack -> collect tracks (all frames)
# 2) VALID TIME: only use [10% .. 90%] frames for selection
# 3) ROI: only consider people whose bbox centers lie inside ROI
#         ROI = crop 15% left/right, 30% top/bottom  (center 70% x 40% area)
# 4) Score motion energy (skeleton-centered + EMA), drop idle tracks
# 5) Pick ONE global actor for whole video (no switching)
# 6) Overlay: shade outside ROI & head/tail; chosen=green, others=red
# 7) (Optional) Save chosen track to JSON / ST-GCN npz
# ------------------------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import cv2
from ultralytics import YOLO

# ==============================
# ==== EDIT THESE PARAMETERS ===
# ==============================
VIDEO_PATH   = "../data/Badminton_Strorke_Video/backhand_drive/backhand_drive (1).mp4"
MODEL_PATH   = "yolo11n-pose.pt"
OUT_PATH     = "overlay.mp4"
JSON_OUT     = "chosen_track.json"   # optional (toggle in main)

CONF   = 0.20
IOU    = 0.50
IMGSZ  = 1280
MAXDET = 12
TRACKER_YAML = "./bytetrack_loose.yaml"   # or "bytetrack.yaml"

# VALID TIME (drop head/tail)
VALID_RATIO         = (0.10, 0.90)    # use only 10%..90% of frames for selection

# ROI crop (keep only center)
CROP_L = 0.25   # cut 15% left
CROP_R = 0.75   # cut 15% right
CROP_T = 0.20   # cut 30% top
CROP_B = 0.75   # cut 30% bottom
PRESENCE_IN_ROI_MIN = 0.50  # require >=50% of a track's frames (inside valid time) to have center inside ROI

# Motion scoring / idle filter
KPT_THR             = 0.25
EMA_ALPHA           = 0.6
IDLE_ACTIVE_THR     = 0.06   # <6% active frames => idle
IDLE_MEAN_SPEED_THR = 0.004  # very small mean speed => idle

# Windowing (optional). If None, we score globally over valid window.
WIN_LEN     = None  # e.g., 64
WIN_STRIDE  = None  # e.g., 32

SHOW = True
# ==============================

# COCO-17 (Ultralytics)
# 0-nose,1-eyeL,2-eyeR,3-earL,4-earR, 5-LShoulder,6-RShoulder,7-LElbow,8-RElbow,
# 9-LWrist,10-RWrist, 11-LHip,12-RHip,13-LKnee,14-RKnee,15-LAnkle,16-RAnkle
COCO_EDGES = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6), (11,12), (5,11),(6,12), (0,5),(0,6)
]
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12

# joints weight (focus on racket arm movements)
JOINT_WEIGHTS = { L_WR:1.6, R_WR:1.6, L_EL:1.4, R_EL:1.4, L_SH:1.2, R_SH:1.2 }

# ---------------- Drawing ----------------
def draw_skeleton(frame, kpts_px, color, score_thr=0.25, radius=3, thickness=2):
    K = kpts_px.shape[0]
    for a, b in COCO_EDGES:
        if a < K and b < K:
            xa, ya, sa = kpts_px[a]
            xb, yb, sb = kpts_px[b]
            if sa >= score_thr and sb >= score_thr:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, thickness)
    for i in range(K):
        x, y, s = kpts_px[i]
        if s >= score_thr:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)

# ---------------- Tracking & Pose ----------------
def track_video(model_path, video_path, conf=0.15, iou=0.5, imgsz=960,
                max_det=10, tracker_yaml="bytetrack.yaml"):
    model = YOLO(model_path)
    results = model.track(
        source=video_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=True,
        verbose=False,
        persist=True,
        max_det=max_det,
        tracker="bytetrack.yaml",
        classes=[0],   # person
    )
    tracks = {}
    fps = None; W = None; H = None
    t_global = -1
    for r in results:
        t_global += 1
        if fps is None:
            fps = 15
            H, W = r.orig_shape
        if r.boxes is None or r.keypoints is None:
            continue
        ids = getattr(r.boxes, "id", None)
        if ids is None:  # tracker not ready
            continue
        ids = ids.int().cpu().tolist()
        kxy = r.keypoints.xy.cpu().numpy()            # (N,V,2)
        ksc = getattr(r.keypoints, "conf", None)
        if ksc is None:
            ksc = np.ones(kxy.shape[:2], dtype=np.float32)
        else:
            ksc = ksc.cpu().numpy()                   # (N,V)
        xyxy = r.boxes.xyxy.cpu().numpy()             # (N,4)
        for i, tid in enumerate(ids):
            tid = int(tid)
            kp = np.concatenate([kxy[i], ksc[i][..., None]], axis=-1).astype(np.float32)  # (V,3)
            tracks.setdefault(tid, {"t": [], "kpt": [], "bbox": []})
            tracks[tid]["t"].append(t_global)
            tracks[tid]["kpt"].append(kp)
            tracks[tid]["bbox"].append(xyxy[i].astype(np.float32))
    for tid, obj in tracks.items():
        order = np.argsort(np.asarray(obj["t"]))
        obj["t"]   = np.asarray(obj["t"])[order]
        obj["kpt"] = np.stack(obj["kpt"], axis=0)[order] if len(obj["kpt"]) else np.zeros((0,0,3), np.float32)
        obj["bbox"]= np.stack(obj["bbox"],axis=0)[order] if len(obj["bbox"]) else np.zeros((0,4), np.float32)
    meta = {"fps": int(round(fps or 15)), "W": int(W), "H": int(H), "T_total": int(t_global + 1)}
    return tracks, meta

# ---------------- Utils: normalize & denoise ----------------
def ema_filter(x, alpha=EMA_ALPHA):
    y = x.copy()
    if y.shape[0] <= 1: return y
    for v in range(y.shape[1]):
        prev = y[0, v]
        for t in range(1, y.shape[0]):
            prev = alpha * prev + (1 - alpha) * y[t, v]
            y[t, v] = prev
    return y

def skeleton_center_scale_norm(kpts_tv3):
    """
    Center by hip-mid, scale by shoulder width (median over time).
    Input: (T,V,3) -> Output: (T,V,3) normalized.
    """
    k = kpts_tv3.copy()
    xy = k[..., :2]
    hip_mid = (xy[:, L_HIP:L_HIP+1, :] + xy[:, R_HIP:R_HIP+1, :]) / 2.0  # (T,1,2)
    xy = xy - hip_mid
    sh_dist = np.linalg.norm(xy[:, L_SH, :] - xy[:, R_SH, :], axis=-1)    # (T,)
    s = np.median(sh_dist[sh_dist > 1e-3])
    if s > 0:
        xy = xy / s
    k[..., :2] = xy
    return k

# ---------------- Motion energy ----------------
def motion_scores(kpts_tv3, kpt_thr=KPT_THR):
    """
    Return (energy, active_ratio, mean_speed) for a sequence (T,V,3).
    - normalize by skeleton_center_scale_norm
    - EMA denoise
    - speed = ||∆xy||, joint-weighted, confidence mask
    """
    T = kpts_tv3.shape[0]
    if T <= 2:
        return 0.0, 0.0, 0.0
    k = skeleton_center_scale_norm(kpts_tv3)
    k[..., :2] = ema_filter(k[..., :2], alpha=EMA_ALPHA)
    xy, sc = k[..., :2], k[..., 2]
    vis = sc >= kpt_thr

    d = np.diff(xy, axis=0)                  # (T-1,V,2)
    speed = np.linalg.norm(d, axis=-1)       # (T-1,V)
    vis_mid = vis[1:] & vis[:-1]             # (T-1,V)

    w = np.ones(speed.shape[1], dtype=np.float32)
    for j, wt in JOINT_WEIGHTS.items():
        if j < w.size: w[j] = wt

    frame_score = (speed * w[None, :] * vis_mid).sum(axis=1)  # (T-1,)
    med = np.median(frame_score)
    mad = np.median(np.abs(frame_score - med)) + 1e-8
    thr = max(med + 2.5 * mad, 0.02)
    active_ratio = float((frame_score > thr).mean())

    denom = max(1, vis_mid.sum())
    energy = float((speed * w[None, :] * vis_mid).sum() / denom)
    mean_speed = float(frame_score.mean()) if frame_score.size else 0.0
    return energy, active_ratio, mean_speed

# ---------------- Time & ROI helpers ----------------
def slice_by_valid_ratio(obj, T_total, valid_ratio):
    """Return indices within [t_start, t_end) according to valid_ratio."""
    t = obj["t"]
    t_start = int(valid_ratio[0] * T_total)
    t_end   = int(valid_ratio[1] * T_total)
    mask = (t >= t_start) & (t < t_end)
    idx = np.where(mask)[0]
    return idx, t_start, t_end

def roi_rect(W, H, l=CROP_L, r=CROP_R, t=CROP_T, b=CROP_B):
    """Return integer ROI rectangle (x1,y1,x2,y2)."""
    x1 = int(l * W); x2 = int(r * W)
    y1 = int(t * H); y2 = int(b * H)
    return x1, y1, x2, y2

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

# ---------------- Actor selection (ONE global winner) ----------------
def select_actor_once_roi(tracks, meta,
                          valid_ratio=VALID_RATIO,
                          presence_in_roi_min=PRESENCE_IN_ROI_MIN,
                          kpt_thr=KPT_THR,
                          idle_active_thr=IDLE_ACTIVE_THR,
                          idle_mean_speed_thr=IDLE_MEAN_SPEED_THR,
                          win_len=WIN_LEN, win_stride=WIN_STRIDE):
    """
    Choose ONE global track_id for the whole video using ROI & valid time.
    Steps:
      - valid window [10%..90%]
      - ROI: require presence_in_roi >= threshold
      - compute (energy, active_ratio, mean_speed) over valid window (or windows)
      - idle removal
      - pick best (no switching)
    """
    W, H, T_total = meta["W"], meta["H"], meta["T_total"]
    roi = roi_rect(W, H)

    # Build candidate list by ROI presence
    candidates = []
    track_valid_idx = {}
    for tid, obj in tracks.items():
        idx_valid, _, _ = slice_by_valid_ratio(obj, T_total, valid_ratio)
        if idx_valid.size < 3:  # not enough samples
            continue
        pres = track_presence_in_roi(obj, roi, idx_valid)
        if pres >= presence_in_roi_min:
            candidates.append(int(tid))
            track_valid_idx[tid] = idx_valid

    if not candidates:
        return None, {"reason": "no_candidate_in_roi"}, roi

    # Scoring (global or windowed), idle filter
    votes = {}
    scores_acc = {}
    if win_len and win_stride:
        # windowed voting inside valid time
        t_start = int(valid_ratio[0] * T_total)
        t_end   = int(valid_ratio[1] * T_total)
        windows = []
        t0 = t_start
        while t0 + win_len <= t_end:
            windows.append((t0, t0 + win_len))
            t0 += win_stride
        if not windows:
            windows = [(t_start, t_end)]

        for (w0, w1) in windows:
            best_tid, best_score = None, -1e9
            for tid in candidates:
                obj = tracks[tid]
                t = obj["t"]
                mask = (t >= w0) & (t < w1)
                idx = np.where(mask)[0]
                if idx.size < 3:
                    continue
                k_win = obj["kpt"][idx]
                E, act, ms = motion_scores(k_win, kpt_thr=kpt_thr)
                # idle filter
                if (act < idle_active_thr) and (ms < idle_mean_speed_thr):
                    continue
                score = E + 0.1 * act
                if score > best_score:
                    best_score = score; best_tid = tid
                scores_acc[tid] = scores_acc.get(tid, 0.0) + score
            if best_tid is not None:
                votes[best_tid] = votes.get(best_tid, 0) + 1

        if not votes:
            return None, {"reason": "no_votes_after_idle_filter"}, roi
        top_votes = max(votes.values())
        cands = [tid for tid, v in votes.items() if v == top_votes]
        chosen = cands[0] if len(cands)==1 else max(cands, key=lambda tid: scores_acc.get(tid, 0.0))
        info = {"mode": "windowed", "votes": votes, "scores_acc": scores_acc}
        return int(chosen), info, roi

    else:
        # global scoring over valid window
        best_tid, best_score = None, -1e9
        details = {}
        for tid in candidates:
            obj = tracks[tid]
            idx_valid = track_valid_idx[tid]
            k_win = obj["kpt"][idx_valid]
            E, act, ms = motion_scores(k_win, kpt_thr=kpt_thr)
            if (act < idle_active_thr) and (ms < idle_mean_speed_thr):
                continue
            score = E + 0.1 * act
            details[int(tid)] = {"energy": E, "active_ratio": act, "mean_speed": ms, "score": score}
            if score > best_score:
                best_score = score; best_tid = tid
        if best_tid is None:
            return None, {"reason": "no_track_survived_idle_filter"}, roi
        info = {"mode": "global", "details": details}
        return int(best_tid), info, roi

# ---------------- Overlay (fixed chosen id; shade outside ROI & head/tail) ----------------
def overlay_video(video_path, out_path, tracks, chosen_tid,
                  score_thr=KPT_THR, show=False, fps_override=None, W=None, H=None,
                  valid_ratio=VALID_RATIO, roi=None):
    frame_map = {}
    for tid, obj in tracks.items():
        for t, kp in zip(obj["t"], obj["kpt"]):
            frame_map.setdefault(int(t), {})[int(tid)] = kp
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps0 = cap.get(cv2.CAP_PROP_FPS)
    H = int(H or H0); W = int(W or W0); fps = float(fps_override or (fps0 if fps0 > 0 else 15))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    T_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_start = int(valid_ratio[0] * T_total)
    t_end   = int(valid_ratio[1] * T_total)
    if roi is None:
        roi = roi_rect(W, H)
    x1,y1,x2,y2 = roi

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))

        # shade head/tail
        if t < t_start or t >= t_end:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (W,H), (64,64,64), -1)
            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

        # shade outside ROI
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.rectangle(mask, (x1,y1), (x2,y2), (255,255,255), -1)
        frame = np.where(mask==0, (frame*0.75).astype(frame.dtype), frame)

        # draw ROI box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)

        actors = frame_map.get(t, {})
        for tid, kp in actors.items():
            color = (0,255,0) if (chosen_tid is not None and tid == chosen_tid) else (0,0,255)
            draw_skeleton(frame, kp, color, score_thr)

        cv2.putText(frame, f"t={t} chosen_id={chosen_tid}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        writer.write(frame)
        if show:
            cv2.imshow("overlay", frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == 27: break
        t += 1
    cap.release(); writer.release()
    if show: cv2.destroyAllWindows()

# ---------------- JSON Export (for RNN/ST-GCN later) ----------------
def save_track_to_json(json_path, video_path, meta, track_obj, valid_idx=None, label=None):
    """
    JSON format:
    {
      "video_path": "...",
      "fps": 30, "W": 1280, "H": 720,
      "track_id": 7,
      "frames": [t0, t1, ...],
      "kpts":   [[[x,y,conf], ... V], ... T],
      "bbox":   [[x1,y1,x2,y2], ... T],
      "label":  null
    }
    """
    t  = track_obj["t"]
    kp = track_obj["kpt"]
    bb = track_obj["bbox"]
    if valid_idx is None:
        valid_idx = np.arange(len(t), dtype=np.int32)

    rec = {
        "video_path": str(video_path),
        "fps": int(meta.get("fps", 0)),
        "W": int(meta.get("W", 0)),
        "H": int(meta.get("H", 0)),
        "track_id": int(track_obj.get("track_id", -1)),
        "frames": [int(x) for x in t[valid_idx].tolist()],
        "kpts":   [[[float(x), float(y), float(c)] for (x,y,c) in kp_i] for kp_i in kp[valid_idx]],
        "bbox":   [[float(a) for a in bb_i] for bb_i in bb[valid_idx]],
        "label":  label
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    print(f"[OK] Saved JSON -> {json_path}")

def save_track_as_stgcn_npz(npz_path, track_obj, valid_idx=None, add_velocity=True):
    """
    Produce array X with shape (C,T,V,1), C=3 or 4 (x,y,score[,|v|])
    """
    kp = track_obj["kpt"]
    if valid_idx is None:
        valid_idx = np.arange(kp.shape[0], dtype=np.int32)
    arr = np.transpose(kp[valid_idx], (2,0,1)).astype(np.float32)  # (3,T,V)
    if add_velocity:
        v = np.zeros_like(arr[0])   # (T,V)
        v[1:] = np.linalg.norm(np.diff(arr[:2], axis=1), axis=0)
        arr = np.concatenate([arr, v[None, ...]], axis=0)  # (4,T,V)
    X = arr[..., None]  # (C,T,V,1)
    np.savez_compressed(npz_path, X=X)
    print(f"[OK] Saved ST-GCN npz -> {npz_path}  {X.shape}")

# ==============================
# ============ MAIN ============
# ==============================
if __name__ == "__main__":
    tracks, meta = track_video(
        MODEL_PATH, VIDEO_PATH,
        conf=CONF, iou=IOU, imgsz=IMGSZ,
        max_det=MAXDET, tracker_yaml=TRACKER_YAML
    )
    print(f"[INFO] meta={meta} | tracks={len(tracks)}")

    # 1) Select exactly ONE actor using ROI + valid time (lock for whole video)
    chosen_tid, info, roi = select_actor_once_roi(tracks, meta)
    print(f"[INFO] chosen_id={chosen_tid} | mode={info.get('mode') if isinstance(info, dict) else info} | roi={roi}")

    # 2) Overlay (chosen stays constant)
    overlay_video(
        VIDEO_PATH, OUT_PATH, tracks, chosen_tid,
        score_thr=KPT_THR, show=SHOW,
        fps_override=meta["fps"], W=meta["W"], H=meta["H"],
        valid_ratio=VALID_RATIO, roi=roi
    )
    print(f"[OK] Saved overlay -> {OUT_PATH}")

    # 3) (OPTIONAL) Save JSON/NPZ for training later
    # if chosen_tid is not None:
    #     obj = tracks[chosen_tid].copy()
    #     obj["track_id"] = chosen_tid
    #     idx_valid, _, _ = slice_by_valid_ratio(obj, meta["T_total"], VALID_RATIO)
    #     save_track_to_json(JSON_OUT, VIDEO_PATH, meta, obj, valid_idx=idx_valid, label=None)
    #     # save_track_as_stgcn_npz("chosen_track_stgcn.npz", obj, valid_idx=idx_valid, add_velocity=True)
