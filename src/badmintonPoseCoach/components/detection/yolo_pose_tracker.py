from __future__ import annotations
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any

def track_video(model_path: str,
                video_path: str,
                conf: float = 0.15,
                iou: float = 0.5,
                imgsz: int = 960,
                max_det: int = 10,
                tracker_yaml: str = "bytetrack.yaml") -> tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    Run YOLO-Pose + ByteTrack on a video and return tracks dict + meta.
    tracks[tid] = {"t": (T,), "kpt": (T,V,3), "bbox": (T,4)} in frame order.
    meta = {"fps","W","H","T_total"}
    """
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
        classes=[0],  # person
    )

    tracks: Dict[int, Dict[str, list]] = {}
    fps = None; W = None; H = None
    t_global = -1

    for r in results:
        t_global += 1
        if fps is None:
            # một số backend không trả fps trong results; ta lấy 15 làm fallback
            fps = 15
            H, W = r.orig_shape

        if r.boxes is None or r.keypoints is None:
            continue

        ids = getattr(r.boxes, "id", None)
        if ids is None:
            continue
        ids = ids.int().cpu().tolist()

        kxy = r.keypoints.xy.cpu().numpy()                  # (N,V,2)
        ksc = getattr(r.keypoints, "conf", None)
        if ksc is None:
            ksc = np.ones(kxy.shape[:2], dtype=np.float32)  # (N,V)
        else:
            ksc = ksc.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()                   # (N,4)

        for i, tid in enumerate(ids):
            tid = int(tid)
            kp = np.concatenate([kxy[i], ksc[i][..., None]], axis=-1).astype(np.float32)  # (V,3)
            tracks.setdefault(tid, {"t": [], "kpt": [], "bbox": []})
            tracks[tid]["t"].append(t_global)
            tracks[tid]["kpt"].append(kp)
            tracks[tid]["bbox"].append(xyxy[i].astype(np.float32))

    # stack & sort by time
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for tid, obj in tracks.items():
        order = np.argsort(np.asarray(obj["t"]))
        tarr = np.asarray(obj["t"])[order]
        karr = np.stack(obj["kpt"], axis=0)[order] if obj["kpt"] else np.zeros((0,0,3), np.float32)
        barr = np.stack(obj["bbox"], axis=0)[order] if obj["bbox"] else np.zeros((0,4), np.float32)
        out[tid] = {"t": tarr, "kpt": karr, "bbox": barr}

    meta = {"fps": int(round(fps or 15)), "W": int(W or 0), "H": int(H or 0), "T_total": int(t_global + 1)}
    return out, meta
