from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Tuple

from badmintonPoseCoach.constants import COCO_EDGES

def draw_skeleton(frame: np.ndarray, kpts_px: np.ndarray,
                  color: tuple[int,int,int], score_thr: float = 0.25,
                  radius: int = 3, thickness: int = 2) -> None:
    """
    kpts_px: (V,3) in pixel (x,y,conf)
    """
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

def overlay_video(video_path: str,
                  out_path: str,
                  tracks: Dict[int, Dict[str, np.ndarray]],
                  chosen_tid: int | None,
                  score_thr: float = 0.25,
                  show: bool = False,
                  fps_override: float | None = None,
                  W: int | None = None,
                  H: int | None = None,
                  valid_ratio: tuple[float, float] = (0.10, 0.90),
                  roi: Tuple[int,int,int,int] | None = None) -> None:
    """
    Vẽ overlay: shade head/tail + ngoài ROI, tô xanh người được chọn, đỏ những người khác.
    """
    # gom keypoint theo frame
    frame_map: Dict[int, Dict[int, np.ndarray]] = {}
    for tid, obj in tracks.items():
        for t, kp in zip(obj["t"], obj["kpt"]):
            frame_map.setdefault(int(t), {})[int(tid)] = kp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps0 = cap.get(cv2.CAP_PROP_FPS)
    H = int(H or H0); W = int(W or W0)
    fps = float(fps_override or (fps0 if fps0 and fps0 > 0 else 15.0))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    T_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    t_start = int(valid_ratio[0] * T_total)
    t_end   = int(valid_ratio[1] * T_total)
    if roi is None:
        x1, y1, x2, y2 = 0, 0, W, H
    else:
        x1, y1, x2, y2 = roi

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))

        # shade head/tail
        if t < t_start or t >= t_end:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (64, 64, 64), -1)
            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

        # shade outside ROI
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
        frame = np.where(mask == 0, (frame * 0.75).astype(frame.dtype), frame)

        # draw ROI box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # draw skeletons
        actors = frame_map.get(t, {})
        for tid, kp in actors.items():
            color = (0, 255, 0) if (chosen_tid is not None and tid == chosen_tid) else (0, 0, 255)
            draw_skeleton(frame, kp, color, score_thr)

        cv2.putText(frame, f"t={t} chosen_id={chosen_tid}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(frame)
        if show:
            cv2.imshow("overlay", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break
        t += 1

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
