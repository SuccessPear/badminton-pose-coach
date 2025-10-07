from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from badmintonPoseCoach.entity.config_entity import DatasetPreprocessingConfig

from badmintonPoseCoach.components.detection.yolo_pose_tracker import track_video
from badmintonPoseCoach.components.selection.actor_selector import select_actor_once_roi
from badmintonPoseCoach.components.selection.roi import slice_by_valid_ratio
from badmintonPoseCoach.components.visualize.overlay import overlay_video  # optional
from badmintonPoseCoach.export.to_npz import save_track_unified_npz_imputed

class DataPreprocessing:
    """
    - Đọc manifest (train/val/test) ở manifest_path
    - Với từng video:
        YOLO-Pose + ByteTrack -> tracks
        ROI + [10%,90%] + idle -> chọn 1 actor duy nhất
        Impute NaN theo thời gian (bỏ nếu >50% frame NaN)
        Lưu NPZ unified: kpts (T,V,3), bbox (T,4), frames (T,), meta_json
    - Lưu về artifacts/data_preprocess/<split>/<stem>.npz
    - (optional) Lưu overlay để kiểm tra (artifacts/data_preprocess/overlays/<split>/*_overlay.mp4)
    """

    def __init__(self, cfg: DatasetPreprocessingConfig):
        self.cfg = cfg

    # ---------- helpers ----------
    def _ensure_dirs(self):
        root = Path(self.cfg.root_dir)
        for split in ("train", "val", "test"):
            (root / split).mkdir(parents=True, exist_ok=True)
            if self.cfg.params_save_overlay:
                (root / self.cfg.overlay_subdir / split).mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> Dict[str, List[Dict[str, str]]]:
        man = json.loads(Path(self.cfg.manifest_path).read_text(encoding="utf-8"))
        assert all(k in man for k in ("train", "val", "test")), "manifest missing splits train/val/test"
        return man

    def _output_paths(self, video_path: str, split: str) -> Tuple[Path, Optional[Path]]:
        path = Path(video_path)
        stem = path.stem
        out_npz = Path(self.cfg.root_dir) / split / f"{path.parts[-2]}_{stem}.npz"
        out_overlay = None
        if self.cfg.params_save_overlay:
            out_overlay = Path(self.cfg.root_dir) / self.cfg.overlay_subdir / split / f"{path.parts[-2]}_{stem}_overlay.mp4"
        return out_npz, out_overlay

    def _select_actor(self, tracks: dict, meta: dict):
        """
        Gọi selector (ROI + valid time + idle) để lấy đúng 1 actor cho toàn video.
        Trả về (chosen_id, info, roi_rect).
        """
        params = {
            "valid_ratio": (self.cfg.params_valid_ratio_start, self.cfg.params_valid_ratio_end),
            "presence_in_roi_min": self.cfg.params_presence_in_roi_min,
            "kpt_thr": self.cfg.params_kpt_thr,
            "ema_alpha": self.cfg.params_ema_alpha,
            "idle_active_thr": self.cfg.params_idle_active_thr,
            "idle_mean_speed_thr": self.cfg.params_idle_mean_speed_thr,
            "win_len": self.cfg.params_win_len,
            "win_stride": self.cfg.params_win_stride,
            "crop": (self.cfg.params_crop_l, self.cfg.params_crop_r, self.cfg.params_crop_t, self.cfg.params_crop_b),
        }
        chosen_id, info, roi = select_actor_once_roi(tracks, meta, params)
        return chosen_id, info, roi

    # ---------- core processing ----------
    def process_one(self, video_path: str, split: str, label: Optional[str] = None) -> Optional[str]:
        """
        Xử lý 1 video -> lưu .npz. Trả về đường dẫn npz (str) hoặc None nếu skip.
        """
        out_npz, out_overlay = self._output_paths(video_path, split)

        # 1) tracking/pose
        tracks, meta = track_video(
            self.cfg.model_name, video_path,
            conf=self.cfg.params_conf, iou=self.cfg.params_iou, imgsz=self.cfg.params_imgsz,
            max_det=self.cfg.params_maxdet,
            tracker_yaml="bytetrack.yaml"
        )

        # 2) chọn đúng 1 actor
        if len(tracks) > 1:
            chosen_id, info, roi = self._select_actor(tracks, meta)
            if chosen_id is None:
                print(f"[SKIP] {video_path} | reason={info.get('reason') if isinstance(info, dict) else info}")
                return None
        else:
            chosen_id = 1

        # 3) slice theo valid_ratio (10%..90%)
        obj = tracks[chosen_id].copy()
        obj["track_id"] = chosen_id
        idx_valid, _, _ = slice_by_valid_ratio(obj, meta["T_total"], (self.cfg.params_valid_ratio_start, self.cfg.params_valid_ratio_end))
        if idx_valid.size < 3:
            print(f"[SKIP] {video_path} | not enough frames in valid window")
            return None

        # 4) save unified npz (impute NaN; skip nếu >50% NaN)
        saved = save_track_unified_npz_imputed(
            out_npz, video_path, meta, obj, valid_idx=idx_valid,
            label=label,
            max_nan_frame_ratio=self.cfg.params_max_nan_frame_ratio
        )
        if saved is None:
            print(f"[SKIP] {video_path} | too many NaN frames (> {self.cfg.params_max_nan_frame_ratio*100:.0f}%)")
            return None

        # 5) (optional) overlay để kiểm tra
        if self.cfg.params_save_overlay and out_overlay is not None:
            try:
                overlay_video(
                    video_path, out_overlay, tracks, chosen_id,
                    score_thr=self.cfg.params_kpt_thr, show=False,
                    fps_override=meta["fps"], W=meta["W"], H=meta["H"],
                    valid_ratio=(self.cfg.params_valid_ratio_start, self.cfg.params_valid_ratio_end), roi=roi
                )
            except Exception as e:
                print(f"[WARN] overlay failed for {video_path}: {e}")

        return str(out_npz)

    def run_split(self, split: str) -> Dict[str, List[str]]:
        """
        Chạy cho 1 split ('train'|'val'|'test'): trả về dict {'ok': [...], 'skip': [...]}
        """
        assert split in ("train", "val", "test")
        man = self._load_manifest()
        items = man[split]
        self._ensure_dirs()

        ok_paths: List[str] = []
        skip_list: List[str] = []

        # Tiện: import tqdm nếu có
        try:
            from tqdm import tqdm
            it = tqdm(items, desc=f"Preprocess {split}", unit="video")
        except Exception:
            it = items

        for rec in it:
            video_path = rec["path"]
            #video_path = "data/Badminton_Strorke_Video/backhand_drive/backhand_drive (4).mp4"
            label = rec.get("label")
            try:
                out = self.process_one(video_path, split, label=label)
                if out is not None:
                    ok_paths.append(out)
                else:
                    skip_list.append(video_path)
            except Exception as e:
                print(f"[ERROR] {video_path}: {e}")
                skip_list.append(video_path)

        # ghi summary mỗi split
        summary_path = Path(self.cfg.root_dir) / f"{split}_summary.json"
        summary = {"split": split, "ok": ok_paths, "skip": skip_list, "config": asdict(self.cfg)}
        #summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {split}: saved={len(ok_paths)} | skipped={len(skip_list)} -> {summary_path}")
        return {"ok": ok_paths, "skip": skip_list}

    def run_all(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Chạy cả train/val/test theo manifest. Trả về dict kết quả mỗi split.
        """
        results = {}
        for split in ("train", "val", "test"):
            results[split] = self.run_split(split)
        # lưu index tổng
        index_path = Path(self.cfg.root_dir) / "index.json"
        index_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote index -> {index_path}")
        return results