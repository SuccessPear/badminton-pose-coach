import os
from badmintonPoseCoach import logger
import os, json, pathlib
from typing import Iterable
from badmintonPoseCoach.utils.common import get_size, extract_keypoints_from_video, is_video_readable
from badmintonPoseCoach.entity.config_entity import DataIngestionConfig
import torch

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def iter_videos(self, root: str) -> Iterable[str]:
        """Yield all video file paths under root recursively."""
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith("mp4"):
                    yield os.path.join(dirpath, fn)

    def mirror_and_save_json(self):
        """
        Walk through input_root/videos/<class>/*.mp4 and save JSON to
        output_root/<class>/<video_stem>.json, preserving the class folder.
        """
        input_root = os.path.abspath(self.config.raw_data_path)
        output_root = os.path.abspath(self.config.processed_data_path)

        for vp in self.iter_videos(input_root):
            # Compute relative path to preserve class folder
            rel = os.path.relpath(vp, input_root)              # e.g., "forehand/clip01.mp4"
            rel_no_ext = os.path.splitext(rel)[0]              # "forehand/clip01"
            out_json_path = os.path.join(output_root, rel_no_ext + ".json")

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

            # Skip if already extracted
            if os.path.exists(out_json_path):
                print(f"[SKIP] Exists: {out_json_path}")
                continue

            if not is_video_readable(vp):
                logger.info(f"[SKIP] Can't read: {rel}")
                continue

            print(f"[EXTRACT] {vp} -> {out_json_path}")
            data = extract_keypoints_from_video(
                video_path=vp,
                model_or_path=self.config.params_keypoint_extraction_model,
                fps_sample=self.config.params_fps,
                conf=self.config.params_conf,
            )
            data["seq"] = self.clean_data(data["seq"])
            if data["seq"] is None:
                continue

            # Attach label from parent directory name (class folder)
            label = pathlib.Path(rel_no_ext).parts[0]  # first folder under input_root
            data["label"] = label
            data["video_relpath"] = rel.replace("\\", "/")

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

    @staticmethod
    def clean_data(seq):
        valid_frames = []
        for i in range(len(seq)):
            if torch.isfinite(torch.tensor(seq[i])).all():
                valid_frames.append(i)

        if len(valid_frames) < int(len(seq)*0.6):
            return None

        return [seq[i] for i in valid_frames]



