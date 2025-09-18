from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path
    processed_data_path: Path
    params_fps: int
    params_conf: int
    params_keypoint_extraction_model: str
