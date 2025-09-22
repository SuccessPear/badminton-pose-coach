from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path
    processed_data_path: Path
    params_fps: int
    params_conf: int
    params_keypoint_extraction_model: str

@dataclass(frozen=True)  # you cant add another element here because frozen = True
class PrepareBaseModelConfig:
    root_dir: Path
    updated_base_model_path: Path
    params_model_name: float

@dataclass(frozen=True)
class ModelConfig:
    params_model_name: float
    params_num_classes: int
    params_hidden: int
    params_layers: int
    params_dropout: int
    params_num_joints: int
    params_channel: int
    params_bidirectional: bool

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int