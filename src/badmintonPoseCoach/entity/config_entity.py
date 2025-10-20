from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_url: Path
    target_path: Path
    manifest_path: Path

@dataclass(frozen=True)
class DatasetPreprocessingConfig:
    root_dir: Path
    model_name: str
    manifest_path: Path
    overlay_subdir: Path
    params_fps: float
    params_conf: float
    params_iou: float
    params_imgsz: int
    params_maxdet: int
    params_valid_ratio_start: float
    params_valid_ratio_end: float
    params_crop_l: float
    params_crop_r: float
    params_crop_t: float
    params_crop_b: float
    params_presence_in_roi_min: float
    params_kpt_thr: float
    params_ema_alpha: float
    params_idle_active_thr: float
    params_idle_mean_speed_thr: float
    params_win_len: int
    params_win_stride: int
    params_save_overlay: bool
    params_max_nan_frame_ratio: float

@dataclass(frozen=True)  # you cant add another element here because frozen = True
class PrepareBaseModelConfig:
    root_dir: Path
    updated_base_model_path: Path
    params_model_name: float

@dataclass(frozen=True)
class GRUModelConfig:
    params_model_name: str
    params_num_classes: int
    params_hidden: int
    params_layers: int
    params_dropout: int
    params_num_joints: int
    params_channel: int
    params_bidirectional: bool

@dataclass(frozen=True)
class STGCNConfig:
    params_model_name: str
    params_in_channels: int
    params_num_class: int
    params_num_point: int
    params_num_person: int
    params_dropout: float
    params_t_kernel: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    checkpoint_dir: Path
    params_model_name: str
    params_epochs: int
    params_batch_size: int
    params_device: str
    params_lr: float
    params_step_size: int
    params_gamma: float
    params_use_amp: bool
