from badmintonPoseCoach.constants import *
from badmintonPoseCoach.utils.common import *
from badmintonPoseCoach.entity.config_entity import *

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            data_url=config.data_url,
            target_path=config.target_path,
            manifest_path=config.manifest_path,
        )
        return data_ingestion_config

    def get_dataset_preprocessing_config(self) -> DatasetPreprocessingConfig:
        config = self.config.dataset_preprocessing
        params = self.params.dataset_preprocessing

        create_directories([config.root_dir])

        dataset_preprocessing_config = DatasetPreprocessingConfig(
            root_dir = Path(config.root_dir),
            model_name = config.model_name,
            manifest_path = Path(config.manifest_path),
            overlay_subdir = Path(config.overlay_subdir),
            params_fps = params.fps,
            params_conf = params.conf,
            params_iou = params.iou,
            params_imgsz = params.imgsz,
            params_maxdet = params.maxdet,
            params_valid_ratio_start = params.valid_ratio_start,
            params_valid_ratio_end = params.valid_ratio_end,
            params_crop_l = params.crop_l,
            params_crop_r = params.crop_r,
            params_crop_t = params.crop_t,
            params_crop_b = params.crop_b,
            params_presence_in_roi_min = params.presence_in_roi_min,
            params_kpt_thr = params.kpt_thr,
            params_ema_alpha = params.ema_alpha,
            params_idle_active_thr = params.idle_active_thr,
            params_idle_mean_speed_thr = params.idle_mean_speed_thr,
            params_win_len = params.win_len,
            params_win_stride = params.win_stride,
            params_save_overlay= params.save_overlay,
            params_max_nan_frame_ratio = params.max_nan_frame_ratio,
        )
        return dataset_preprocessing_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_model_name=params.model_name,
        )
        return prepare_base_model_config

    def get_model_config(self) -> ModelConfig:
        params = None
        if self.params.prepare_base_model.model_name == "gru":
            params = self.params.prepare_base_model.gru
        model_config = ModelConfig(
            params_hidden=params.hidden,
            params_model_name=params.model_name,
            params_num_classes=params.num_classes,
            params_layers=params.layers,
            params_dropout=params.dropout,
            params_num_joints=params.num_joints,
            params_channel=params.channel,
            params_bidirectional=params.bidirectional,
        )
        return model_config

    def get_training_config(self) -> TrainingConfig:
        prepare_base_model_config = self.config.prepare_base_model
        training_config = self.config.training
        params = self.params.training

        create_directories([training_config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(training_config.root_dir),
            trained_model_path=Path(training_config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model_config.updated_base_model_path),
            training_data=Path(training_config.training_data),
            checkpoint_dir=Path(training_config.checkpoint_dir),
            params_epochs=params.epochs,
            params_batch_size=params.batch_size,
            params_device=params.device,
            params_step_size=params.step_size,
            params_lr=params.lr,
            params_gamma=params.gamma,
            params_use_amp=params.use_amp,
        )
        return training_config