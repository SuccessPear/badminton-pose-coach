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
        params = self.params.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            raw_data_path = Path(config.raw_data_path),
            processed_data_path = Path(config.processed_data_path),
            params_fps = params.fps,
            params_conf = params.conf,
            params_keypoint_extraction_model = params.keypoint_extraction_model,
        )
        return data_ingestion_config

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