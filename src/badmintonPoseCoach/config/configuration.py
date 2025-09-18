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