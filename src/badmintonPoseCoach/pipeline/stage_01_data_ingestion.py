from badmintonPoseCoach.config.configuration import ConfigurationManager
from badmintonPoseCoach.components.data_ingestion import DataIngestion
from badmintonPoseCoach import logger
from badmintonPoseCoach.utils.build_manifest import build_manifest

STAGE_NAME = "Data ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        build_manifest(root_dir=data_ingestion_config.target_path,
                       out_path=data_ingestion_config.manifest_path)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} start <<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============")
    except Exception as e:
        logger.exception(e)
        raise e