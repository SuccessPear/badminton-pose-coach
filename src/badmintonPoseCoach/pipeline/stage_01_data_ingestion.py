from badmintonPoseCoach.config.configuration import ConfigurationManager
from badmintonPoseCoach.components.data_ingestion import DataIngestion
from badmintonPoseCoach import logger

STAGE_NAME = "Data ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.mirror_and_save_json()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} start <<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============")
    except Exception as e:
        logger.exception(e)
        raise e