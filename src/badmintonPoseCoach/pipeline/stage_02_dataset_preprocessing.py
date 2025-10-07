from badmintonPoseCoach.config.configuration import ConfigurationManager
from badmintonPoseCoach.components.dataset_preprocessing import DataPreprocessing
from badmintonPoseCoach import logger

STAGE_NAME = "Dataset Preprocessing stage"


class DatasetPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        dataset_preprocessing_config = config.get_dataset_preprocessing_config()
        dataset_preprocessing = DataPreprocessing(dataset_preprocessing_config)
        dataset_preprocessing.run_all()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} start <<<<<<<<")
        obj = DatasetPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============")
    except Exception as e:
        logger.exception(e)
        raise e