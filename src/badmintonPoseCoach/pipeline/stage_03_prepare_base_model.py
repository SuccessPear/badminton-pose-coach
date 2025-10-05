from badmintonPoseCoach.config.configuration import ConfigurationManager
from badmintonPoseCoach.components.prepare_base_model import PrepareBaseModel
from badmintonPoseCoach import logger

STAGE_NAME = "Prepare base model stage"


class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        model_config = config.get_model_config()
        prepare_base_model = PrepareBaseModel(prepare_base_model_config, model_config)
        prepare_base_model.get_base_model()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} start <<<<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============")
    except Exception as e:
        logger.exception(e)
        raise e