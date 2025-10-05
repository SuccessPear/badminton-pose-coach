from badmintonPoseCoach import logger
from badmintonPoseCoach.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from badmintonPoseCoach.pipeline.stage_03_prepare_base_model import PrepareBaseModelPipeline
from badmintonPoseCoach.pipeline.stage_04_model_training import TrainingModelPipeline

import multiprocessing as mp
import platform

def run_stage(stage_name: str, PipelineCls):
    logger.info(f">>> stage {stage_name} started. <<<")
    try:
        obj = PipelineCls()
        obj.main()
        logger.info(f">>> stage {stage_name} finished. <<<")
    except Exception as e:
        logger.exception(e)
        raise

def main():
    run_stage("Data Ingestion", DataIngestionTrainingPipeline)
    #run_stage("Prepare Base Model", PrepareBaseModelPipeline)
    #run_stage("Training Model", TrainingModelPipeline)

if __name__ == "__main__":
    mp.freeze_support()
    if platform.system() == "Windows":
        mp.set_start_method("spawn", force=True)
    main()