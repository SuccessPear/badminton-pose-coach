from badmintonPoseCoach import logger
from badmintonPoseCoach.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

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

if __name__ == "__main__":
    mp.freeze_support()
    if platform.system() == "Windows":
        mp.set_start_method("spawn", force=True)
    main()