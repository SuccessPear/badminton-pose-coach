from badmintonPoseCoach.config.configuration import ConfigurationManager
from badmintonPoseCoach.components.trainer import Trainer
from badmintonPoseCoach.components.badminton_pose_dataset import BadmintonPoseDataset
from badmintonPoseCoach import logger
from torch.utils.data import Dataset, DataLoader
from badmintonPoseCoach.utils.collate import pack_collate
STAGE_NAME = "Training model stage"


class TrainingModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        train_config = config.get_training_config()
        train_data = BadmintonPoseDataset(train_config, split='train')
        val_data = BadmintonPoseDataset(train_config, split='val')
        test_data = BadmintonPoseDataset(train_config, split='test')

        train_loader = DataLoader(train_data, batch_size=train_config.params_batch_size, collate_fn=pack_collate,
                                  shuffle=True)
        val_loader = DataLoader(val_data, batch_size=train_config.params_batch_size, collate_fn=pack_collate,
                                shuffle=True)
        test_loader = DataLoader(test_data, batch_size=train_config.params_batch_size, collate_fn=pack_collate)

        trainer = Trainer(train_config, train_loader, val_loader, test_loader)
        trainer.fit()
        trainer.test()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} start <<<<<<<<")
        obj = TrainingModelPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============")
    except Exception as e:
        logger.exception(e)
        raise e