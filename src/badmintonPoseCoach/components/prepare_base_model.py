import torch

from badmintonPoseCoach.components.models.stgcn import STGCN
from badmintonPoseCoach.entity.config_entity import PrepareBaseModelConfig
from badmintonPoseCoach.components.models.gru_model import GRUModel
from badmintonPoseCoach.components.models.stgcn import STGCN

class PrepareBaseModel:
    def __init__(self, prepare_base_model_config: PrepareBaseModelConfig, model_config):
        self.model = None
        self.prepare_base_model_config = prepare_base_model_config
        self.model_config = model_config

    def get_base_model(self):
        if self.prepare_base_model_config.params_model_name == "gru":
            self.model = GRUModel(config=self.model_config)
        elif self.prepare_base_model_config.params_model_name == "stgcn":
            self.model = STGCN(config=self.model_config)
            print(type(self.model))

        self.save_model(self.prepare_base_model_config.updated_base_model_path, self.model)
        return self.model

    @staticmethod
    def save_model(path, model):
        torch.save(model, path)