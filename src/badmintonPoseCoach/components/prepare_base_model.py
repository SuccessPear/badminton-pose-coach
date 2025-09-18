import torch
from badmintonPoseCoach.entity.config_entity import PrepareBaseModelConfig, ModelConfig
from badmintonPoseCoach.components.models.gru_model import GRUModel

class PrepareBaseModel:
    def __init__(self, prepare_base_model_config: PrepareBaseModelConfig, model_config: ModelConfig):
        self.model = None
        self.prepare_base_model_config = prepare_base_model_config
        self.model_config = model_config

    def get_base_model(self):
        if self.prepare_base_model_config.params_model_name == "gru":
            self.model = GRUModel(config=self.model_config)
            self.save_model(self.prepare_base_model_config.updated_base_model_path, self.model)
        return self.model

    @staticmethod
    def save_model(path, model):
        torch.save(model.state_dict(), path)