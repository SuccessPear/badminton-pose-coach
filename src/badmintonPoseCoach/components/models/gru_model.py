import torch.nn as nn
from badmintonPoseCoach.entity.config_entity import ModelConfig

class GRUModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.gru = nn.GRU(
                    input_size=self.config.params_hidden * self.config.params_layers,
                    hidden_size=self.config.params_hidden,
                    num_layers=self.config.params_layers,
                    dropout=self.config.params_dropout,
                    batch_first=True,
                    bidirectional=self.config.params_bidirectional,
                ),
        self.fc = nn.Linear(self.config.params_hidden * (2 if self.config.params_bidirectional else 1), self.config.params_num_classes)
    def forward(self, x):
        # x: [B,T,C,J] -> [B,T,C*J]
        B,T,C,J = x.shape
        x = x.reshape(B,T,C*J)
        out,_ = self.gru(x)
        last = out[:,-1,:]
        return self.fc(last)