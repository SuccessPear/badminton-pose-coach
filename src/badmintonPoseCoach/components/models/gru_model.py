import torch.nn as nn
from badmintonPoseCoach.entity.config_entity import ModelConfig
import torch

class GRUModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.gru = nn.GRU(
                    input_size=self.config.params_num_joints * self.config.params_channel,
                    hidden_size=self.config.params_hidden,
                    num_layers=self.config.params_layers,
                    dropout=self.config.params_dropout,
                    batch_first=True,
                    bidirectional=self.config.params_bidirectional,
                )

        out_dim = self.config.params_hidden * (2 if self.config.params_bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(0.3),
            nn.Linear(out_dim, self.config.params_num_classes),
        )
    def forward(self, packed):
        # packed: PackedSequence of (B, T, K*3)
        _, hn = self.gru(packed)
        # hn shape: (num_layers * num_directions, B, hidden_size)
        if self.config.params_bidirectional:
            # Take last layer's forward and backward hidden, concat
            h_f = hn[-2]
            h_b = hn[-1]
            h = torch.cat([h_f, h_b], dim=-1)  # (B, 2*hidden)
        else:
            h = hn[-1]  # (B, hidden)
        logits = self.head(h)
        return logits