from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from badmintonPoseCoach.entity.config_entity import STGCNConfig

# bạn đã có các hằng số này trong constants
try:
    from badmintonPoseCoach.constants import COCO_EDGES
except Exception:
    # fallback (để tránh lỗi nếu import chưa sẵn sàng)
    COCO_EDGES = [
        (5,7),(7,9),(6,8),(8,10),
        (11,13),(13,15),(12,14),(14,16),
        (5,6),(11,12),(5,11),(6,12),(0,5),(0,6)
    ]


# --------------------------
# Graph utils (COCO 17 kp)
# --------------------------
class Graph:
    def __init__(self, num_node: int = 17, edges: List[Tuple[int,int]] = None):
        self.num_node = num_node
        self.edges = edges if edges is not None else COCO_EDGES
        self.A = self._build_adjacency(num_node, self.edges)  # (V, V)

    @staticmethod
    def _build_adjacency(V: int, edges: List[Tuple[int,int]]) -> torch.Tensor:
        A = torch.eye(V)
        for i, j in edges:
            if 0 <= i < V and 0 <= j < V:
                A[i, j] = 1.0
                A[j, i] = 1.0
        # normalize by degree
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        A_norm = A / deg
        return A_norm  # (V, V)


# --------------------------
# ST-GCN basic building blocks
# --------------------------
class ConvTemporalGraphical(nn.Module):
    """
    Spatial graph conv:
      X: (N, C, T, V)
      A: (K, V, V)   (support multiple partitions; here K=1)
    Output: (N, C_out, T, V)
    """
    def __init__(self, in_channels, out_channels, K: int = 1, bias=True):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=(1,1), bias=bias)

    def forward(self, x, A):
        # x: (N,C,T,V); A: (K,V,V)
        N, C, T, V = x.shape
        K, _, _ = A.shape
        y = self.conv(x)  # (N, out_channels*K, T, V)
        y = y.view(N, -1, K, T, V)  # (N, out_channels, K, T, V)

        # graph message passing: sum_k (y_k @ A_k)
        # y_k: (N, out_channels, T, V) ; A_k: (V,V)
        out = torch.zeros((N, y.size(1), T, V), dtype=y.dtype, device=y.device)
        for k in range(K):
            out += torch.einsum('nctv,vw->nctw', y[:, :, k], A[k])
        return out


class STGCNBlock(nn.Module):
    """
    One ST-GCN block = Spatial GCN + Temporal Conv + Residual + BN + ReLU + Dropout
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_size: Tuple[int,int,int],  # (K,V,V)
                 stride: int = 1,
                 residual: bool = False,
                 t_kernel: int = 9,
                 dropout: float = 0.0):
        super().__init__()
        K, V, _ = A_size
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, K=K, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.is_residual = residual
        pad = (t_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(t_kernel, 1), stride=(stride, 1), padding=(pad, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        if not residual:
            self.residual = Zero()
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)
        # learnable edge importance (K,V,V)
        self.edge_importance = nn.Parameter(torch.ones(*A_size), requires_grad=True)

    def forward(self, x, A):
        # x: (N,C,T,V)
        # A: (K,V,V)
        # spatial GCN
        x_gcn = self.gcn(x, A * self.edge_importance)
        x_gcn = self.bn(x_gcn)
        x_gcn = self.relu(x_gcn)

        # temporal
        x_t = self.tcn(x_gcn)
        if self.is_residual:
            x_t = x_t + self.residual(x)
        x_t = self.relu(x_t)
        return x_t


# --------------------------
# ST-GCN Model
# --------------------------
class Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

class STGCN(nn.Module):
    """
    Input: (N, C, T, V, M) with M=1
    Output: class logits (N, num_class)
    """
    def __init__(self, config: STGCNConfig, graph: Optional[Graph] = None):
        super().__init__()
        self.config = config
        self.graph = graph if graph is not None else Graph(num_node=config.params_num_point)
        V = self.graph.num_node

        # A of shape (K,V,V); here K=1
        A = self.graph.A
        self.register_buffer('A', A.unsqueeze(0))  # (1,V,V)

        # data BN over (C*V*M)
        self.data_bn = nn.BatchNorm1d(config.params_in_channels * V * config.params_num_person)

        # build blocks
        in_c = config.params_in_channels
        self.blocks = nn.ModuleList()

        self.channels = (64, 64, 64, 128, 128, 128, 256, 256, 256)
        self.strides = (1,   1,  1,   2,   1,   1,   2,   1,   1)

        c_list = self.channels
        s_list = self.strides
        assert len(c_list) == len(s_list)
        for i, (c_out, s) in enumerate(zip(c_list, s_list)):
            block = STGCNBlock(
                in_channels=in_c,
                out_channels=c_out,
                A_size=(self.A.shape[0], V, V),
                stride=s,
                residual=True if i != 0 else False,
                t_kernel=config.params_t_kernel,
                dropout=config.params_dropout
            )
            self.blocks.append(block)
            in_c = c_out

        self.fc = nn.Conv2d(in_c, config.params_num_class, kernel_size=1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (N, C, T, V, M)
        Returns: logits (N, num_class)
        """
        N, C, T, V, M = x.shape
        assert M == self.config.params_num_person, f"Expected M={self.config.params_num_person}, got {M}"

        # BN on input
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (N,M,C,T,V)
        x = x.view(N, M * C, T, V)                 # (N, M*C, T, V)
        x = x.permute(0, 2, 3, 1).contiguous().view(N * T, V * M * C)
        x = self.data_bn(x)
        x = x.view(N, T, V, M * C).permute(0, 3, 1, 2).contiguous()  # (N, M*C, T, V)

        # spatial-temporal blocks
        for block in self.blocks:
            x = block(x, self.A)  # (N, C', T', V)

        # global average pool over T and V
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)))  # (N, C', 1, 1)

        # classifier
        x = self.fc(x)  # (N, num_class, 1, 1)
        x = x.view(x.size(0), x.size(1))  # (N, num_class)
        return x


# --------------------------
# Helper: adapt from npz -> tensor input
# --------------------------
def from_kpts_tv3_to_stgcn_input(kpts_tv3: torch.Tensor, with_conf: bool = True) -> torch.Tensor:
    """
    kpts_tv3: (T, V, 3) in torch (normalized theo Dataset)
    return: (C, T, V, M=1)  with C=3 or 2, M=1
    """
    if not with_conf:
        kpts_tv3 = kpts_tv3[..., :2]
    # (T,V,C) -> (C,T,V,1)
    x = kpts_tv3.permute(2, 0, 1).contiguous().unsqueeze(-1)
    return x
