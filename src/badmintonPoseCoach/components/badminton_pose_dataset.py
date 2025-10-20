from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import json
from badmintonPoseCoach.entity.config_entity import TrainingConfig
import numpy as np
import os

class BadmintonPoseDataset(Dataset):
    """
    Dataset class that load json file and output a dataframe of keypoints
    """
    def __init__(self,
                 config: TrainingConfig,
                 seed: int = 42,
                 split: str = 'train',
                 frame_format: str = 'auto',
                 num_joints: int = 17,):
        self.config = config
        self.training_data = Path(config.training_data)
        self.num_joints = num_joints
        self.split = split
        self.frame_format = frame_format
        self.split_dir = self.config.training_data / self.split

        self.classes, self.label_to_id = self._discover_classes()
        # list all files in data folder
        self.files = [self.split_dir / file for file in os.listdir(self.split_dir)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.files[index]

        data = np.load(path)
        meta = json.loads(str(data["meta_json"]))
        label = meta.get("label")
        seq = data.get("kpts")

        if seq is None:
            raise ValueError(f"Missing 'kpts' in {path}")

        if self.config.params_model_name == "gru":
            pose = self._to_tensor_TxKx3(seq.tolist())
            return self.normalize_pose_rnn(pose, meta.get("W"), meta.get("H")), self.label_to_id[label]

        elif self.config.params_model_name == "stgcn":
            pose  = self.normalize_stgcn(seq, meta.get("W"), meta.get("H"))
            data = np.transpose(pose, (2, 0, 1))[..., np.newaxis]
            data = torch.from_numpy(data).float()
            return data, self.label_to_id[label]

    def _discover_classes(self):
        labels = set()
        for p in self.split_dir.glob("*.npz"):
            z = np.load(p, allow_pickle=False)
            meta = json.loads(str(z["meta_json"]))
            labels.add(meta.get("label"))
        classes = sorted(labels)
        return classes, {c: i for i, c in enumerate(classes)}

    def _to_tensor_TxKx3(self, seq: any) -> torch.tensor:
        if self.frame_format in ("auto", "Kx3") and isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
            sample = seq[0]
            if len(sample) > 0 and isinstance(sample[0], list):
                return torch.tensor(seq, dtype=torch.float32)
            else:
                if self.frame_format == "flat" and self.num_joints is not None:
                    K = int(self.num_joints)
                else:
                    flen = len(sample)
                    if flen % 3 != 0:
                        raise ValueError("Cannot infer num_keypoints")
                    K = flen // 3
                frames_Kx3 = []
                for fr in seq:
                    triplets = [fr[i:i+3] for i in range(0, len(fr), 3)]
                    frames_Kx3.append(triplets)
                return torch.tensor(frames_Kx3, dtype=torch.float32)

        if self.frame_format in ("auto", "flat") and isinstance(seq, list) and seq and isinstance(seq[0], (int,float)):
            if self.num_joints is None:
                raise ValueError("Need num_keypoints for flat seq")
            K = int(self.num_joints)
            T = len(seq) // (K*3)
            return torch.tensor(seq, dtype=torch.float32).view(T, K, 3)

        raise ValueError("Unsupported 'seq' structure")

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

    @staticmethod
    def normalize_pose_rnn(pose, W, H, method="skeleton"):
        # pose: (T,K,3)
        if method == "image":
            pose[...,0] /= W
            pose[...,1] /= H
        elif method == "skeleton":
            # pelvis = joint 11,12 in average
            pelvis = pose[:,[11,12],:2].mean(1, keepdims=True)
            pose[...,:2] -= pelvis
            # scale with the shoulder
            shoulder = pose[:,[5,6],:2].mean(1, keepdims=True)
            scale = (pose[:,5,:2]-pose[:,6,:2]).norm(dim=-1, keepdim=True).clamp(min=1e-6)
            pose[...,:2] /= scale[:,None,:]
        return pose

    def normalize_stgcn(self, kpts, W, H, mode="root"):
        """
        kpts: (T, V, 3) float32, pixel-space
        W, H: frame width, height
        mode: "image" | "root" | "person"
        """
        L_HIP, R_HIP = 11, 12
        k = kpts.copy().astype(np.float32)
        # 1. scale to [0,1]
        k[..., 0] /= float(W)
        k[..., 1] /= float(H)

        if mode == "image":
            return k

        if mode == "root":
            # center theo trung bình 2 hip
            root = np.nanmean(k[:, [L_HIP, R_HIP], :2], axis=1, keepdims=True)  # (T,1,2)
            k[..., :2] = k[..., :2] - root
            return k

        if mode == "person":
            # normalize theo bbox người
            x_min = np.nanmin(k[..., 0], axis=1, keepdims=True)
            x_max = np.nanmax(k[..., 0], axis=1, keepdims=True)
            y_min = np.nanmin(k[..., 1], axis=1, keepdims=True)
            y_max = np.nanmax(k[..., 1], axis=1, keepdims=True)
            scale = np.maximum(x_max - x_min, y_max - y_min)
            k[..., 0:2] = (k[..., 0:2] - np.stack([x_min, y_min], axis=-1)) / np.clip(scale, 1e-6, None)
            return k

        raise ValueError(f"Unknown normalize mode: {mode}")
