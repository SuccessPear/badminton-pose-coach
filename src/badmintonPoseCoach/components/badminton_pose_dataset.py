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

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, int]:
        path = self.files[index]

        data = np.load(path)
        meta = json.loads(str(data["meta_json"]))
        label = meta.get("label")
        seq = data.get("kpts")

        if seq is None:
            raise ValueError(f"Missing 'kpts' in {path}")

        pose = self._to_tensor_TxKx3(seq.tolist())

        return self.normalize_pose(pose, 720, 1280), self.label_to_id[label]

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

    @staticmethod
    def normalize_pose(pose, W, H, method="skeleton"):
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