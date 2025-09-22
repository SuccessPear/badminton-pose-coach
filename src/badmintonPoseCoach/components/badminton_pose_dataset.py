from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import json
from badmintonPoseCoach.entity.config_entity import TrainingConfig

class BadmintonPoseDataset(Dataset):
    """
    Dataset class that load json file and output a dataframe of keypoints
    """
    def __init__(self,
                 config: TrainingConfig,
                 seed: int = 42,
                 split: str = 'train',
                 split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
                 frame_format: str = 'auto',
                 num_joints: int = 17,):
        self.training_data = Path(config.training_data)
        self.frame_format = frame_format
        self.num_joints = num_joints

        class_dirs = sorted([d for d in self.training_data.iterdir() if d.is_dir()])
        self.class_names = [d.name for d in class_dirs]

        # list all files in data folder
        self.file_list = []
        for ci, d in enumerate(class_dirs):
            for p in sorted(d.rglob("*.json")):
                self.file_list.append((p, ci))

        # Train/val/test split
        g = torch.Generator().manual_seed(seed)
        per_class_idx = [[] for _ in self.class_names]
        for idx, (_p, ci) in enumerate(self.file_list):
            per_class_idx[ci].append(idx)
        for lst in per_class_idx:
            perm = torch.randperm(len(lst), generator=g).tolist()
            lst = [lst[i] for i in perm]

        def take_splits(idxs: list[int]) -> tuple[list[int], list[int], list[int]]:
            n = len(idxs)
            n_train = int(n * split_ratio[0])
            n_val = int(n * split_ratio[1])
            return idxs[:n_train], idxs[n_train:n_train+n_val], idxs[n_train+n_val:]

        split_map = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}
        which = split_map[split]

        selected: list[int] = []
        for lst in per_class_idx:
            tr, va, te = take_splits(lst)
            selected.extend([tr, va, te][which])
        selected = sorted(selected)

        self.files: list[Path] = [self.file_list[i][0] for i in selected]
        self.labels: list[int] = [self.file_list[i][1] for i in selected]



    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, int]:
        path = self.files[index]
        label = self.labels[index]
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        seq = obj.get("seq")
        if seq is None:
            raise ValueError(f"Missing 'seq' in {path}")

        pose = self._to_tensor_TxKx3(seq)

        return pose, label

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
