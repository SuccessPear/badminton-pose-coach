import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from badmintonPoseCoach.components.models.gru_model import GRUModel
from badmintonPoseCoach.config.configuration import ConfigurationManager
from badmintonPoseCoach.entity.config_entity import *
from badmintonPoseCoach.components.dataset_preprocessing import DataPreprocessing
from badmintonPoseCoach.core.pose.normalize import normalize_pose_rnn
from badmintonPoseCoach.components.visualize.overlay import overlay_video
import json

video_path = ("data/Badminton_Strorke_Video/forehand_net_shot/017.mp4")
video_path = "data/VideoBadminton_Dataset/VideoBadminton_Dataset/02_Lift/2022-08-30_18-00-09_dataset_set1_024_002481_002509_B_02.mp4"

config = ConfigurationManager()
dataset_preprocessing_config = config.get_dataset_preprocessing_config()
dataset_preprocessing = DataPreprocessing(dataset_preprocessing_config)

tracks, actors_details, roi = dataset_preprocessing.process_one_test(video_path, "")

model_state = torch.load("artifacts/training/checkpoints/best.pkl", weights_only=False)
print(model_state.keys())
base_model = torch.load("artifacts/prepare_base_model/base_model.pkl", weights_only=False)
base_model.load_state_dict(model_state.get("model_state"))

import json
preds = {}
for key, data in actors_details.items():
    meta = json.loads(str(data["meta_json"]))
    label = meta.get("label")
    seq = data.get("kpts")
    W = data.get("W")
    H = data.get("H")

    norm_seq = normalize_pose_rnn(torch.tensor(seq), W, H)
    logits = base_model(norm_seq.view(norm_seq.shape[0], -1))
    preds[key] = logits.argmax(-1).item()

classes_json = json.load(open("labels.json"))
labels = {idx: classes_json["classes"][i] for idx, i in preds.items()}

print(labels)


overlay_video(
    video_path, "overlay.mp4", tracks, next(iter(actors_details)),
    score_thr=0.5, show=False,
    fps_override=meta["fps"], W=meta["W"], H=meta["H"],
    valid_ratio=(0.0, 1.0), roi=roi, labels = labels
)