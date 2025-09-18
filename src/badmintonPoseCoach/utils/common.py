import os
from box.exceptions import BoxValueError
import yaml
from badmintonPoseCoach import logger
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import json, cv2, numpy as np
from typing import Dict, Any, List, Optional
from ultralytics import YOLO


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

# pose_extract.py


def extract_keypoints_from_video(
    video_path: str,
    model_or_path: str = "yolov12n-pose.pt",   # you can switch to yolov8s/m/l-pose if GPU allows
    fps_sample: int = 15,                     # sample rate in frames-per-second for keypoints
    conf: float = 0.25,                       # detection confidence
    pick: str = "largest",                    # which person to pick: "largest" | "highest_conf" | "first"
) -> Dict[str, Any]:
    """
    Extract pose keypoints from a single video.
    Returns a dict: {"fps_sample": int, "seq": List[List[[x,y,score], ...]]}
    where seq shape is [T, J, 3]. Missing frames filled with NaN.
    """
    # Load pose model (lazy load once per call; you can lift outside if batch processing)
    model = YOLO(model_or_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / fps_sample)), 1)

    sequence: List[List[List[float]]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # Inference
            results = model.predict(frame, verbose=False, conf=conf)

            # Default: fill with NaN (will become zeros after normalization if you want)
            kps_np: Optional[np.ndarray] = None

            if len(results) > 0:
                res = results[0]
                # res.keypoints.data: [N, J, 3], res.boxes.xyxy: [N, 4]
                if res.keypoints is not None and res.keypoints.data is not None and len(res.keypoints.data) > 0:
                    kpts = res.keypoints.data.cpu().numpy()  # [N, J, 3]
                    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else None

                    if pick == "largest" and boxes is not None and len(boxes) == len(kpts):
                        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
                        idx = int(np.argmax(areas))
                        kps_np = kpts[idx]  # [J,3]
                    elif pick == "highest_conf":
                        # mean score per person
                        confs = kpts[:,:,2].mean(axis=1)
                        idx = int(np.argmax(confs))
                        kps_np = kpts[idx]
                    else:
                        kps_np = kpts[0]

            if kps_np is None:
                # YOLOv8-pose has 17 joints; if you use a different model, adjust J accordingly
                kps_np = np.full((17, 3), np.nan, dtype=np.float32)

            sequence.append(kps_np.tolist())

        frame_idx += 1

    cap.release()
    return {"fps_sample": fps_sample, "seq": sequence}

def is_video_readable(video_path: str, n_try_frames: int = 3) -> bool:
    """
    Quick sanity check: can we open the video and read a few frames?
    Returns False if the video cannot be opened or frames cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ok_any = False
    for _ in range(n_try_frames):
        ok, _ = cap.read()
        if ok:
            ok_any = True
            break
    cap.release()
    return ok_any

def is_json_processed_ok(json_path: str, min_frames: int = 1) -> bool:
    """
    Consider a JSON 'processed' if:
      - file exists
      - can be parsed
      - has a non-empty 'seq' with at least min_frames
    """
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        seq = j.get("seq", [])
        return isinstance(seq, list) and len(seq) >= min_frames
    except Exception:
        return False
