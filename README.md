# 🏸 Badminton Pose Recognition

An AI-powered system that analyzes badminton player movements using human pose estimation and action recognition.  
It detects players, tracks body keypoints, filters active players, and classifies strokes (e.g., forehand, backhand, smash) in real-time videos.

---

## 📘 Overview

**Badminton Pose Coach** combines YOLO11-Pose for keypoint extraction and a temporal model (GRU or ST-GCN) for action classification.  
The system is designed to process badminton match videos, identify the player performing the action, and generate both a **JSON dataset** and **overlay videos** with predicted stroke labels.

<p align="center">
  <img src="docs/demo_overlay.gif" width="600">
</p>



---

## ⚙️ Key Components

| Module | Description |
|--------|--------------|
| **YOLO Pose Extractor** | Detects and tracks human keypoints frame-by-frame. |
| **Actor Selector** | Filters out inactive people and selects moving players using motion energy & ROI. |
| **Data Preprocessing** | Normalizes keypoints, pads/crops sequences, imputes NaNs, and saves as `.json` / `.npz`. |
| **Action Models** | GRU and ST-GCN architectures for temporal motion classification. |
| **Demo Pipeline** | Runs end-to-end inference from video → overlay video + predicted label. |

---

## 🧠 Core Ideas

- **Pose-Only Representation:**  
  The model uses only (x, y, conf) joint coordinates — robust against lighting or color variations.

- **Actor Selection via Motion Energy:**  
  Automatically filters out referees, audience, or inactive players by measuring normalized keypoint movement.

- **Temporal Action Models:**  
  - GRU (lightweight recurrent model for fast inference)  
  - ST-GCN (graph-based spatial-temporal reasoning)

- **Normalization:**  
  Root-centered and shoulder-scaled joint coordinates for model invariance to scale and position.

---

