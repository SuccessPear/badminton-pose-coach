# --- COCO keypoint indices (17 points) ---
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# --- Index aliases for convenience ---
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

# --- Skeleton edges (for drawing or graph adjacency) ---
COCO_EDGES = [
    # arms
    (L_SH, L_EL), (L_EL, L_WR),
    (R_SH, R_EL), (R_EL, R_WR),
    # legs
    (L_HIP, L_KNE), (L_KNE, L_ANK),
    (R_HIP, R_KNE), (R_KNE, R_ANK),
    # body core
    (L_SH, R_SH), (L_HIP, R_HIP),
    (L_SH, L_HIP), (R_SH, R_HIP),
    # neck
    (NOSE, L_SH), (NOSE, R_SH)
]

# joints weight (focus on racket arm movements)
JOINT_WEIGHTS = { L_WR:1.6, R_WR:1.6, L_EL:1.4, R_EL:1.4, L_SH:1.2, R_SH:1.2 }
