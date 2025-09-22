import numpy as np
from collections import deque

# --- KONSTANTA YANG DIKALIBRASI ---
# (Semua konstanta dari skrip sebelumnya diletakkan di sini)
HISTORY_SIZE = 30
SHORT_HISTORY = 12
HIP_Y_CHANGE_JUMP_THRESH = -15.0
ANKLE_SPEED_KICK_THRESH = 40.0
WRIST_SPEED_PUNCH_THRESH = 40.0
AVG_ANKLE_SPEED_RUN_THRESH = 22.0
HIP_STD_DEV_RUN_THRESH = 4.0
WALK_ANKLE_SPEED_THRESH = 4.5
WALK_HIP_STD_DEV_THRESH = 2.0
STATIC_SPEED_THRESH = 3.0
STATIC_HIP_STD_DEV_THRESH = 1.8
SITTING_KNEE_ANGLE_THRESH = 130.0
QUICK_ACTION_FRAMES = 3
TRANSITION_SMOOTHING = 6

# --- FUNGSI BANTUAN ---
def get_keypoint(keypoints, index):
    if keypoints is not None and len(keypoints) > index:
        kpt = keypoints[index]
        if len(kpt) > 2 and kpt[2] > 0.35:
            return np.array(kpt[:2])
    return None

def calculate_angle(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None: return None
    v1 = p1 - p2
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_velocity(history, keypoint_idx, frames_back):
    if len(history) < frames_back + 1: return 0
    velocities = []
    # Menggunakan deque secara langsung lebih efisien
    history_list = list(history)
    start_index = max(0, len(history_list) - frames_back)
    for i in range(start_index, len(history_list)):
        curr_kpt = get_keypoint(history_list[i], keypoint_idx)
        prev_kpt = get_keypoint(history_list[i-1], keypoint_idx)
        if curr_kpt is not None and prev_kpt is not None:
            velocities.append(np.linalg.norm(curr_kpt - prev_kpt))
    return np.mean(velocities) if velocities else 0

# --- FUNGSI UTAMA KLASIFIKASI ---
def classify_action(history, last_stable_action):
    if len(history) < 2:
        return last_stable_action

    keypoint_indices = {
        'l_hip': 11, 'r_hip': 12, 'l_knee': 13, 'r_knee': 14,
        'l_ankle': 15, 'r_ankle': 16, 'l_wrist': 9, 'r_wrist': 10
    }
    kpts = {name: get_keypoint(history[-1], idx) for name, idx in keypoint_indices.items()}

    if not all(k is not None for k in [kpts['l_hip'], kpts['r_hip'], kpts['l_knee'], kpts['r_knee']]):
        return last_stable_action

    # 1. DETEKSI AKSI CEPAT
    if len(history) >= QUICK_ACTION_FRAMES:
        hip_y_vel = 0
        if kpts['l_hip'] is not None and kpts['r_hip'] is not None:
            curr_hip_y = (kpts['l_hip'][1] + kpts['r_hip'][1]) / 2
            prev_l_hip = get_keypoint(history[-QUICK_ACTION_FRAMES], 11)
            prev_r_hip = get_keypoint(history[-QUICK_ACTION_FRAMES], 12)
            if prev_l_hip is not None and prev_r_hip is not None:
                prev_hip_y = (prev_l_hip[1] + prev_r_hip[1]) / 2
                hip_y_vel = curr_hip_y - prev_hip_y
        if hip_y_vel < HIP_Y_CHANGE_JUMP_THRESH: return "Jumping"
        if max(calculate_velocity(history, 15, QUICK_ACTION_FRAMES), calculate_velocity(history, 16, QUICK_ACTION_FRAMES)) > ANKLE_SPEED_KICK_THRESH: return "Kicking"
        if max(calculate_velocity(history, 9, QUICK_ACTION_FRAMES), calculate_velocity(history, 10, QUICK_ACTION_FRAMES)) > WRIST_SPEED_PUNCH_THRESH: return "Punching"

    # 2. DETEKSI POSTUR
    if kpts['l_ankle'] is not None and kpts['r_ankle'] is not None:
        left_knee_angle = calculate_angle(kpts['l_hip'], kpts['l_knee'], kpts['l_ankle'])
        right_knee_angle = calculate_angle(kpts['r_hip'], kpts['r_knee'], kpts['r_ankle'])
        if left_knee_angle is not None and right_knee_angle is not None:
            if (left_knee_angle + right_knee_angle) / 2 < SITTING_KNEE_ANGLE_THRESH:
                return "Sitting"

    # 3. ANALISIS GERAKAN BERKELANJUTAN
    if len(history) >= SHORT_HISTORY:
        avg_ankle_speed = (calculate_velocity(history, 15, SHORT_HISTORY) + calculate_velocity(history, 16, SHORT_HISTORY)) / 2
        hip_y_positions = [
            (get_keypoint(frame, 11)[1] + get_keypoint(frame, 12)[1]) / 2
            for frame in list(history)[-SHORT_HISTORY:]
            if get_keypoint(frame, 11) is not None and get_keypoint(frame, 12) is not None
        ]
        hip_std_dev = np.std(hip_y_positions) if len(hip_y_positions) > 1 else 0

        if avg_ankle_speed > AVG_ANKLE_SPEED_RUN_THRESH and hip_std_dev > HIP_STD_DEV_RUN_THRESH: return "Running"
        if avg_ankle_speed < STATIC_SPEED_THRESH and hip_std_dev < STATIC_HIP_STD_DEV_THRESH: return "Standing"
        if avg_ankle_speed > WALK_ANKLE_SPEED_THRESH or hip_std_dev > WALK_HIP_STD_DEV_THRESH: return "Walking"

    return last_stable_action