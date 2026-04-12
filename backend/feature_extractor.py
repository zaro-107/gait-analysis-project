import os
import time
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


# UPDATED: Now extracts x, y, and z (depth)
def _point_xyz(landmarks, idx, w, h):
    lm = landmarks[idx]
    # MediaPipe Z is roughly scaled to X width, so we multiply by w
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float32)


# UPDATED: True 3D angle calculation using vector dot products
def _angle_3d(a, b, c):
    """
    Angle ABC in degrees using 3D points.
    """
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosang = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _summ_stats(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "rom": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "rom": float(np.max(arr) - np.min(arr)),
    }


def _symmetry_score(left_rom, right_rom):
    denom = max(abs(left_rom) + abs(right_rom), 1e-6)
    return float(abs(left_rom - right_rom) / denom)


def _estimate_cadence_from_signal(signal, fps):
    x = np.asarray(signal, dtype=np.float32)
    if len(x) < 5 or fps <= 0:
        return 0.0, 0, []

    kernel = np.ones(5, dtype=np.float32) / 5.0
    xs = np.convolve(x, kernel, mode="same")

    peaks = []
    min_gap = max(3, int(0.25 * fps))
    for i in range(1, len(xs) - 1):
        if xs[i] > xs[i - 1] and xs[i] > xs[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_gap:
                peaks.append(i)

    if len(peaks) < 2:
        return 0.0, len(peaks), []

    intervals = np.diff(peaks) / float(fps)
    mean_interval = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
    cadence_spm = 60.0 / mean_interval if mean_interval > 1e-6 else 0.0

    return float(cadence_spm), int(len(peaks)), intervals.tolist()


# NEW: Calculate Kinematics (Velocity & Acceleration)
def _calculate_kinematics(position_series, fps):
    if len(position_series) < 2 or fps <= 0:
        return {"max_velocity": 0.0, "mean_acceleration": 0.0}
    
    dt = 1.0 / fps
    velocity = np.diff(position_series) / dt
    acceleration = np.diff(velocity) / dt if len(velocity) > 1 else np.array([0.0])
    
    return {
        "max_velocity": float(np.max(np.abs(velocity))) if len(velocity) > 0 else 0.0,
        "mean_acceleration": float(np.mean(np.abs(acceleration))) if len(acceleration) > 0 else 0.0
    }


def extract_pose_sequence(
    video_path,
    max_frames=120,
    sample_every=3,
    target_frames=60,
    min_visibility=0.35,
    resize_width=640,
    timeout_sec=60,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0

    frames = []
    frame_idx = 0
    valid_pose_frames = 0
    start_time = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            if time.time() - start_time > timeout_sec:
                print(f"Timeout reached for: {video_path}")
                break

            ok, frame = cap.read()
            if not ok:
                break

            if frame is None or frame.size == 0:
                frame_idx += 1
                continue

            frame_idx += 1

            if frame_idx % sample_every != 0:
                continue

            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                continue

            if resize_width is not None and resize_width > 0 and w != resize_width:
                new_h = int((resize_width / float(w)) * h)
                if new_h > 0:
                    frame = cv2.resize(frame, (resize_width, new_h))
                    h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks is None:
                if len(frames) >= max_frames:
                    break
                continue

            lms = res.pose_landmarks.landmark

            L_SHO, R_SHO = 11, 12
            L_HIP, R_HIP = 23, 24
            L_KNEE, R_KNEE = 25, 26
            L_ANK, R_ANK = 27, 28
            L_HEEL, R_HEEL = 29, 30
            L_FOOT, R_FOOT = 31, 32
            NOSE = 0

            important = [L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]
            vis_ok = np.mean([lms[i].visibility for i in important]) >= min_visibility
            if not vis_ok:
                if len(frames) >= max_frames:
                    break
                continue

            valid_pose_frames += 1

            # USE UPDATED 3D POINTS
            p_lsho = _point_xyz(lms, L_SHO, w, h)
            p_rsho = _point_xyz(lms, R_SHO, w, h)
            p_lhip = _point_xyz(lms, L_HIP, w, h)
            p_rhip = _point_xyz(lms, R_HIP, w, h)
            p_lknee = _point_xyz(lms, L_KNEE, w, h)
            p_rknee = _point_xyz(lms, R_KNEE, w, h)
            p_lank = _point_xyz(lms, L_ANK, w, h)
            p_rank = _point_xyz(lms, R_ANK, w, h)
            p_lheel = _point_xyz(lms, L_HEEL, w, h)
            p_rheel = _point_xyz(lms, R_HEEL, w, h)
            p_lfoot = _point_xyz(lms, L_FOOT, w, h)
            p_rfoot = _point_xyz(lms, R_FOOT, w, h)
            p_nose = _point_xyz(lms, NOSE, w, h)

            shoulder_mid = (p_lsho + p_rsho) / 2.0
            hip_mid = (p_lhip + p_rhip) / 2.0
            torso_len = max(_dist(shoulder_mid, hip_mid), 1e-6)

            # USE UPDATED 3D ANGLES
            left_knee_ang = _angle_3d(p_lhip, p_lknee, p_lank)
            right_knee_ang = _angle_3d(p_rhip, p_rknee, p_rank)
            left_hip_ang = _angle_3d(p_lsho, p_lhip, p_lknee)
            right_hip_ang = _angle_3d(p_rsho, p_rhip, p_rknee)
            left_ankle_ang = _angle_3d(p_lknee, p_lank, p_lfoot)
            right_ankle_ang = _angle_3d(p_rknee, p_rank, p_rfoot)

            # 3D vertical reference
            vertical_ref = hip_mid + np.array([0.0, -100.0, 0.0], dtype=np.float32)
            trunk_lean = _angle_3d(p_nose, shoulder_mid, vertical_ref)

            pelvis_width = _dist(p_lhip, p_rhip) / torso_len
            ankle_dist = _dist(p_lank, p_rank) / torso_len
            heel_dist = _dist(p_lheel, p_rheel) / torso_len
            step_width_proxy = abs(p_lank[0] - p_rank[0]) / torso_len

            lank_y_rel = (p_lank[1] - hip_mid[1]) / torso_len
            rank_y_rel = (p_rank[1] - hip_mid[1]) / torso_len
            lank_x_rel = (p_lank[0] - hip_mid[0]) / torso_len
            rank_x_rel = (p_rank[0] - hip_mid[0]) / torso_len
            lhip_x_rel = (p_lhip[0] - hip_mid[0]) / torso_len
            rhip_x_rel = (p_rhip[0] - hip_mid[0]) / torso_len
            lhip_y_rel = (p_lhip[1] - hip_mid[1]) / torso_len
            rhip_y_rel = (p_rhip[1] - hip_mid[1]) / torso_len
            lknee_y_rel = (p_lknee[1] - hip_mid[1]) / torso_len
            rknee_y_rel = (p_rknee[1] - hip_mid[1]) / torso_len

            feat = [
                left_knee_ang, right_knee_ang, left_hip_ang, right_hip_ang,
                left_ankle_ang, right_ankle_ang, trunk_lean, pelvis_width,
                ankle_dist, heel_dist, step_width_proxy, lank_y_rel, rank_y_rel,
                lank_x_rel, rank_x_rel, lhip_x_rel, rhip_x_rel, lhip_y_rel,
                rhip_y_rel, lknee_y_rel, rknee_y_rel,
            ]
            frames.append(np.asarray(feat, dtype=np.float32))

            if len(frames) >= max_frames:
                break

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No usable pose frames extracted from: {video_path}")

    seq = np.stack(frames, axis=0)
    old_idx = np.linspace(0, len(seq) - 1, num=len(seq))
    new_idx = np.linspace(0, len(seq) - 1, num=target_frames)

    seq_rs = []
    for d in range(seq.shape[1]):
        seq_rs.append(np.interp(new_idx, old_idx, seq[:, d]))
    seq_rs = np.stack(seq_rs, axis=1).astype(np.float32)

    meta = {
        "video_path": video_path,
        "fps_effective": float(fps / max(sample_every, 1)),
        "frames_raw_used": int(len(frames)),
        "frames_after_resample": int(target_frames),
        "feature_dim": int(seq_rs.shape[1]),
        "valid_pose_frames": int(valid_pose_frames),
    }
    return seq_rs, meta


def extract_gait_features(
    video_path,
    max_frames=120,
    sample_every=3,
    target_pose_frames=60,
    return_series=False,
    enable_enhancement=False,
    resize_width=640,
    timeout_sec=60,
    **kwargs,
):
    seq, meta = extract_pose_sequence(
        video_path=video_path,
        max_frames=max_frames,
        sample_every=sample_every,
        target_frames=target_pose_frames,
        resize_width=resize_width,
        timeout_sec=timeout_sec,
    )

    lk, rk = seq[:, 0], seq[:, 1]
    lh, rh = seq[:, 2], seq[:, 3]
    la, ra = seq[:, 4], seq[:, 5]
    trunk = seq[:, 6]
    pelvis_width = seq[:, 7]
    ankle_dist = seq[:, 8]
    heel_dist = seq[:, 9]
    step_width = seq[:, 10]
    lank_y, rank_y = seq[:, 11], seq[:, 12]
    lhip_x, rhip_x = seq[:, 15], seq[:, 16]

    cadence_left, peaks_left, intervals_left = _estimate_cadence_from_signal(-lank_y, meta["fps_effective"])
    cadence_right, peaks_right, intervals_right = _estimate_cadence_from_signal(-rank_y, meta["fps_effective"])

    step_var_left = float(np.std(intervals_left)) if len(intervals_left) > 1 else 0.0
    step_var_right = float(np.std(intervals_right)) if len(intervals_right) > 1 else 0.0
    step_variability = float((step_var_left + step_var_right) / 2.0)

    pelvis_sway = float(np.std(lhip_x - rhip_x))

    knee_sym = _symmetry_score(_summ_stats(lk)["rom"], _summ_stats(rk)["rom"])
    hip_sym = _symmetry_score(_summ_stats(lh)["rom"], _summ_stats(rh)["rom"])
    ankle_sym = _symmetry_score(_summ_stats(la)["rom"], _summ_stats(ra)["rom"])

    # ADD KINEMATICS
    left_ankle_kinematics = _calculate_kinematics(lank_y, meta["fps_effective"])
    right_ankle_kinematics = _calculate_kinematics(rank_y, meta["fps_effective"])

    feats = {
        "meta": meta,
        "left_knee": _summ_stats(lk),
        "right_knee": _summ_stats(rk),
        "left_hip": _summ_stats(lh),
        "right_hip": _summ_stats(rh),
        "left_ankle": _summ_stats(la),
        "right_ankle": _summ_stats(ra),
        "trunk_lean": _summ_stats(trunk),
        "pelvis_width": _summ_stats(pelvis_width),
        "step_length_proxy": _summ_stats(heel_dist),
        "ankle_distance": _summ_stats(ankle_dist),
        "step_width_proxy": _summ_stats(step_width),
        "pelvis_sway": float(pelvis_sway),
        "cadence_left_proxy_spm": float(cadence_left),
        "cadence_right_proxy_spm": float(cadence_right),
        "cadence_proxy_peaks_left": int(peaks_left),
        "cadence_proxy_peaks_right": int(peaks_right),
        "step_variability": float(step_variability),
        "symmetry": {
            "knee_rom_0to1": float(knee_sym),
            "hip_rom_0to1": float(hip_sym),
            "ankle_rom_0to1": float(ankle_sym),
        },
        "kinematics": {
            "left_ankle": left_ankle_kinematics,
            "right_ankle": right_ankle_kinematics
        },
        "quality_flag": "ok" if meta["frames_raw_used"] >= 15 else "low_quality_few_pose_frames",
    }

    if return_series:
        feats["series"] = seq.tolist()

    return feats