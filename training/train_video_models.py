import os
import sys
import glob
import cv2
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------------------------------------
# Fix import path so Python can find /backend folder
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../gait/training
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../gait

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.feature_extractor import extract_gait_features


SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)


def get_video_files(folder):
    # Removed .webm because many of your webm files are not decoding properly
    patterns = ("*.mp4", "*.avi", "*.mov", "*.mkv","*.MP4", "*.AVI", "*.MOV", "*.MKV","*.WebM", "*.WEBM")
    vids = []
    for ext in patterns:
        vids.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(vids)


def is_valid_video(video_path, max_frames_allowed=2000):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        return False, "cannot open"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frames <= 0:
        return False, f"invalid frame count ({frames})"
    if width <= 0 or height <= 0:
        return False, f"invalid size ({width}x{height})"
    if frames > max_frames_allowed:
        return False, f"too long ({frames} frames)"
    if fps <= 0:
        return False, f"invalid fps ({fps})"

    return True, f"fps={fps:.2f}, frames={frames}, size={width}x{height}"


def flatten_gait_features(feats):
    return [
        feats["left_knee"]["mean"],
        feats["left_knee"]["std"],
        feats["left_knee"]["rom"],

        feats["right_knee"]["mean"],
        feats["right_knee"]["std"],
        feats["right_knee"]["rom"],

        feats["left_hip"]["mean"],
        feats["left_hip"]["std"],
        feats["left_hip"]["rom"],

        feats["right_hip"]["mean"],
        feats["right_hip"]["std"],
        feats["right_hip"]["rom"],

        feats["left_ankle"]["mean"],
        feats["left_ankle"]["std"],
        feats["left_ankle"]["rom"],

        feats["right_ankle"]["mean"],
        feats["right_ankle"]["std"],
        feats["right_ankle"]["rom"],

        feats["trunk_lean"]["mean"],
        feats["trunk_lean"]["std"],
        feats["trunk_lean"]["rom"],

        feats["pelvis_width"]["mean"],
        feats["pelvis_width"]["std"],

        feats["step_length_proxy"]["mean"],
        feats["step_length_proxy"]["std"],
        feats["step_length_proxy"]["rom"],

        feats["ankle_distance"]["mean"],
        feats["ankle_distance"]["std"],
        feats["ankle_distance"]["rom"],

        feats["step_width_proxy"]["mean"],
        feats["step_width_proxy"]["std"],
        feats["step_width_proxy"]["rom"],

        feats["pelvis_sway"],
        feats["cadence_left_proxy_spm"],
        feats["cadence_right_proxy_spm"],
        feats["cadence_proxy_peaks_left"],
        feats["cadence_proxy_peaks_right"],
        feats["step_variability"],

        feats["symmetry"]["knee_rom_0to1"],
        feats["symmetry"]["hip_rom_0to1"],
        feats["symmetry"]["ankle_rom_0to1"],

        feats["meta"]["fps_effective"],
        feats["meta"]["frames_raw_used"],
        feats["meta"]["feature_dim"],
    ]


def build_dataset_from_folders(root_dir, classes):
    X, y, paths = [], [], []

    print(f"=== Building dataset from: {root_dir} ===")

    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_dir):
            print(f"Skipping missing class folder: {class_dir}")
            continue

        videos = get_video_files(class_dir)
        if len(videos) == 0:
            print(f"No videos found in: {class_dir}")
            continue

        print(f"\nClass '{class_name}' -> found {len(videos)} video(s)")

        for idx, video_path in enumerate(videos, 1):
            print(f"  [{idx}/{len(videos)}] Checking: {os.path.basename(video_path)}")

            valid, info = is_valid_video(video_path, max_frames_allowed=2000)
            if not valid:
                print(f"     Skipping: {info}")
                continue
            else:
                print(f"     Video OK: {info}")

            try:
                feats = extract_gait_features(
                    video_path=video_path,
                    max_frames=120,          # reduced for speed
                    sample_every=3,         # skip more frames for speed
                    target_pose_frames=60,
                    enable_enhancement=True,
                    resize_width=640,
                    timeout_sec=60,
                )

                if not feats:
                    print(f"     Skipping: empty features")
                    continue

                if feats.get("quality_flag") == "low_quality_few_pose_frames":
                    print(f"     Skipping: low-quality video")
                    continue

                feature_vector = flatten_gait_features(feats)

                if not np.all(np.isfinite(feature_vector)):
                    print(f"     Skipping: non-finite feature values")
                    continue

                X.append(feature_vector)
                y.append(class_name)
                paths.append(video_path)

                print(f"     Added sample successfully")

            except Exception as e:
                print(f"     Skipping due to error: {e}")

    if len(X) == 0:
        raise RuntimeError(f"No usable videos found in {root_dir}.")

    return np.array(X, dtype=np.float32), np.array(y), paths


def main():
    root_dir = os.path.join(PROJECT_ROOT, "video_data")
    classes = ["normal_gait", "antalgic_gait", "spastic_gait", "ataxic_gait", "parkinsonian_gait", "waddling_gait","cautious_gait","choreiform_gait","hemiplegic_gait","steppage_gait","propulsive_gait","scissor_gait","myopathic_gait","neuropathic_gait","vestibular_gait","diplegic_gait"]

    X, y_text, paths = build_dataset_from_folders(root_dir, classes)

    print("\nDataset shape:", X.shape)
    print("Total samples:", len(X))

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise RuntimeError("Need at least 2 classes with usable videos for training.")

    counts = np.bincount(y)
    use_stratify = counts.min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
    )

    print("\nTraining RandomForest...")
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("\nAccuracy:", acc)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

    print(
        "\nClassification Report:\n",
        classification_report(
            y_test,
            pred,
            labels=np.arange(len(le.classes_)),
            target_names=le.classes_,
            zero_division=0,
        ),
    )

    joblib.dump(clf, os.path.join(SAVE_DIR, "video_gait_clf.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "video_gait_scaler.pkl"))
    joblib.dump(le, os.path.join(SAVE_DIR, "video_gait_label_encoder.pkl"))
    joblib.dump(
        {
            "classes": list(le.classes_),
            "feature_type": "summary_gait_features",
            "supported_extensions": [".mp4", ".avi", ".mov", ".mkv"],
            "n_features": int(X.shape[1]),
        },
        os.path.join(SAVE_DIR, "video_gait_meta.pkl"),
    )

    print("\nSaved classical video gait model to:", SAVE_DIR)


if __name__ == "__main__":
    main()