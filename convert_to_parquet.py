import os
import re
import glob
import numpy as np
import pandas as pd

IN_FOLDER = r"C:\Users\ashok\Downloads\HumanGaitDataBase\Data"
OUT_FILE = "hugodab_dataset.parquet"

# common separators in datasets: tabs, commas, multiple spaces
SPLIT_RE = re.compile(r"[,\t ]+")


def extract_metadata(path: str):
    """
    Example filename:
      HuGaDB_v1_sitting_in_car_01_05.txt
      HuGaDB_v1_walking_12_03.txt
      HuGaDB_v1_various_17_21.txt

    Returns:
      activity, subject_id, trial_id
    """
    base = os.path.basename(path).replace(".txt", "")
    parts = base.split("_")

    # activity = everything after "HuGaDB_v1" until last 2 tokens (subject, trial)
    # parts[0]=HuGaDB, parts[1]=v1, parts[-2]=subject, parts[-1]=trial
    activity = "_".join(parts[2:-2]) if len(parts) > 4 else "unknown"
    subject_id = parts[-2] if len(parts) >= 2 else "unknown"
    trial_id = parts[-1] if len(parts) >= 1 else "unknown"

    return activity, subject_id, trial_id


def load_numeric_matrix(path: str) -> pd.DataFrame:
    """
    Reads a HuGaDB txt file robustly:
    - skips non-numeric header/metadata lines
    - splits by tab/comma/spaces
    - keeps only rows that are fully numeric
    - pads ragged rows with NaN
    """
    numeric_rows = []
    max_cols = 0

    try:
        lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    except Exception:
        lines = open(path, "r", encoding="latin-1", errors="ignore").read().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = [p for p in SPLIT_RE.split(line) if p != ""]
        if len(parts) < 3:
            continue

        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue

        numeric_rows.append(row)
        if len(row) > max_cols:
            max_cols = len(row)

    if not numeric_rows:
        raise ValueError("No numeric rows found")

    padded = []
    for r in numeric_rows:
        if len(r) < max_cols:
            r = r + [np.nan] * (max_cols - len(r))
        padded.append(r)

    return pd.DataFrame(padded)


def main():
    print("Using folder:", IN_FOLDER)

    if not os.path.exists(IN_FOLDER):
        raise RuntimeError(f"Folder does not exist: {IN_FOLDER}")

    files = glob.glob(os.path.join(IN_FOLDER, "*.txt"))
    print("Files found:", len(files))

    if not files:
        raise RuntimeError(f"No .txt files found in {IN_FOLDER}")

    all_frames = []
    skipped = 0

    for i, f in enumerate(files, 1):
        try:
            df = load_numeric_matrix(f)

            activity, subject_id, trial_id = extract_metadata(f)
            df["activity"] = activity
            df["subject_id"] = subject_id
            df["trial_id"] = trial_id
            df["source_file"] = os.path.basename(f)

            all_frames.append(df)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"[SKIP] {os.path.basename(f)} -> {e}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} files... (kept {len(all_frames)}, skipped {skipped})")

    if not all_frames:
        raise RuntimeError("All files were skipped. Nothing to save.")

    final_df = pd.concat(all_frames, ignore_index=True)
    print("Final shape:", final_df.shape)

    final_df.to_parquet(OUT_FILE, index=False)
    print(f" Saved: {OUT_FILE}")
    print(f"Kept files: {len(all_frames)} | Skipped files: {skipped}")


if __name__ == "__main__":
    main()
