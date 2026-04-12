import numpy as np
import pandas as pd

def make_windows(df: pd.DataFrame,
                 feature_cols,
                 label_col: str,
                 group_col: str = None,
                 window: int = 128,
                 stride: int = 64):
    """
    Returns:
      X: (N, window, num_features)
      y: (N,)
    """
    X_list, y_list = [], []

    if group_col and group_col in df.columns:
        groups = df.groupby(group_col)
        iter_groups = (g for _, g in groups)
    else:
        iter_groups = [df]

    for g in iter_groups:
        g = g.reset_index(drop=True)

        feats = g[feature_cols].to_numpy(dtype=np.float32)
        labels = g[label_col].to_numpy()

        # If label changes within a window, we take majority label
        for start in range(0, len(g) - window + 1, stride):
            end = start + window
            xw = feats[start:end]
            yw = labels[start:end]
            # majority label
            vals, counts = np.unique(yw, return_counts=True)
            y_major = vals[np.argmax(counts)]

            X_list.append(xw)
            y_list.append(y_major)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y