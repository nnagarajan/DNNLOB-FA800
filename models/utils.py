import numpy as np
from sklearn.preprocessing import StandardScaler

def add_horizons(df,horizons,alpha):
    rolling_mid = df["mid_price"]
    rolling_mid = rolling_mid.to_numpy().flatten()
    for h in horizons:
        delta_ticks = (rolling_mid[h:] - df["mid_price"][:-h])
        df[f"Raw_Target_{str(h)}"] = delta_ticks
    df["Target_10"] = np.where(df["Raw_Target_10"] >= alpha, 2, np.where(df["Raw_Target_10"] <= -alpha, 1, 0))
    df["Target_50"] = np.where(df["Raw_Target_50"] >= alpha, 2, np.where(df["Raw_Target_50"] <= -alpha, 1, 0))
    df["Target_100"] = np.where(df["Raw_Target_100"] >= alpha, 2, np.where(df["Raw_Target_100"] <= -alpha, 1, 0))
    df[~df.isna().any(axis=1)]

def normalize(df):
    # ---- select columns to normalize: all L1–L10 Bid/Ask Price and Size ----
    price_cols = [f"L{i}-BidPrice" for i in range(1, 11)] + [f"L{i}-AskPrice" for i in range(1, 11)]
    size_cols = [f"L{i}-BidSize" for i in range(1, 11)] + [f"L{i}-AskSize" for i in range(1, 11)]
    cols_to_normalize = price_cols + size_cols
    # ---- apply z-score normalization ----
    scaler = StandardScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])