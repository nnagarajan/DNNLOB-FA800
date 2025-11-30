import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd

def add_horizons(df,horizons,alpha):
    #rolling_mid = df["mid_price"]
    #rolling_mid = rolling_mid.to_numpy().flatten()
    for h in horizons:
        #delta_ticks = (rolling_mid[h:] - df["mid_price"][:-h])
        rolling_mid_minus = df['mid_price'].rolling(window=h, min_periods=h).mean().shift(h)
        rolling_mid_plus = df["mid_price"].rolling(window=h, min_periods=h).mean().to_numpy().flatten()
        delta_ticks = rolling_mid_plus - rolling_mid_minus
        df[f"Raw_Target_{str(h)}"] = delta_ticks
    df["Target_10"] = np.where(df["Raw_Target_10"] >= alpha, 2, np.where(df["Raw_Target_10"] <= -alpha, 1, 0))
    df["Target_50"] = np.where(df["Raw_Target_50"] >= alpha, 2, np.where(df["Raw_Target_50"] <= -alpha, 1, 0))
    df["Target_100"] = np.where(df["Raw_Target_100"] >= alpha, 2, np.where(df["Raw_Target_100"] <= -alpha, 1, 0))
    return  df[~df.isna().any(axis=1)]

def normalize(df):
    # ---- select columns to normalize: all L1–L10 Bid/Ask Price and Size ----
    price_cols = [f"L{i}-BidPrice" for i in range(1, 11)] + [f"L{i}-AskPrice" for i in range(1, 11)]
    size_cols = [f"L{i}-BidSize" for i in range(1, 11)] + [f"L{i}-AskSize" for i in range(1, 11)]
    cols_to_normalize = price_cols + size_cols
    # ---- apply z-score normalization ----
    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

def normalize_train(df):
    scaler = StandardScaler()
    price_cols = [f"L{i}-BidPrice" for i in range(1, 11)] + [f"L{i}-AskPrice" for i in range(1, 11)]
    size_cols = [f"L{i}-BidSize" for i in range(1, 11)] + [f"L{i}-AskSize" for i in range(1, 11)]
    cols_to_normalize = price_cols + size_cols
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    return df, scaler

def normalize_apply(df, scaler):
    price_cols = [f"L{i}-BidPrice" for i in range(1, 11)] + [f"L{i}-AskPrice" for i in range(1, 11)]
    size_cols = [f"L{i}-BidSize" for i in range(1, 11)] + [f"L{i}-AskSize" for i in range(1, 11)]
    cols_to_normalize = price_cols + size_cols
    df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])
    return df

def ad_normalize(df, window_size=100):
    """
    Apply dynamic rolling (streaming) normalization to L1-L10 Bid/Ask Price & Size.
    Drops the first `window_size` rows explicitly after normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by time.
    window_size : int
        Number of past windows to use for normalization statistics.

    Returns
    -------
    df_norm : pd.DataFrame
        Normalized dataframe with first `window_size` rows removed.
    """

    df_norm = df.copy()

    # ---- Select columns to normalize ----
    price_cols = [f"L{i}-BidPrice" for i in range(1, 11)] + \
                 [f"L{i}-AskPrice" for i in range(1, 11)]
    size_cols  = [f"L{i}-BidSize"  for i in range(1, 11)] + \
                 [f"L{i}-AskSize"  for i in range(1, 11)]
    feature_names = price_cols + size_cols

    # ---- Initialize first window statistics ----
    first_mean   = df_norm[feature_names].iloc[:window_size].mean(axis=0)
    first_mean2  = (df_norm[feature_names].iloc[:window_size] ** 2).mean(axis=0)
    first_count  = pd.Series(np.ones(len(feature_names)), index=feature_names)

    mean_df     = pd.DataFrame([first_mean]  * window_size)
    mean2_df    = pd.DataFrame([first_mean2] * window_size)
    nsamples_df = pd.DataFrame([first_count] * window_size)

    # Store µ and σ for each row
    means_list = []
    stds_list = []

    # ---- Iterate through all rows ----
    for i in range(len(df_norm)):

        # Compute window mean
        z_mean = (nsamples_df * mean_df).sum(axis=0) / nsamples_df.sum(axis=0)

        # Compute window std
        z_std = np.sqrt(
            (nsamples_df * mean2_df).sum(axis=0) / nsamples_df.sum(axis=0)
            - z_mean ** 2
        )

        means_list.append(z_mean.values)
        stds_list.append(z_std.values)

        # Slide window forward
        mean_df     = mean_df.iloc[1:]
        mean2_df    = mean2_df.iloc[1:]
        nsamples_df = nsamples_df.iloc[1:]

        # Add new stats from the current row
        row_vals = df_norm.iloc[i][feature_names].values.astype(float)

        mean_df = pd.concat(
            [mean_df, pd.DataFrame([row_vals], columns=feature_names)],
            ignore_index=True
        )
        mean2_df = pd.concat(
            [mean2_df, pd.DataFrame([row_vals**2], columns=feature_names)],
            ignore_index=True
        )
        nsamples_df = pd.concat(
            [nsamples_df,
             pd.DataFrame([np.ones(len(feature_names))], columns=feature_names)],
            ignore_index=True
        )

    # ---- Convert µ and σ into DataFrames ----
    mean_matrix = pd.DataFrame(means_list, columns=feature_names, index=df_norm.index)
    std_matrix  = pd.DataFrame(stds_list,  columns=feature_names, index=df_norm.index)

    # ---- Apply normalization ----
    df_norm[feature_names] = (df_norm[feature_names] - mean_matrix) / std_matrix.replace(0, np.nan)

    # ---- Explicitly drop the first window_size rows ----
    df_norm = df_norm.iloc[window_size:].copy()

    return df_norm

def normalize_by_prev_day(
    df,
    datetime_col="Date-Time",
    feature_cols=None,
    drop_first_day=True
):
    """
    Normalize each day's rows using ONLY the previous day's feature values.
    Does NOT sort the dataframe — assumes df is already sorted by day.

    For day D:
        mean/std computed from day D-1
        applied to all rows in day D

    The first day cannot be normalized and is dropped by default.
    """

    df_norm = df.copy()

    # ---- default feature set (LOB L1-L10 price/size) ----
    if feature_cols is None:
        price_cols = [f"L{i}-BidPrice" for i in range(1, 11)] + \
                     [f"L{i}-AskPrice" for i in range(1, 11)]
        size_cols  = [f"L{i}-BidSize"  for i in range(1, 11)] + \
                     [f"L{i}-AskSize"  for i in range(1, 11)]
        feature_cols = price_cols + size_cols

    df_norm[feature_cols] = df_norm[feature_cols].astype(float)

    # ---- Extract day (without time). NO sorting. ----
    df_norm["_date"] = df_norm[datetime_col].dt.normalize()

    unique_days = df_norm["_date"].unique()

    # Storage for means & stds row-wise
    means_df = pd.DataFrame(index=df_norm.index, columns=feature_cols, dtype=float)
    stds_df  = pd.DataFrame(index=df_norm.index, columns=feature_cols, dtype=float)

    # ---- Process day-by-day (NO SORT) ----
    for i in range(1, len(unique_days)):

        prev_day = unique_days[i - 1]
        curr_day = unique_days[i]

        prev_mask = df_norm["_date"] == prev_day
        curr_mask = df_norm["_date"] == curr_day

        prev_values = df_norm.loc[prev_mask, feature_cols].values

        # compute previous-day statistics
        mean_prev = prev_values.mean(axis=0)
        var_prev  = prev_values.var(axis=0)

        # numerical stability
        var_prev[var_prev < 0] = 0.0
        std_prev = np.sqrt(var_prev)
        std_prev[std_prev == 0] = np.nan   # avoid divide-by-zero

        # broadcast stats to all rows in current day
        means_df.loc[curr_mask, :] = mean_prev
        stds_df.loc[curr_mask, :]  = std_prev

    # ---- apply normalization ----
    std_safe = stds_df.replace(0, np.nan)
    df_norm[feature_cols] = (df_norm[feature_cols] - means_df) / std_safe

    # replace NaNs (from 1st day or std=0) → 0
    df_norm[feature_cols] = df_norm[feature_cols].fillna(0.0)

    # ---- drop 1st day if requested ----
    if drop_first_day:
        first_day = unique_days[0]
        df_norm = df_norm[df_norm["_date"] != first_day].copy()

    df_norm = df_norm.drop(columns="_date")

    return df_norm
