"""
Reusable data pipeline for ASIC WLT data exploration.

Usage:
    from data_pipeline import load_and_prepare, engineer_features, stratified_group_split

    # Step 1: Load & clean
    pivoted = load_and_prepare(parquet_path)

    # Step 2: Feature engineering (redundancy removal)
    data, features_clean = engineer_features(pivoted, targets=['hot_part_fail', 'cold_part_fail'])

    # Step 3: Stratified wafer-group split
    split = stratified_group_split(data, targets=['hot_part_fail', 'cold_part_fail'])
"""

import re
import numpy as np
import pandas as pd
from itertools import combinations

# Columns always dropped from the raw parquet
_DROP_COLS = ['prod_norm', 'prod_idx', 'lotid_norm', 'test_cod', 'waferid_norm']

# Default CP2 job name used for pivoting
_CP2_JOB = 'Herschel_CA_CP2_V2'

# Columns that are never features
_NON_FEATURE_COLS = ['wafer_id', 'cold_hard_bin', 'cold_soft_bin',
                     'hot_hard_bin', 'hot_soft_bin']


def load_and_prepare(parquet_path, cp2_job=_CP2_JOB, drop_cols=None):
    """Load parquet, clean, pivot CP2 tests, and create fail targets.

    Returns the pivoted DataFrame with one row per die, including columns:
        wafer_id, x_coord, y_coord, <test features>,
        cold_hard_bin, cold_soft_bin, hot_hard_bin, hot_soft_bin,
        cold_part_fail, hot_part_fail
    """
    drop_cols = drop_cols or _DROP_COLS
    df_raw = pd.read_parquet(parquet_path)
    df = df_raw.drop(columns=drop_cols)

    # Extract cold / hot bin labels per die
    cold = (
        df[df['job_name'].str.contains('COLD', case=False, na=False)]
        .drop_duplicates(subset=['wafer_id', 'x_coord', 'y_coord'])
        [['wafer_id', 'x_coord', 'y_coord', 'hard_bin', 'soft_bin']]
        .rename(columns={'hard_bin': 'cold_hard_bin', 'soft_bin': 'cold_soft_bin'})
    )
    hot = (
        df[df['job_name'].str.contains('HOT', case=False, na=False)]
        .drop_duplicates(subset=['wafer_id', 'x_coord', 'y_coord'])
        [['wafer_id', 'x_coord', 'y_coord', 'hard_bin', 'soft_bin']]
        .rename(columns={'hard_bin': 'hot_hard_bin', 'soft_bin': 'hot_soft_bin'})
    )

    df = df.merge(cold, on=['wafer_id', 'x_coord', 'y_coord'], how='left')
    df = df.merge(hot, on=['wafer_id', 'x_coord', 'y_coord'], how='left')

    # Pivot test_txt → test_result for CP2 rows
    cp2 = df[df['job_name'] == cp2_job]
    pivoted = cp2.pivot_table(
        index=['wafer_id', 'x_coord', 'y_coord'],
        columns='test_txt',
        values='test_result',
        aggfunc='first',
    ).reset_index()

    # Merge bin columns (one row per die)
    bins = df[['wafer_id', 'x_coord', 'y_coord',
               'cold_hard_bin', 'cold_soft_bin',
               'hot_hard_bin', 'hot_soft_bin']].drop_duplicates()
    pivoted = pivoted.merge(bins, on=['wafer_id', 'x_coord', 'y_coord'], how='left')

    # Binary fail targets
    pivoted['cold_part_fail'] = ~((pivoted['cold_hard_bin'] == 1) & (pivoted['cold_soft_bin'] == 1))
    pivoted['hot_part_fail'] = ~((pivoted['hot_hard_bin'] == 1) & (pivoted['hot_soft_bin'] == 1))

    return pivoted


def engineer_features(pivoted, targets, corr_threshold=0.95, near_const_threshold=1e-6):
    """Sanitize names, drop near-constant & highly correlated features.

    Returns:
        data : DataFrame ready for modelling (targets cast to str '0'/'1')
        features_clean : list of surviving feature column names
    """
    all_features = [col for col in pivoted.columns
                    if col not in _NON_FEATURE_COLS + targets]

    data = pivoted.dropna(subset=all_features + targets).copy()
    for t in targets:
        data[t] = data[t].astype(int).astype(str)

    # Sanitize feature names for tree-based models
    clean_names = {col: re.sub(r'[\[\]<]', '_', col) for col in all_features}
    data = data.rename(columns=clean_names)
    all_features_clean = [clean_names[f] for f in all_features]

    # Drop near-constant columns
    stds = data[all_features_clean].std()
    near_const = stds[stds < near_const_threshold].index.tolist()
    surviving = [f for f in all_features_clean if f not in near_const]

    # Drop one of each pair with |Pearson r| > threshold
    corr_matrix = data[surviving].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    to_drop_corr = set()
    for col in upper.columns:
        for h in upper.index[upper[col] > corr_threshold]:
            if col not in to_drop_corr:
                to_drop_corr.add(h)
    surviving = [f for f in surviving if f not in to_drop_corr]

    print(f"Feature reduction: {len(all_features_clean)} -> {len(surviving)} "
          f"({len(near_const)} near-constant, {len(to_drop_corr)} correlated)")

    return data, surviving


def stratified_group_split(data, targets, n_train_range=(3, 5)):
    """Find the wafer-group split that minimises fail-rate variance across splits.

    Returns dict with keys:
        train_wafers, val_wafers, holdout_wafers,
        train_idx, val_idx, holdout_idx
    """
    groups = data['wafer_id'].values
    unique_wafers = sorted(set(groups))
    wafer_to_idx = {w: np.where(groups == w)[0] for w in unique_wafers}

    print("\nPer-wafer fail rates:")
    for w in unique_wafers:
        idx = wafer_to_idx[w]
        rates_str = " | ".join(
            f"{t}: {(data[t].values[idx] == '1').mean():.3f}" for t in targets
        )
        print(f"  {w} ({len(idx)} dies): {rates_str}")

    best_split, best_score = None, float('inf')
    for n_train in range(*n_train_range):
        for train_wafers in combinations(unique_wafers, n_train):
            remaining = [w for w in unique_wafers if w not in train_wafers]
            for n_val in range(1, len(remaining)):
                for val_wafers in combinations(remaining, n_val):
                    holdout_wafers = tuple(w for w in remaining if w not in val_wafers)
                    if not holdout_wafers:
                        continue
                    score = 0.0
                    for t in targets:
                        rates = []
                        for sw in [train_wafers, val_wafers, holdout_wafers]:
                            idx = np.concatenate([wafer_to_idx[w] for w in sw])
                            rates.append((data[t].values[idx] == '1').mean())
                        score += max(rates) - min(rates)
                    if score < best_score:
                        best_score = score
                        best_split = (train_wafers, val_wafers, holdout_wafers)

    train_wafers, val_wafers, holdout_wafers = best_split
    train_idx = np.concatenate([wafer_to_idx[w] for w in train_wafers])
    val_idx = np.concatenate([wafer_to_idx[w] for w in val_wafers])
    holdout_idx = np.concatenate([wafer_to_idx[w] for w in holdout_wafers])

    print(f"\nStratified split:")
    print(f"  Train:   {len(train_idx)} dies / {len(train_wafers)} wafers {list(train_wafers)}")
    print(f"  Val:     {len(val_idx)} dies / {len(val_wafers)} wafers {list(val_wafers)}")
    print(f"  Holdout: {len(holdout_idx)} dies / {len(holdout_wafers)} wafers {list(holdout_wafers)}")
    for t in targets:
        tr = (data[t].values[train_idx] == '1').mean()
        vr = (data[t].values[val_idx] == '1').mean()
        hr = (data[t].values[holdout_idx] == '1').mean()
        print(f"  {t} fail rate — Train: {tr:.4f} | Val: {vr:.4f} | Holdout: {hr:.4f}")

    return {
        'train_wafers': train_wafers, 'val_wafers': val_wafers, 'holdout_wafers': holdout_wafers,
        'train_idx': train_idx, 'val_idx': val_idx, 'holdout_idx': holdout_idx,
    }
