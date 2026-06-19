"""
Reusable data pipeline for ASIC WLT CQ slippage prediction.

Based on the map_WLT_value_to_CQ_slippage analysis pipeline.
Maps WLT accelerometer noise measurements to CQ values and predicts
temperature-dependent failures (CP3/CP4) from CP2 room-temperature data.

Target is WLT_fail (categorical: CP2/CP3/CP4/Pass) defined by WLT limit
exceedance on the F3_ACCGMN_SD test:
  - CP2:  exceeds HPM WLT limit at CP2 (25 °C)
  - CP3:  exceeds HPM WLT_CP3 limit at CP3 (0 °C)
  - CP4:  exceeds HPM WLT_CP4 limit at CP4 (65 °C)
  - Pass: none of the above

Usage:
    from data_pipeline import load_and_prepare, engineer_features, stratified_group_split

    # parquet_paths: list of DPK456_11..17 parquet files (or a single glob pattern)
    pivoted = load_and_prepare(parquet_paths)
    data, features_clean = engineer_features(pivoted, target='WLT_fail')
    split = stratified_group_split(data, target='WLT_fail')
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

# Module-level cache for load_and_prepare results
_CACHE = {}

# --- BMI423 limit definitions ---
HPM_WLT_RT = 330            # HPM WLT limit at room temp (CP2)
HPM_WLT_CP3 = 330 * 1.25   # HPM WLT limit at cold temp (CP3) with guard-band
HPM_WLT_CP4 = 330 * 1.25   # HPM WLT limit at hot temp (CP4) with guard-band

# WLT → CQ linear mapping: WLT = CQ_SLOPE * CQ + CQ_INTERCEPT
CQ_SLOPE = 2250
CQ_INTERCEPT = 20.5

# Fail classification test (F3 accelerometer noise standard deviation)
FAIL_TEST_MODE = 'T17_62_FW_Combo_Sense_CBIST:F3_ACCGMN_SD'

# Default parquet source (7 lot-level files from BMI420 ASIC evaluation)
_PARQUET_BASE_DIR = r"G:\DfsDE\LOC\Rt\BST\09_projects\BMI420\External\02_Product_Development\03_System\04_Accel_System_Development\10_ASIC_evaluations\CA\lot1_lot2_lot3_wafermap_tp_V2"
DEFAULT_PARQUET_FILES = [
    rf"{_PARQUET_BASE_DIR}\DPK456_11.parquet",
    rf"{_PARQUET_BASE_DIR}\DPK456_12.parquet",
    rf"{_PARQUET_BASE_DIR}\DPK456_13.parquet",
    rf"{_PARQUET_BASE_DIR}\DPK456_14.parquet",
    rf"{_PARQUET_BASE_DIR}\DPK456_15.parquet",
    rf"{_PARQUET_BASE_DIR}\DPK456_16.parquet",
    rf"{_PARQUET_BASE_DIR}\DPK456_17.parquet",
]

# Columns that are never features
_NON_FEATURE_COLS = ['usn', 'wafer_id', 'WLT_fail']


def _create_usn(df):
    """Construct unique serial number per die from wafer/coord info."""
    df = df.copy()
    df['usn'] = ('0000' + df['waferid_norm'].astype(str).str[:6] +
                 df['waferid_norm'].astype(str).str.split('-').str[-1].str.zfill(3) +
                 df['x_coord'].astype(int).astype(str).str.zfill(3) +
                 df['y_coord'].astype(int).astype(str).str.zfill(3) + 'KY')
    return df


def _normalize_test_names(df):
    """Strip CP3/CP4 suffixes, remap axis channels, extract metadata."""
    df = df.copy()
    # Strip insertion suffixes so same physical test aligns across CP2/CP3/CP4
    df['test_mapped'] = df['test_txt'].str.replace('_CP3', '', regex=False)
    df['test_mapped'] = df['test_mapped'].str.replace('_CP4', '', regex=False)

    # Channel remapping [x,y,z] -> [y,z,x] to match ASIC orientation convention
    df['test_mapped'] = df['test_mapped'].str.replace('_Y[1]', '_CH_Z', regex=False)
    df['test_mapped'] = df['test_mapped'].str.replace('_X[1]', '_CH_Y', regex=False)
    df['test_mapped'] = df['test_mapped'].str.replace('_Z[1]', '_CH_X', regex=False)

    df['test_mapped'] = df['test_mapped'].str.replace('_Y_REMEAS[1]', '_CH_Z', regex=False)
    df['test_mapped'] = df['test_mapped'].str.replace('_X_REMEAS[1]', '_CH_Y', regex=False)
    df['test_mapped'] = df['test_mapped'].str.replace('_Z_REMEAS[1]', '_CH_X', regex=False)

    # Extract metadata columns
    df['axis'] = df['test_mapped'].apply(
        lambda x: 'X' if '_CH_X' in x else 'Y' if '_CH_Y' in x else 'Z' if '_CH_Z' in x else 'NaN')
    df['F_test'] = df['test_mapped'].apply(
        lambda x: 'F0' if 'F0' in x else 'F3' if 'F3' in x else 'F5' if 'F5' in x else 'F7' if 'F7' in x else 'NaN')
    df['test_mode'] = df['test_mapped'].str.replace(r'_CH_[XYZ]', '', regex=True)

    return df


def _classify_fails(df):
    """Classify parts as CP2/CP3/CP4 fail based on F3_ACCGMN_SD exceeding HPM WLT limits."""
    cp2_fail_usn = set(df[
        (df['flow_id'] == 'CP2') &
        (df['test_mode'] == FAIL_TEST_MODE) &
        (df['test_result'] > HPM_WLT_RT)
    ]['usn'].unique())

    cp3_fail_usn = set(df[
        (df['flow_id'] == 'CP3') &
        (df['test_mode'] == FAIL_TEST_MODE) &
        (df['test_result'] > HPM_WLT_CP3)
    ]['usn'].unique())

    cp4_fail_usn = set(df[
        (df['flow_id'] == 'CP4') &
        (df['test_mode'] == FAIL_TEST_MODE) &
        (df['test_result'] > HPM_WLT_CP4)
    ]['usn'].unique())

    df = df.copy()
    df['WLT_fail'] = df['usn'].map(
        lambda x: 'CP2' if x in cp2_fail_usn
        else 'CP3' if x in cp3_fail_usn
        else 'CP4' if x in cp4_fail_usn
        else 'Pass')

    return df


def _filter_acc_noise_tests(df):
    """Filter to accelerometer noise tests (T17_62/T17_65 × ACC × SD/MEAN) + hard_bin."""
    mask = df['test_txt'].apply(
        lambda t: (('T17_62' in t or 'T17_65' in t) and 'ACC' in t and ('SD' in t or 'MEAN' in t))
                  or t == 'hard_bin'
    )
    return df[mask].copy()


def load_and_prepare(parquet_paths=None):
    """Load raw WLT parquets, filter acc noise tests, normalize, classify, pivot.

    Parameters
    ----------
    parquet_paths : list of str/Path, str, Path, or None
        - If None: loads the 7 DEFAULT_PARQUET_FILES (DPK456_11..17)
        - If a list: loads and concatenates all listed parquet files
        - If a single file path: loads that one parquet

    Pipeline (from map_WLT_value_to_CQ_slippage):
      1. Concatenate lot-level parquet files
      2. Filter to accelerometer noise tests (T17_62/T17_65 × ACC × SD/MEAN)
      3. Create USN per die
      4. Filter to hard_bin in {1, 17}
      5. Normalize test names (strip CP3/CP4 suffix, remap axes)
      6. Map WLT → CQ
      7. Classify fails based on HPM WLT limits on F3_ACCGMN_SD
      8. Pivot CP2 data wide (one row per USN, columns = test measurements)
      9. Add slope features (CP2 − CP4 drift for ACCGMN tests)

    Returns the pivoted DataFrame with one row per USN, including:
        usn, wafer_id, <CP2 test features>, <slope features>,
        WLT_fail (categorical: CP2/CP3/CP4/Pass)
    """
    # --- Resolve parquet files ---
    if parquet_paths is None:
        files = [Path(f) for f in DEFAULT_PARQUET_FILES]
    elif isinstance(parquet_paths, (list, tuple)):
        files = [Path(f) for f in parquet_paths]
    else:
        files = [Path(parquet_paths)]

    if not files:
        raise FileNotFoundError(f"No parquet files found. Paths: {parquet_paths}")

    # --- Cache check (keyed on resolved file paths) ---
    cache_key = tuple(str(f.resolve()) for f in files)
    if cache_key in _CACHE:
        print(f"Using cached result ({len(files)} file(s), {len(_CACHE[cache_key])} rows)")
        return _CACHE[cache_key]

    # --- Local parquet cache (persists across kernel restarts) ---
    import hashlib
    cache_hash = hashlib.md5('|'.join(cache_key).encode()).hexdigest()[:12]
    local_cache_path = Path('_pivoted_cache') / f'pivoted_{cache_hash}.parquet'
    if local_cache_path.exists():
        pivoted = pd.read_parquet(local_cache_path)
        print(f"Loaded cached pivoted data from {local_cache_path} ({len(pivoted)} rows)")
        _CACHE[cache_key] = pivoted
        return pivoted

    print(f"Loading {len(files)} parquet file(s)...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  Raw data: {len(df)} rows")

    # --- Filter to accelerometer noise tests ---
    df = _filter_acc_noise_tests(df)
    print(f"  After ACC noise filter: {len(df)} rows")

    # Create unique serial number per die
    df = _create_usn(df)

    # Keep only good die (hard_bin 1 or 17)
    # hard_bin info comes from rows where test_txt == 'hard_bin' (test_result = bin value)
    hb_rows = df[df['test_txt'] == 'hard_bin']
    if len(hb_rows) > 0:
        good_usns = set(hb_rows[hb_rows['test_result'].isin([1, 17])]['usn'].unique())
        df = df[df['usn'].isin(good_usns)].copy()
        print(f"  After hard_bin filter (via test_txt rows): {len(df)} rows, {len(good_usns)} USNs")
    elif 'hard_bin' in df.columns:
        df = df[df['hard_bin'].isin([1, 17, 1.0, 17.0])].copy()
        print(f"  After hard_bin filter (via column): {len(df)} rows")

    # Remove the hard_bin test rows themselves (not actual measurements)
    df = df[df['test_txt'] != 'hard_bin'].copy()

    # Normalize test names and extract metadata
    df = _normalize_test_names(df)

    # Map WLT → CQ
    df['CQ_mapped'] = (df['test_result'] - CQ_INTERCEPT) / CQ_SLOPE

    # Classify fails based on WLT limits
    df = _classify_fails(df)

    # Use waferid_norm directly as group for splitting (7 lots: DPK456-11..17)
    df['wafer_id'] = df['waferid_norm'].astype(str)
    print(f"  Unique wafer_id: {sorted(df['wafer_id'].unique())}")

    # --- Pivot CP2 data wide (features = all CP2 test measurements) ---
    cp2 = df[df['flow_id'] == 'CP2'].copy()
    pivoted = cp2.pivot_table(
        index='usn',
        columns='test_mapped',
        values='test_result',
        aggfunc='first',
    ).reset_index()

    # --- Compute slope features (CP2 − CP4) for ACCGMN tests ---
    accgmn = df[df['test_mode'].str.contains('ACCGMN', na=False)].copy()
    if len(accgmn) > 0:
        slope_pivot = accgmn.pivot_table(
            index=['usn', 'test_mode', 'axis'],
            columns='flow_id',
            values='test_result',
        ).reset_index()
        if 'CP2' in slope_pivot.columns and 'CP4' in slope_pivot.columns:
            slope_pivot['slope'] = slope_pivot['CP2'] - slope_pivot['CP4']
            # Pivot slopes wide per USN
            slope_wide = slope_pivot.pivot_table(
                index='usn',
                columns=['test_mode', 'axis'],
                values='slope',
                aggfunc='first',
            )
            slope_wide.columns = ['slope_' + '_'.join(c) for c in slope_wide.columns]
            slope_wide = slope_wide.reset_index()
            pivoted = pivoted.merge(slope_wide, on='usn', how='left')

    # --- Merge target and wafer_id ---
    target_cols = df[['usn', 'wafer_id', 'WLT_fail']].drop_duplicates(subset='usn')
    pivoted = pivoted.merge(target_cols, on='usn', how='left')

    # Summary
    n_total = len(pivoted)
    vc = pivoted['WLT_fail'].value_counts()
    print(f"Loaded {n_total} dies from CP2 data")
    print("  WLT_fail distribution:")
    for label in ['Pass', 'CP2', 'CP3', 'CP4']:
        n = vc.get(label, 0)
        print(f"    {label}: {n} ({n/n_total*100:.2f}%)")

    # Save to local parquet cache for next run
    local_cache_path.parent.mkdir(exist_ok=True)
    pivoted.to_parquet(local_cache_path, index=False)
    print(f"  Saved pivoted cache to {local_cache_path}")

    _CACHE[cache_key] = pivoted
    return pivoted


def engineer_features(pivoted, target='WLT_fail', corr_threshold=0.95, near_const_threshold=1e-6):
    """Sanitize names, drop near-constant & highly correlated features.

    Returns:
        data : DataFrame ready for modelling (WLT_fail as categorical target)
        features_clean : list of surviving feature column names
    """
    # Only use features that are from CP2 (room temperature) measurements.
    # Exclude any features that are slope features (which require CP3/CP4) or contain 'slope_' in their name.
    all_features = [col for col in pivoted.columns
                    if col not in _NON_FEATURE_COLS + [target] and not col.startswith('slope_')]

    data = pivoted.dropna(subset=all_features + [target]).copy()

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


def stratified_group_split(data, target='WLT_fail', n_train_range=None):
    """Find the lot-group split that minimises fail-rate variance across splits.

    Balances the fail rate (proportion of non-Pass parts) across train/val/holdout.
    Groups by wafer_id (= lotid_norm, giving 7 lots for DPK456-11..17).

    Returns dict with keys:
        train_wafers, val_wafers, holdout_wafers,
        train_idx, val_idx, holdout_idx
    """
    groups = data['wafer_id'].values
    unique_wafers = sorted(set(groups))
    n_wafers = len(unique_wafers)
    print(f"\n{n_wafers} groups (lots) found: {unique_wafers}")

    wafer_to_idx = {w: np.where(groups == w)[0] for w in unique_wafers}

    # Fail = any non-Pass label
    fail_labels = [l for l in data[target].unique() if l != 'Pass']

    print("\nPer-lot fail rates:")
    for w in unique_wafers:
        idx = wafer_to_idx[w]
        rates_str = " | ".join(
            f"{label}: {(data[target].values[idx] == label).mean():.3f}" for label in fail_labels
        )
        print(f"  {w} ({len(idx)} dies): {rates_str}")

    # Auto-determine n_train_range if not provided
    if n_train_range is None:
        if n_wafers < 3:
            raise ValueError(
                f"Need at least 3 groups for train/val/holdout split, got {n_wafers}."
            )
        # For 7 lots: try 4-5 train, leaving 2-3 for val+holdout
        min_train = max(1, n_wafers - 4)
        max_train = n_wafers - 1  # exclusive upper bound for range()
        n_train_range = (min_train, max_train)
        print(f"  Auto n_train_range: {n_train_range} (from {n_wafers} lots)")

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
                    for label in fail_labels:
                        rates = []
                        for sw in [train_wafers, val_wafers, holdout_wafers]:
                            idx = np.concatenate([wafer_to_idx[w] for w in sw])
                            rates.append((data[target].values[idx] == label).mean())
                        score += max(rates) - min(rates)
                    if score < best_score:
                        best_score = score
                        best_split = (train_wafers, val_wafers, holdout_wafers)

    if best_split is None:
        raise ValueError(
            f"No valid split found with n_train_range={n_train_range} and {n_wafers} groups. "
            "Try adjusting n_train_range or max_groups."
        )

    train_wafers, val_wafers, holdout_wafers = best_split
    train_idx = np.concatenate([wafer_to_idx[w] for w in train_wafers])
    val_idx = np.concatenate([wafer_to_idx[w] for w in val_wafers])
    holdout_idx = np.concatenate([wafer_to_idx[w] for w in holdout_wafers])

    print("\nStratified split:")
    print(f"  Train:   {len(train_idx)} dies / {len(train_wafers)} wafers {list(train_wafers)}")
    print(f"  Val:     {len(val_idx)} dies / {len(val_wafers)} wafers {list(val_wafers)}")
    print(f"  Holdout: {len(holdout_idx)} dies / {len(holdout_wafers)} wafers {list(holdout_wafers)}")
    for label in fail_labels:
        tr = (data[target].values[train_idx] == label).mean()
        vr = (data[target].values[val_idx] == label).mean()
        hr = (data[target].values[holdout_idx] == label).mean()
        print(f"  {label} rate — Train: {tr:.4f} | Val: {vr:.4f} | Holdout: {hr:.4f}")

    return {
        'train_wafers': train_wafers, 'val_wafers': val_wafers, 'holdout_wafers': holdout_wafers,
        'train_idx': train_idx, 'val_idx': val_idx, 'holdout_idx': holdout_idx,
    }
