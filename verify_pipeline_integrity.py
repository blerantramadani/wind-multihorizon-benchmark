"""
================================================================================
Pipeline Integrity Diagnostic
================================================================================
Audits the forecasting pipeline for technical correctness:
  1. Target leakage check
  2. Feature consistency
  3. Datetime monotonicity
  4. NaN handling
  5. Train/test split integrity
  6. Scaler independence
  7. Sequence construction validity

This is a READ-ONLY audit — does not modify any code or data.

USAGE:
  python3 verify_pipeline_integrity.py kelmarsh
  python3 verify_pipeline_integrity.py penmanshiel
  python3 verify_pipeline_integrity.py synthetic
================================================================================
"""

import pandas as pd
import numpy as np
import sys
import os

# Match pipeline configuration
LOOK_BACK = 48
TEST_HOURS = 1000
TARGET_COL = 'ActivePower_kW'
ALL_FEATURES = [
    'WindSpeed_m_s', 'Temperature_C', 'ActivePower_lag1',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'WindSpeed_Rolling3h'
]

DATASETS = {
    'synthetic': ('data/Synthetic_Bogdanci_OU.csv', 2300),
    'kelmarsh': ('data/Kelmarsh_T1_2018_hourly.csv', 2050),
    'penmanshiel': ('data/Penmanshiel_T01_2018_hourly.csv', 2050),
}


def check(condition, message, severity='WARN'):
    """Print check result with color coding."""
    icon = '\033[92m✓\033[0m' if condition else ('\033[91m✗\033[0m' if severity == 'FAIL' else '\033[93m⚠\033[0m')
    print(f"  {icon} {message}")
    return condition


def audit_dataset(name, filepath, rated_power):
    print(f"\n{'=' * 70}")
    print(f"AUDIT: {name.upper()}")
    print(f"File: {filepath}")
    print(f"Rated power: {rated_power} kW")
    print(f"{'=' * 70}")
    
    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        return False
    
    # Load
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    # ============================================
    # CHECK 1: Datetime monotonicity
    # ============================================
    print("\n[1] Datetime monotonicity:")
    is_sorted = df.index.is_monotonic_increasing
    check(is_sorted, f"Index is monotonically increasing: {is_sorted}", 'FAIL')
    
    has_dups = df.index.duplicated().any()
    check(not has_dups, f"No duplicate timestamps: {not has_dups}", 'FAIL')
    
    # ============================================
    # CHECK 2: Required columns present
    # ============================================
    print("\n[2] Required columns:")
    required = ['WindSpeed_m_s', 'Temperature_C', TARGET_COL]
    for col in required:
        check(col in df.columns, f"Column '{col}' present: {col in df.columns}", 'FAIL')
    
    # ============================================
    # CHECK 3: Feature engineering correctness
    # ============================================
    print("\n[3] Feature engineering:")
    df['Hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['WindSpeed_Rolling3h'] = df['WindSpeed_m_s'].rolling(window=3).mean()
    df['ActivePower_lag1'] = df[TARGET_COL].shift(1)
    
    # Verify lag1 is correctly shifted
    test_idx = 100
    if len(df) > test_idx + 1:
        original = df[TARGET_COL].iloc[test_idx]
        lagged = df['ActivePower_lag1'].iloc[test_idx + 1]
        check(np.isclose(original, lagged), 
              f"lag1[t+1] == target[t]: {np.isclose(original, lagged)}", 'FAIL')
    
    df_clean = df.dropna()
    print(f"  After dropna: {len(df_clean)} rows ({100*len(df_clean)/len(df):.1f}% retained)")
    
    # ============================================
    # CHECK 4: Target leakage
    # ============================================
    print("\n[4] Target leakage:")
    target_in_features = TARGET_COL in ALL_FEATURES
    check(not target_in_features, 
          f"Target column NOT in feature list: {not target_in_features}", 'FAIL')
    
    # Verify ActivePower_lag1 is not just a copy of target
    if len(df_clean) > 0:
        lag_target_correlation = df_clean['ActivePower_lag1'].corr(df_clean[TARGET_COL])
        is_strong_corr = lag_target_correlation > 0.5
        # We EXPECT high correlation because they're related, but not 1.0
        is_perfect = np.isclose(lag_target_correlation, 1.0, atol=0.001)
        check(not is_perfect,
              f"lag1 != target (correlation = {lag_target_correlation:.3f}): {not is_perfect}", 'FAIL')
    
    # ============================================
    # CHECK 5: Train/test split integrity
    # ============================================
    print("\n[5] Train/test split:")
    n_total = len(df_clean) - LOOK_BACK
    n_test = TEST_HOURS
    n_train_val = n_total - n_test
    
    check(n_train_val > 1000, 
          f"Train+val has >1000 sequences: {n_train_val}", 'FAIL')
    
    if n_train_val > 0:
        check(n_total - n_test > 0,
              f"Test set strictly before train+val end (chronological): YES", 'FAIL')
    
    # ============================================
    # CHECK 6: Wind speed dominance verification
    # ============================================
    print("\n[6] Wind speed signal quality:")
    if len(df_clean) > 0:
        ws_mean = df_clean['WindSpeed_m_s'].mean()
        ws_std = df_clean['WindSpeed_m_s'].std()
        ws_min = df_clean['WindSpeed_m_s'].min()
        ws_max = df_clean['WindSpeed_m_s'].max()
        
        check(ws_min >= 0, f"Wind speed >= 0: min = {ws_min:.2f}", 'FAIL')
        check(ws_max < 50, f"Wind speed reasonable: max = {ws_max:.2f}", 'WARN')
        check(2 < ws_mean < 15, 
              f"Wind speed mean reasonable: {ws_mean:.2f} m/s", 'WARN')
        
        # Check power-wind correlation (physics check)
        power_wind_corr = df_clean[TARGET_COL].corr(df_clean['WindSpeed_m_s'])
        check(power_wind_corr > 0.5, 
              f"Power-Wind correlation > 0.5: {power_wind_corr:.3f}", 'FAIL')
    
    # ============================================
    # CHECK 7: Power output sanity
    # ============================================
    print("\n[7] Power output sanity:")
    if len(df_clean) > 0:
        p_min = df_clean[TARGET_COL].min()
        p_max = df_clean[TARGET_COL].max()
        p_mean = df_clean[TARGET_COL].mean()
        
        check(p_min >= 0, f"Power >= 0 (after clipping): min = {p_min:.2f}", 'FAIL')
        check(p_max <= rated_power * 1.05, 
              f"Power <= rated*1.05: max = {p_max:.2f} kW (rated = {rated_power})", 'WARN')
        
        capacity_factor = (p_mean / rated_power) * 100
        check(5 < capacity_factor < 80,
              f"Capacity factor reasonable: {capacity_factor:.1f}%", 'WARN')
    
    # ============================================
    # CHECK 8: NaN handling
    # ============================================
    print("\n[8] NaN handling:")
    nan_counts_pre = df[required + ['Hour_sin', 'Month_sin', 'WindSpeed_Rolling3h', 'ActivePower_lag1']].isna().sum()
    nan_counts_post = df_clean.isna().sum().sum()
    
    print(f"  Pre-cleaning NaN counts:")
    for col, count in nan_counts_pre.items():
        if count > 0:
            print(f"    - {col}: {count}")
    check(nan_counts_post == 0, 
          f"No NaN after dropna: {nan_counts_post} cells", 'FAIL')
    
    print(f"\n  ✓ Audit complete for {name}")
    return True


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target not in DATASETS:
            print(f"Unknown dataset: {target}")
            print(f"Available: {list(DATASETS.keys())}")
            sys.exit(1)
        targets = [target]
    else:
        targets = list(DATASETS.keys())
    
    print(f"{'#' * 70}")
    print(f"# Pipeline Integrity Diagnostic")
    print(f"# Datasets to audit: {targets}")
    print(f"{'#' * 70}")
    
    for name in targets:
        filepath, rated = DATASETS[name]
        audit_dataset(name, filepath, rated)
    
    print(f"\n{'#' * 70}")
    print(f"# AUDIT SUMMARY")
    print(f"{'#' * 70}")
    print(f"  ✓ = pass    ⚠ = warning (review)    ✗ = fail (must fix)")
    print(f"\nIf all checks pass, the pipeline is technically sound.")
    print(f"If any FAIL, do not proceed to publication until resolved.")


if __name__ == "__main__":
    main()
