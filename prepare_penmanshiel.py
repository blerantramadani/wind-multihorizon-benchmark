"""
================================================================================
Penmanshiel SCADA Adapter v1.1 — adapted to actual Zenodo folder structure
================================================================================
EXPECTED FOLDER STRUCTURE:
  data/penmanshiel_raw/Penmanshiel_SCADA_2018_WT01-10_3113/
    Turbine_Data_Penmanshiel_01_2018-01-01_-_2019-01-01_NNNN.csv
    ...

USAGE:
  python3 prepare_penmanshiel.py 01 2018
================================================================================
"""

import pandas as pd
import numpy as np
import os
import glob
import sys

RAW_DATA_DIR = 'data/penmanshiel_raw'
OUTPUT_DIR = 'data'
DEFAULT_TURBINE_ID = '01'
DEFAULT_YEAR = 2018
MAX_GAP_HOURS = 3


def find_turbine_file(turbine_id, year):
    patterns = [
        os.path.join(RAW_DATA_DIR, f'Penmanshiel_SCADA_{year}_*',
                     f'Turbine_Data_Penmanshiel_{turbine_id}_{year}-*.csv'),
        os.path.join(RAW_DATA_DIR, '*',
                     f'Turbine_Data_Penmanshiel_{turbine_id}_{year}-*.csv'),
        os.path.join(RAW_DATA_DIR,
                     f'Turbine_Data_Penmanshiel_{turbine_id}_{year}-*.csv'),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"\nNo Turbine_Data file found for Turbine {turbine_id}, Year {year}.\n"
        f"Searched:\n  " + "\n  ".join(patterns) + "\n"
    )


def load_and_clean_raw(filepath):
    print(f"Loading: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, skiprows=9, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    cols = df.columns.tolist()
    ts_col = next((c for c in cols if 'date' in c.lower() and 'time' in c.lower()), None)
    ws_col = next((c for c in cols if 'wind speed' in c.lower() and '(m/s)' in c.lower()), None)
    if not ws_col:
        ws_col = next((c for c in cols if 'wind speed' in c.lower()), None)
    temp_col = next((c for c in cols
                     if 'ambient temperature' in c.lower() and ('°c' in c.lower() or 'celsius' in c.lower())), None)
    if not temp_col:
        temp_col = next((c for c in cols if 'ambient temperature' in c.lower()), None)
    if not temp_col:
        temp_candidates = [c for c in cols if 'temperature' in c.lower()
                          and not any(x in c.lower() for x in ['gear', 'gen', 'oil', 'bearing', 'rotor'])]
        if temp_candidates:
            temp_col = temp_candidates[0]
    power_col = next((c for c in cols if c.lower().strip() == 'power (kw)'), None)
    if not power_col:
        power_col = next((c for c in cols if 'power' in c.lower()
                         and 'kw' in c.lower() and 'reactive' not in c.lower()
                         and 'setpoint' not in c.lower() and 'min' not in c.lower()
                         and 'max' not in c.lower() and 'stddev' not in c.lower()), None)

    print(f"  Selected columns:")
    print(f"    Timestamp:   {ts_col}")
    print(f"    Wind speed:  {ws_col}")
    print(f"    Temperature: {temp_col}")
    print(f"    Power:       {power_col}")

    if not all([ts_col, ws_col, temp_col, power_col]):
        print(f"\nERROR: Missing columns. Available wind/temp/power columns:")
        for c in cols:
            if any(kw in c.lower() for kw in ['wind', 'temp', 'power']):
                print(f"  - {c}")
        sys.exit(1)

    out = pd.DataFrame({
        'Timestamp': pd.to_datetime(df[ts_col], errors='coerce'),
        'WindSpeed_m_s': pd.to_numeric(df[ws_col], errors='coerce'),
        'Temperature_C': pd.to_numeric(df[temp_col], errors='coerce'),
        'ActivePower_kW': pd.to_numeric(df[power_col], errors='coerce')
    })
    out = out.dropna(subset=['Timestamp'])
    return out


def resample_to_hourly(df_10min):
    df = df_10min.set_index('Timestamp').sort_index()
    hourly = df.resample('1h').agg({
        'WindSpeed_m_s': 'mean',
        'Temperature_C': 'mean',
        'ActivePower_kW': 'mean'
    })
    return hourly.reset_index()


def quality_control(df_hourly):
    print(f"\nQuality control:")
    print(f"  Initial rows: {len(df_hourly):,}")
    initial = len(df_hourly)
    df_hourly = df_hourly.dropna(subset=['WindSpeed_m_s', 'ActivePower_kW'], how='all')
    print(f"  After dropping all-NaN rows: {len(df_hourly):,} (-{initial - len(df_hourly)})")
    n_neg = (df_hourly['ActivePower_kW'] < 0).sum()
    df_hourly.loc[df_hourly['ActivePower_kW'] < 0, 'ActivePower_kW'] = 0
    print(f"  Negative power clipped: {n_neg}")
    fault = (df_hourly['WindSpeed_m_s'] < 2) & (df_hourly['ActivePower_kW'] > 100)
    n_fault = fault.sum()
    df_hourly = df_hourly[~fault]
    print(f"  Sensor fault records removed: {n_fault}")
    n_before = df_hourly.isna().sum().sum()
    df_hourly = df_hourly.interpolate(method='linear', limit=MAX_GAP_HOURS)
    n_after = df_hourly.isna().sum().sum()
    print(f"  Interpolated cells: {n_before - n_after}")
    initial = len(df_hourly)
    df_hourly = df_hourly.dropna()
    print(f"  Final rows: {len(df_hourly):,} (-{initial - len(df_hourly)})")
    return df_hourly


def report_statistics(df, label="Final dataset"):
    print(f"\n{'=' * 70}")
    print(f"STATISTICS: {label}")
    print(f"{'=' * 70}")
    print(f"  Records: {len(df):,}")
    if len(df) == 0:
        return
    print(f"  Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"  Wind speed:   mu={df['WindSpeed_m_s'].mean():.2f}, sigma={df['WindSpeed_m_s'].std():.2f} m/s")
    print(f"  Temperature:  mu={df['Temperature_C'].mean():.2f} C")
    print(f"  Active Power: mu={df['ActivePower_kW'].mean():.2f} kW")
    rated_power = 2050
    cf = (df['ActivePower_kW'].mean() / rated_power) * 100
    print(f"  Capacity factor: {cf:.1f}% (rated: {rated_power} kW)")


def main():
    turbine_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TURBINE_ID
    year = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_YEAR
    if len(turbine_id) == 1:
        turbine_id = f"0{turbine_id}"

    print(f"{'#' * 70}")
    print(f"# Penmanshiel SCADA Adapter v1.1")
    print(f"# Turbine: WT{turbine_id}, Year: {year}")
    print(f"{'#' * 70}\n")

    try:
        filepath = find_turbine_file(turbine_id, year)
    except FileNotFoundError as e:
        print(e)
        return 1

    df_raw = load_and_clean_raw(filepath)
    df_year = df_raw[df_raw['Timestamp'].dt.year == year].copy()
    print(f"  Filtered to {year}: {len(df_year):,} records")

    if len(df_year) < 1000:
        print(f"\nWARNING: Only {len(df_year)} records. Aborting.")
        return 1

    df_hourly = resample_to_hourly(df_year)
    print(f"\nResampled to hourly: {len(df_hourly):,} rows")
    df_clean = quality_control(df_hourly)
    report_statistics(df_clean, f"Penmanshiel WT{turbine_id}, Year {year}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIR, f'Penmanshiel_T{turbine_id}_{year}_hourly.csv')
    df_clean.to_csv(output_filepath, index=False)
    print(f"\n{'=' * 70}")
    print(f"SAVED: {output_filepath}")
    print(f"{'=' * 70}")
    print(f"\nNext: Add 'penmanshiel' to DATASET_CONFIG in forecasting_pipeline_v2_3.py")
    print(f"      Then run: python3 forecasting_pipeline_v2_3.py penmanshiel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
