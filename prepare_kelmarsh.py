"""
Kelmarsh SCADA Adapter
======================
Converts raw Kelmarsh wind farm SCADA data into the format expected
by our forecasting pipeline.

Input: Raw CSV from Zenodo (10-minute resolution, 299 columns)
Output: Clean CSV with our standard columns (Timestamp, WindSpeed_m_s,
        Temperature_C, ActivePower_kW) ready for pipeline v2.2.

Options:
  - Select specific turbine(s)
  - Resample to hourly for consistency with synthetic data
  - Or keep 10-minute resolution for higher-fidelity analysis

Usage:
    python3 prepare_kelmarsh.py

Authors: Blerant Ramadani, Vangel Fustic
"""

import os
import pandas as pd
import numpy as np
import glob

# ============================================================
# CONFIGURATION
# ============================================================
# Where you extracted the Kelmarsh ZIP files
KELMARSH_DIR = 'kelmarsh_raw'

# Output
OUTPUT_DIR = 'data'

# Which turbines to use (1-6). Use [1] for single turbine, or [1,2,3] for multiple
TURBINES = [1]

# Which year(s)
YEARS = [2018]

# Resolution: 'hourly' (average to 1h) or 'native' (keep 10-min)
RESOLUTION = 'hourly'

# Rated power of Kelmarsh MM92 turbines (IMPORTANT: different from Bogdanci 2300 kW)
KELMARSH_RATED_POWER_KW = 2050


def load_turbine_data(turbine_id, year):
    """Load raw turbine data for a given turbine and year."""
    pattern = f'{KELMARSH_DIR}/Turbine_Data_Kelmarsh_{turbine_id}_{year}-01-01_*.csv'
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No data for Turbine {turbine_id}, year {year}. "
                                f"Pattern: {pattern}")
    
    # skiprows=9 skips the header comments (# Turbine: ... etc.)
    df = pd.read_csv(files[0], skiprows=9, low_memory=False)
    
    # Identify and rename columns
    date_col = df.columns[0]  # "# Date and time"
    
    # Extract only the columns we need
    key_cols = [
        date_col,
        'Wind speed (m/s)',
        'Power (kW)',
        'Nacelle ambient temperature (°C)',
        'Wind direction (°)'
    ]
    
    # Check all columns exist
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    
    df = df[key_cols].copy()
    df.columns = ['Timestamp', 'WindSpeed_m_s', 'ActivePower_kW',
                  'Temperature_C', 'WindDirection_deg']
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Turbine'] = turbine_id
    df['Year'] = year
    
    return df


def clean_data(df, turbine_id):
    """Apply quality filters."""
    n_initial = len(df)
    
    # Remove NaN rows (turbine offline or sensor issues)
    df = df.dropna(subset=['WindSpeed_m_s', 'ActivePower_kW', 'Temperature_C'])
    n_after_nan = len(df)
    
    # Physical bounds
    df = df[(df['WindSpeed_m_s'] >= 0) & (df['WindSpeed_m_s'] <= 40)]
    df = df[(df['ActivePower_kW'] >= -50) & (df['ActivePower_kW'] <= KELMARSH_RATED_POWER_KW * 1.05)]
    df = df[(df['Temperature_C'] >= -30) & (df['Temperature_C'] <= 50)]
    n_after_bounds = len(df)
    
    # Clip negative power to 0 (turbine consuming electricity in stand-by)
    df.loc[df['ActivePower_kW'] < 0, 'ActivePower_kW'] = 0
    
    # Clip wind speed below cut-in to have zero power (consistency check)
    # This removes records where there's power despite very low wind (likely sensor issue)
    suspicious = (df['WindSpeed_m_s'] < 2.0) & (df['ActivePower_kW'] > 100)
    df = df[~suspicious]
    n_final = len(df)
    
    print(f"  Turbine {turbine_id}: {n_initial} -> {n_after_nan} (removed NaN) "
          f"-> {n_after_bounds} (bounds) -> {n_final} (final)")
    print(f"    Data availability: {n_final/n_initial*100:.1f}%")
    
    return df


def resample_hourly(df):
    """Resample 10-minute data to hourly averages."""
    df = df.set_index('Timestamp')
    
    numeric_cols = ['WindSpeed_m_s', 'ActivePower_kW', 'Temperature_C', 'WindDirection_deg']
    
    # Use mean for continuous quantities
    df_hourly = df[numeric_cols].resample('1h').mean()
    
    # Drop hours where the resample produced NaN (no data in that hour)
    df_hourly = df_hourly.dropna(subset=['WindSpeed_m_s', 'ActivePower_kW'])
    
    df_hourly = df_hourly.reset_index()
    return df_hourly


def main():
    print("=" * 60)
    print("Kelmarsh SCADA Data Adapter")
    print("=" * 60)
    
    if not os.path.exists(KELMARSH_DIR):
        print(f"\n⚠ Directory '{KELMARSH_DIR}' not found.")
        print(f"Extract Kelmarsh_SCADA_2018_3084.zip into {KELMARSH_DIR}/ first.")
        print(f"Example:")
        print(f"  mkdir {KELMARSH_DIR}")
        print(f"  unzip Kelmarsh_SCADA_2018_3084.zip -d {KELMARSH_DIR}/")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_data = []
    for turbine in TURBINES:
        for year in YEARS:
            print(f"\n[Turbine {turbine}, Year {year}]")
            try:
                df = load_turbine_data(turbine, year)
                df = clean_data(df, turbine)
                
                if RESOLUTION == 'hourly':
                    df = resample_hourly(df)
                    print(f"    Resampled to hourly: {len(df)} rows")
                
                all_data.append(df)
            except (FileNotFoundError, KeyError) as e:
                print(f"    ERROR: {e}")
                continue
    
    if not all_data:
        print("\n⚠ No data loaded. Check that ZIP files are extracted correctly.")
        return
    
    # Combine all
    combined = pd.concat(all_data, ignore_index=True)
    
    # Save output
    if len(TURBINES) == 1 and len(YEARS) == 1:
        filename = f'Kelmarsh_T{TURBINES[0]}_{YEARS[0]}_{RESOLUTION}.csv'
    else:
        filename = f'Kelmarsh_combined_{RESOLUTION}.csv'
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Keep only the columns needed by pipeline v2.2
    output_cols = ['Timestamp', 'WindSpeed_m_s', 'Temperature_C', 'ActivePower_kW']
    combined[output_cols].to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(combined)}")
    print(f"Period: {combined['Timestamp'].min()} to {combined['Timestamp'].max()}")
    print()
    print("Statistics:")
    print(combined[['WindSpeed_m_s', 'ActivePower_kW', 'Temperature_C']].describe().round(2))
    print()
    capacity_factor = combined['ActivePower_kW'].mean() / KELMARSH_RATED_POWER_KW * 100
    print(f"Capacity factor: {capacity_factor:.1f}%")
    print()
    print("✅ Ready to use with forecasting_pipeline_v2_2.py")
    print(f"   Update DATA_AUTO in the pipeline to: '{output_path}'")


if __name__ == "__main__":
    main()
