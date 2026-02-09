# solar_data_preparation.py
# Standalone script to prepare the 50 MW solar station dataset for PINN training
# Creates: prepared_solar_data_for_pinn.csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

INPUT_FILE = "/home/saurav/Desktop/SolarPV/Solarcapacity50MW.csv"
OUTPUT_FILE = "prepared_solar_data_for_pinn.csv"

# Plant parameters
NOMINAL_CAPACITY_MW = 50.0

# Daytime / quality filters
MIN_IRRADIANCE_WM2 = 30.0            # stricter than 20 to avoid noisy sunrise/sunset
MIN_POWER_MW = 0.1

print("Starting data preparation...\n")

# ────────────────────────────────────────────────
# 1. Load and rename columns
# ────────────────────────────────────────────────

df = pd.read_csv(INPUT_FILE,
                 parse_dates=['Time(year-month-day h:m:s)'],
                 dayfirst=False)

df = df.rename(columns={
    'Time(year-month-day h:m:s)': 'timestamp',
    'Total solar irradiance (W/m2)': 'G',
    'Direct normal irradiance (W/m2)': 'DNI',
    'Global horizontal irradiance (W/m2)': 'GHI',
    'Air temperature  (°C) ': 'T_amb',
    'Atmosphere (hpa)': 'pressure_hPa',
    'Power (MW)': 'P'
})

df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)

print(f"Original rows: {len(df):,}")

df_day = df[
    (df['G'] >= MIN_IRRADIANCE_WM2) &
    (df['P'] >= MIN_POWER_MW)
].copy()

print(f"After daytime/power filter: {len(df_day):,} rows  ({len(df_day)/len(df)*100:.1f}%)")

# Rough physical plausibility check
max_theoretical = NOMINAL_CAPACITY_MW * (df_day['G'] / 1000) * 1.25  # +25% margin
df_day = df_day[df_day['P'] <= max_theoretical]

print(f"After plausibility filter: {len(df_day):,} rows")

# ────────────────────────────────────────────────
# 4. Compute targets
# ────────────────────────────────────────────────

# Basic normalized power (STC-like)
df_day['PR_raw'] = df_day['P'] / (df_day['G'] / 1000 * NOMINAL_CAPACITY_MW)



# Days since beginning of dataset
df_day['t_days'] = (df_day.index - df_day.index.min()).total_seconds() / 86400.0

# Day of year cyclical
df_day['doy'] = df_day.index.dayofyear.astype(float)
df_day['doy_sin'] = np.sin(2 * np.pi * df_day['doy'] / 365.25)
df_day['doy_cos'] = np.cos(2 * np.pi * df_day['doy'] / 365.25)

# Hour of day cyclical
df_day['hour'] = df_day.index.hour + df_day.index.minute / 60.0
df_day['hour_sin'] = np.sin(2 * np.pi * df_day['hour'] / 24)
df_day['hour_cos'] = np.cos(2 * np.pi * df_day['hour'] / 24)

# ────────────────────────────────────────────────
# 6. Features to keep & normalize
# ────────────────────────────────────────────────

features_to_normalize = [
    'G', 'DNI', 'GHI', 'T_amb', 'pressure_hPa',
    'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'
]

features_final = ['t_days'] + features_to_normalize + [
    'PR_raw', 'P'          # keep originals for debugging/validation
]

# Normalize (t_days is NOT normalized – kept in physical units)
scaler = MinMaxScaler()
df_day[features_to_normalize] = scaler.fit_transform(df_day[features_to_normalize])

# Round numerical noise
df_day = df_day.round(6)



df_prepared = df_day[features_final].copy()
df_prepared.to_csv(OUTPUT_FILE, index=True)

print(f"\nPrepared dataset saved → {OUTPUT_FILE}")
print(f"Rows: {len(df_prepared):,}")
print(f"Columns: {list(df_prepared.columns)}")
print("\nFirst 5 rows preview:")
print(df_prepared.head())

print("\nPreparation finished.\n")
print("You can now use this file directly for PINN training.")
