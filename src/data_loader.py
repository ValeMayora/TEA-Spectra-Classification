import pandas as pd
import numpy as np
import re
from pathlib import Path
import csv
from io import StringIO

def load_uvvis_data(base_path="data/01_02_TEA-Spectra"):

    trials_path = Path(base_path) / "The_trials.csv"
    meta_path = Path(base_path) / "uv-vis sample description.csv"

    # --- Load spectral data (skip instrument metadata rows) ---
    # Find the row that contains the "x" header
    with open(trials_path, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.count(",") >= 2:
            header_idx = i
            break
    

    if header_idx is None:
        raise ValueError("Could not find header row starting with 'x'")
    
    

    # Skip the first header_idx lines and parse with csv module to avoid pandas parser issues
    data_lines = lines[header_idx:]
    data_lines = [line.rstrip('\r\n') for line in data_lines]
    reader = csv.reader(data_lines, delimiter=',')
    rows = list(reader)
    df_trials = pd.DataFrame(rows[1:], columns=rows[0])
    # Rename first col (spectral wavelength)
    df_trials.rename(columns={"x": "wavelength"}, inplace=True)

    # Drop Capture + colorimetry if present
    cols_to_drop = [c for c in df_trials.columns if c.lower() in ("capture", "colorimetry")]
    df_trials = df_trials.drop(columns=cols_to_drop, errors="ignore")

    # Extract wavelengths
    wavelengths = df_trials["wavelength"].values

    # All sample columns
    sample_cols = [c for c in df_trials.columns if c != "wavelength"]

    # --- Normalize sample column names ---
    # Convert A-1_a → A1_a, B_2-b → B2_b, etc.
    def normalize_name(col):
        col = col.strip()
        # Replace dash variants 
        col = col.replace("-", "_").replace("__", "_")
        # Now expected format: A_1_a or B_2_b
        parts = col.split("_")
        if len(parts) >= 2 and parts[0].isalpha() and parts[1].isdigit():
            return parts[0] + parts[1] + "_" + parts[-1]   # A + 1 → A1, keep replicate
        else:
            return col

    normalized_cols = {c: normalize_name(c) for c in sample_cols}
    df_trials.rename(columns=normalized_cols, inplace=True)

    # --- Now load metadata ---
    df_meta = pd.read_csv(meta_path, sep=";")

    # Drop empty rows at bottom
    df_meta = df_meta.dropna(subset=["sample"])

    # Exclude D3, D4
    df_meta = df_meta[~df_meta["sample"].isin(["D3", "D4"])]

    # --- Construct final matrix X and metadata ---
    X = []
    sample_ids = []
    for col in df_trials.columns:
        if col == "wavelength":
            continue

        base = col.split("_")[0]  # e.g., A1 from A1_a
        replicate = col.split("_")[-1]  # a, b, c

        if base not in df_meta["sample"].values:
            # Skip sample not in metadata (e.g., D3/D4)
            continue

        sample_ids.append(col)
        X.append(df_trials[col].values)

    X = np.array(X)  # shape (n_samples, n_wavelengths)

    # Create a metadata table aligned with X
    meta_list = []
    for sid in sample_ids:
        base = sid.split("_")[0]
        replicate = sid.split("_")[-1]
        row = df_meta[df_meta["sample"] == base].iloc[0]
        meta_list.append({
            "sample_id": sid,
            "sample_base": base,
            "replicate": replicate,
            "brand": row["brand"],
            "infusion": row["Infusion time (min)"],
            "sugar": row["sugar content (g/l)"],
        })

    df_final_meta = pd.DataFrame(meta_list)

    return X, wavelengths, df_final_meta
