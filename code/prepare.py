
import re

import numpy as np
import pandas as pd

from stat_helpers import ROOT, save_json

RAW_FILE = ROOT / "accident_data.csv"

NOMINAL = ["Light_Conditions", "Road_Surface_Conditions", "Road_Type",
           "Urban_or_Rural_Area", "Weather_Conditions", "Vehicle_Type"]

NUMERIC = ["Latitude", "Longitude", "Number_of_Casualties",
           "Number_of_Vehicles"]

SEVERITY_ORDER = {"Slight": 0, "Serious": 1, "Fatal": 2}
CAR_TYPES = {"Car", "Taxi/Private hire car"}


def load_raw():
    return pd.read_csv(RAW_FILE, dtype={"Index": str})


def district_typo_map(names):
    """Some district names lost the letters 'na' (e.g. 'Stevege' for
    'Stevenage', 'Blaeu Gwent' for 'Blaenau Gwent'). Map every name that
    equals another name with one 'na' removed back to the full name."""
    names = set(names)
    mapping = {}
    for full in names:
        for m in re.finditer("(?i)na", full):
            short = full[:m.start()] + full[m.end():]
            if short in names and short != full:
                mapping[short] = full
    return mapping


def vehicle_group(v):
    if v.startswith("Motorcycle"):
        return "Motorcycle"
    if v in CAR_TYPES:
        return "Car"
    return "Other"


def clean(raw):
    
    df = raw.copy()
    log = {}

    # exact duplicate rows
    dup = df.duplicated()
    log["exact_duplicates_dropped"] = int(dup.sum())
    df = df[~dup].copy()

    # the accident ID was mangled into scientific notation (e.g.
    # '2.01E+07') for many rows, so it cannot serve as a key. We keep
    # it only to check the date shift, and never use it to match rows.
    sci = df["Index"].str.contains("E+", regex=False)
    log["index_in_sci_notation"] = int(sci.sum())
    log["index_duplicates_raw"] = int(raw["Index"].duplicated().sum())

    # dates: for rows with a readable ID the year in the ID is exactly
    # 12 years before the year in 'Accident Date', so dates are shifted.
    df["Accident Date"] = pd.to_datetime(df["Accident Date"], format="%d-%m-%Y")
    ok_id = df["Index"].str.match(r"^20\d\d") & ~sci
    gap = (df.loc[ok_id, "Accident Date"].dt.year
           - df.loc[ok_id, "Index"].str[:4].astype(int))
    log["date_minus_id_year"] = gap.value_counts().to_dict()

    # district typos
    tmap = district_typo_map(df["District Area"].unique())
    log["district_levels_before"] = int(df["District Area"].nunique())
    log["district_typo_map"] = tmap
    log["district_rows_fixed"] = int(df["District Area"].isin(tmap).sum())
    df["District Area"] = df["District Area"].replace(tmap)
    log["district_levels_after"] = int(df["District Area"].nunique())

    # missing / unusable categories -> explicit 'Unknown' level
    log["missing_before"] = raw.isna().sum()[lambda s: s > 0].to_dict()
    df["Urban_or_Rural_Area"] = df["Urban_or_Rural_Area"].replace(
        {"Unallocated": np.nan})
    df["Vehicle_Type"] = df["Vehicle_Type"].replace(
        {"Data missing or out of range": np.nan})
    for col in NOMINAL:
        df[col] = df[col].fillna("Unknown")
    log["missing_coords_kept"] = int(df["Latitude"].isna().sum())

    # outliers in the counts: the IQR fence is degenerate for
    # casualties (Q1 = Q3 = 1), so it would flag every multi-casualty
    # crash. These are real (e.g. bus crashes), so we keep them.
    for col in ["Number_of_Casualties", "Number_of_Vehicles"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        fence = q3 + 1.5 * (q3 - q1)
        log[f"{col}_iqr"] = {"q1": q1, "q3": q3, "upper_fence": fence,
                             "above_fence": int((df[col] > fence).sum()),
                             "max": int(df[col].max())}

    # derived columns used by the tests
    df["severity_ord"] = df["Accident_Severity"].map(SEVERITY_ORDER)
    df["KSI"] = (df["Accident_Severity"] != "Slight").astype(int)
    df["vehicle_group"] = df["Vehicle_Type"].map(
        lambda v: "Unknown" if v == "Unknown" else vehicle_group(v))
    return df, log


def encode(df):
    enc = pd.get_dummies(df[NOMINAL], prefix=NOMINAL, dtype=np.uint8)
    out = pd.concat([
        df[NUMERIC],
        df[["severity_ord", "KSI"]],
        df["District Area"].map(df["District Area"].value_counts(
            normalize=True)).rename("district_freq"),
        enc,
    ], axis=1)
    return out


def load_clean():
    df, _ = clean(load_raw())
    return df


def main():
    raw = load_raw()
    df, log = clean(raw)
    enc = encode(df)

    summary = {
        "raw_rows": raw.shape[0], "raw_cols": raw.shape[1],
        "raw_cells": raw.shape[0] * raw.shape[1],
        "clean_rows": df.shape[0],
        "encoded_cols": enc.shape[1],
        "encoded_cells": enc.shape[0] * enc.shape[1],
        "raw_missing_cells": int(raw.isna().sum().sum()),
        "clean_missing_cells_in_used_cols": int(
            df[NOMINAL + ["Number_of_Casualties", "Number_of_Vehicles"]]
            .isna().sum().sum()),
        "onehot_levels": {c: int(df[c].nunique()) for c in NOMINAL},
        "severity_counts": df["Accident_Severity"].value_counts().to_dict(),
        "ksi_share": df["KSI"].mean(),
        "log": log,
    }
    save_json(summary, "preparation.json")
    print(f"raw {raw.shape} -> clean {df.shape}, encoded {enc.shape}")
    print(f"size check: {enc.shape[0]:,} x {enc.shape[1]} = "
          f"{enc.shape[0] * enc.shape[1]:,}")
    print("district typo fixes:", log["district_typo_map"])


if __name__ == "__main__":
    main()
