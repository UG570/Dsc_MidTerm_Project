import numpy as np
import pandas as pd
from scipy import stats

from prepare import load_clean
from stat_helpers import FIGDATA, save_json

LIGHT_SHORT = {
    "Daylight": "Daylight",
    "Darkness - lights lit": "Lights lit",
    "Darkness - no lighting": "No lighting",
    "Darkness - lights unlit": "Lights unlit",
    "Darkness - lighting unknown": "Unknown",
}


def broad_vehicle(v):
    if v.startswith("Motorcycle"):
        return "Motorcycle"
    if v in ("Car", "Taxi/Private hire car"):
        return "Car/taxi"
    if v.startswith("Van") or v.startswith("Goods"):
        return "Van/goods"
    if v.startswith("Bus") or v.startswith("Minibus"):
        return "Bus/minibus"
    return "Other"


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))


def main():
    df = load_clean()
    out = {}

    # Q1: are crashes in the dark more severe, and does it depend on area?
    a = df[df["Urban_or_Rural_Area"].isin(["Urban", "Rural"])]
    t = (a.groupby(["Light_Conditions", "Urban_or_Rural_Area"])["KSI"]
         .mean().unstack() * 100)
    t = t.loc[list(LIGHT_SHORT)].rename(index=LIGHT_SHORT)
    t.index.name = "light"
    t.round(2).to_csv(FIGDATA / "ksi_by_light.csv")
    out["ksi_by_light_pct"] = t.round(2).to_dict()
    out["ksi_by_area_pct"] = (a.groupby("Urban_or_Rural_Area")["KSI"]
                              .mean() * 100).round(2).to_dict()

    # Q2: which vehicle types are in the most crashes?
    v = df[df["Vehicle_Type"] != "Unknown"].copy()
    v["broad"] = v["Vehicle_Type"].map(broad_vehicle)
    g = v.groupby("broad").agg(crashes=("KSI", "size"), ksi=("KSI", "mean"))
    g["share_pct"] = g["crashes"] / g["crashes"].sum() * 100
    g["ksi_pct"] = g["ksi"] * 100
    g = g.sort_values("crashes", ascending=False)
    g["crashes_k"] = g["crashes"] / 1000
    g.index.name = "vehicle"
    g[["crashes_k", "share_pct", "ksi_pct"]].round(2).to_csv(
        FIGDATA / "vehicle_counts.csv")
    out["vehicle"] = g[["crashes", "share_pct", "ksi_pct"]].round(2).to_dict()

    # Q3: how many people are hurt per crash, and does it differ by
    #     vehicle in two-vehicle crashes?
    two = df[(df["Number_of_Vehicles"] == 2)
             & df["vehicle_group"].isin(["Motorcycle", "Car"])].copy()
    two["cas"] = two["Number_of_Casualties"].clip(upper=5)
    d = (two.groupby("vehicle_group")["cas"].value_counts(normalize=True)
         .unstack(0) * 100).round(2)
    d.index = ["1", "2", "3", "4", "5+"]
    d.index.name = "casualties"
    d.to_csv(FIGDATA / "casualty_dist.csv")
    out["casualty_dist_pct"] = d.to_dict()

    # Q4: which columns carry information about severity? Cramer's V of
    #     each nominal column with KSI.
    cols = {"Urban_or_Rural_Area": "Urban/rural",
            "Light_Conditions": "Light",
            "Road_Type": "Road type",
            "Weather_Conditions": "Weather",
            "Road_Surface_Conditions": "Road surface",
            "Number_of_Vehicles": "No. of vehicles",
            "Vehicle_Type": "Vehicle type"}
    cv_rows = []
    for col, label in cols.items():
        sub = df[df[col] != "Unknown"]
        cv_rows.append({"column": label, "v": cramers_v(sub[col], sub["KSI"])})
    cv = pd.DataFrame(cv_rows).sort_values("v")
    cv.round(4).to_csv(FIGDATA / "cramers_v_ksi.csv", index=False)
    out["cramers_v_with_ksi"] = cv.set_index("column")["v"].round(4).to_dict()
    vt = v["Vehicle_Type"]
    out["cramers_v_vehicle_type_with"] = {
        c: cramers_v(vt, v[c]) for c in
        ["Urban_or_Rural_Area", "Number_of_Vehicles", "Light_Conditions",
         "Road_Type", "Accident_Severity"]}
    
    mix = pd.crosstab(v["Vehicle_Type"], v["Number_of_Vehicles"] == 1,
                      normalize="columns") * 100
    
    out["vehicle_mix_pct_single_vs_multi"] = mix.round(2).rename(
        columns={True: "single", False: "multi"}).to_dict()

    # Q5: weather and road surface
    out["ksi_by_surface_pct"] = (df.groupby("Road_Surface_Conditions")["KSI"]
                                 .mean() * 100).round(2).to_dict()
    out["ksi_by_weather_pct"] = (df.groupby("Weather_Conditions")["KSI"]
                                 .mean() * 100).round(2).to_dict()
    
    out["casualties_summary"] = df["Number_of_Casualties"].describe().to_dict()
    out["casualties_skew"] = df["Number_of_Casualties"].skew()
    out["single_casualty_pct"] = (df["Number_of_Casualties"] == 1).mean() * 100

    save_json(out, "eda.json")
    print(t.round(1))
    print(g[["crashes", "share_pct", "ksi_pct"]].round(1))
    print(d)


if __name__ == "__main__":
    main()
