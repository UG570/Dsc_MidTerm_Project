"""PS2: motorcycle crashes - fewer people hurt, but hurt worse?

Rows: two-vehicle accidents whose vehicle is a motorcycle (M, all engine
sizes) or a car (C, car + taxi/private hire).
Test 3: two-proportion z-test on KSI share, M vs C (right-sided).
Test 4: Welch t-test on casualties per accident, M vs C (left-sided).
Plus the alternatives discussed in the report.

"""
import numpy as np
import pandas as pd
from scipy import stats

from prepare import load_clean
from stat_helpers import Z975, save_json, two_prop_ztest, welch_ttest

MATTERS_KSI = 0.05          # 5 percentage points
MATTERS_CAS = 0.10          # 0.10 casualties per crash


def two_vehicle(df):
    two = df[df["Number_of_Vehicles"] == 2]
    sub = two[two["vehicle_group"].isin(["Motorcycle", "Car"])].copy()
    return two, sub


def test3(sub):
    m = sub[sub.vehicle_group == "Motorcycle"]["KSI"]
    c = sub[sub.vehicle_group == "Car"]["KSI"]
    res = two_prop_ztest(m.sum(), len(m), c.sum(), len(c), "greater")

    # chi-square on all three severity levels
    tab = pd.crosstab(sub["vehicle_group"], sub["Accident_Severity"])
    tab = tab.loc[["Motorcycle", "Car"], ["Fatal", "Serious", "Slight"]]
    chi2, p, dof, expected = stats.chi2_contingency(tab.values,
                                                     correction=False)
    res["chi2_3level"] = {"stat": chi2, "df": dof, "p_value": p,
                          "table": tab.to_dict(),
                          "min_expected": expected.min()}
    res["fatal_share"] = (tab["Fatal"] / tab.sum(axis=1)).to_dict()

    # relative risk with a CI on the log scale
    a, n1, b, n2 = m.sum(), len(m), c.sum(), len(c)
    rr = (a / n1) / (b / n2)
    se = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n2)
    res["relative_risk"] = {"rr": rr, "ci95": [rr * np.exp(-Z975 * se),
                                               rr * np.exp(Z975 * se)]}
    res["matters"] = MATTERS_KSI
    return res


def test4(sub):
    m = sub[sub.vehicle_group == "Motorcycle"]["Number_of_Casualties"].values
    c = sub[sub.vehicle_group == "Car"]["Number_of_Casualties"].values
    res = welch_ttest(m, c, alternative="less")

    # equal-variance checks and the pooled t-test
    lev = stats.levene(m, c, center="median")
    f = m.var(ddof=1) / c.var(ddof=1)
    p_f = 2 * min(stats.f.cdf(f, len(m) - 1, len(c) - 1),
                  stats.f.sf(f, len(m) - 1, len(c) - 1))
    pooled = stats.ttest_ind(m, c, equal_var=True, alternative="less")
    res["levene"] = {"W": lev.statistic, "p_value": lev.pvalue}
    res["f_test"] = {"F": f, "df": [len(m) - 1, len(c) - 1], "p_value": p_f}
    res["pooled_t"] = {"t": pooled.statistic, "df": len(m) + len(c) - 2,
                       "p_value": pooled.pvalue}
    z = res["diff"] / res["se"]
    res["z_test"] = {"z": z, "p_value": stats.norm.cdf(z)}

    # Mann-Whitney U
    mw = stats.mannwhitneyu(m, c, alternative="less")
    res["mann_whitney"] = {
        "U": mw.statistic, "p_value": mw.pvalue,
        "prob_superiority": mw.statistic / (len(m) * len(c)),
        "share_tied_at_1": float(np.mean(np.r_[m, c] == 1)),
    }

    # Poisson rate test
    km, kc = m.sum(), c.sum()
    pois = stats.binomtest(int(km), int(km + kc), len(m) / (len(m) + len(c)),
                           alternative="less")
    res["poisson"] = {
        "dispersion_M": m.var(ddof=1) / m.mean(),
        "dispersion_C": c.var(ddof=1) / c.mean(),
        "dispersion_M_minus1": (m - 1).var(ddof=1) / (m - 1).mean(),
        "dispersion_C_minus1": (c - 1).var(ddof=1) / (c - 1).mean(),
        "rate_test_p": pois.pvalue,
        "min_value": int(min(m.min(), c.min())),
    }

    # sensitivity: drop crashes with more than 10 casualties
    cap = 10
    s = welch_ttest(m[m <= cap], c[c <= cap], alternative="less")
    res["welch_le10"] = {"diff": s["diff"], "t": s["t"],
                         "p_value": s["p_value"],
                         "dropped": int((m > cap).sum() + (c > cap).sum())}
    res["max_M"], res["max_C"] = int(m.max()), int(c.max())
    res["matters"] = MATTERS_CAS
    return res


def main():
    df = load_clean()
    two, sub = two_vehicle(df)
    out = {
        "two_vehicle_rows": len(two),
        "rows_used": len(sub),
        "group_sizes": sub["vehicle_group"].value_counts().to_dict(),
        "motorcycle_levels": sub[sub.vehicle_group == "Motorcycle"][
            "Vehicle_Type"].value_counts().to_dict(),
        "test3": test3(sub),
        "test4": test4(sub),
    }
    # does the vehicle split behave differently in the dark/rural subset?
    rur = sub[sub["Urban_or_Rural_Area"] == "Rural"]
    out["rural_ksi"] = rur.groupby("vehicle_group")["KSI"].mean().to_dict()
    save_json(out, "ps2.json")
    t3, t4 = out["test3"], out["test4"]
    print(f"Test 3: p_M={t3['p1']:.4f} p_C={t3['p2']:.4f} diff={t3['diff']:.4f}"
          f" z={t3['z']:.2f} p={t3['p_value']:.3g} CI={t3['ci95']}")
    print(f"  chi2(3-level)={t3['chi2_3level']['stat']:.2f}"
          f" p={t3['chi2_3level']['p_value']:.3g}"
          f" RR={t3['relative_risk']['rr']:.3f}")
    print(f"Test 4: mean_M={t4['mean1']:.4f} mean_C={t4['mean2']:.4f}"
          f" diff={t4['diff']:.4f} t={t4['t']:.2f} df={t4['df']:.0f}"
          f" p={t4['p_value']:.3g} CI={t4['ci95']}")
    print(f"  Levene p={t4['levene']['p_value']:.3g}"
          f" MW p={t4['mann_whitney']['p_value']:.3g}"
          f" dispersion M={t4['poisson']['dispersion_M']:.2f}"
          f" C={t4['poisson']['dispersion_C']:.2f}")


if __name__ == "__main__":
    main()
