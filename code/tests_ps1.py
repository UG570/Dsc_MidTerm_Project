"""PS1: do street lights make rural night-time crashes less severe?

Test 1: two-proportion z-test on KSI share, no lighting (N) vs lit (L).
Test 2: paired t-test on per-district differences d_i = KSI_N - KSI_L.
Plus the alternatives discussed in the report (chi-square, CMH by road
surface, Wilcoxon signed-rank, sign test, unpaired Welch).

"""
import numpy as np
import pandas as pd
from scipy import stats

from prepare import load_clean
from stat_helpers import FIGDATA, cmh_test, save_json, two_prop_ztest

NO_LIGHT = "Darkness - no lighting"
LIT = "Darkness - lights lit"
MIN_PER_GROUP = 20           # districts need >= 20 crashes in each group
MATTERS = 0.02               # 2 percentage points


def rural_night(df):
    base = df[(df["Urban_or_Rural_Area"] == "Rural")
              & (df["Road_Type"] == "Single carriageway")]
    sub = base[base["Light_Conditions"].isin([NO_LIGHT, LIT])].copy()
    sub["grp"] = np.where(sub["Light_Conditions"] == NO_LIGHT, "N", "L")
    excluded = base["Light_Conditions"].value_counts().drop(
        [NO_LIGHT, LIT], errors="ignore")
    return base, sub, excluded


def test1(sub):
    n = sub[sub.grp == "N"]["KSI"]
    l = sub[sub.grp == "L"]["KSI"]
    res = two_prop_ztest(n.sum(), len(n), l.sum(), len(l), "greater")

    # alternatives
    table = pd.crosstab(sub["grp"], sub["KSI"]).loc[["N", "L"], [1, 0]]
    chi2, p_chi2, dof, expected = stats.chi2_contingency(
        table.values, correction=False)
    res["chi2"] = {"stat": chi2, "df": dof, "p_two_sided": p_chi2,
                   "min_expected": expected.min()}
    
    strata = []
    by_surface = {}
    for s, g in sub[sub["Road_Surface_Conditions"] != "Unknown"].groupby(
            "Road_Surface_Conditions"):
        
        a = int(g[(g.grp == "N")]["KSI"].sum())
        b = int((g.grp == "N").sum()) - a
        c = int(g[(g.grp == "L")]["KSI"].sum())
        d = int((g.grp == "L").sum()) - c

        strata.append([[a, b], [c, d]])
        by_surface[s] = {"n_N": a + b, "ksi_N": a / (a + b) if a + b else None,
                         "n_L": c + d, "ksi_L": c / (c + d) if c + d else None}
        
    res["cmh_by_surface"] = cmh_test(strata, "greater")
    res["cmh_by_surface"]["by_surface"] = by_surface
    res["odds_ratio"] = (table.iloc[0, 0] * table.iloc[1, 1]) / (
        table.iloc[0, 1] * table.iloc[1, 0])
    
    res["matters"] = MATTERS
    res["ci_entirely_above_matters"] = res["ci95"][0] > MATTERS
    res["ci_contains_matters"] = res["ci95"][0] <= MATTERS <= res["ci95"][1]

    return res


def test2(sub):
    g = sub.groupby(["District Area", "grp"])["KSI"].agg(["mean", "size"])
    g = g.unstack("grp")
    keep = (g[("size", "N")] >= MIN_PER_GROUP) & (
        g[("size", "L")] >= MIN_PER_GROUP)
    g = g[keep]
    share_n = g[("mean", "N")].values
    share_l = g[("mean", "L")].values
    d = share_n - share_l
    k = len(d)

    t = stats.ttest_1samp(d, 0.0, alternative="greater")
    se = d.std(ddof=1) / np.sqrt(k)
    tcrit = stats.t.ppf(0.975, k - 1)
    res = {
        "districts_total": int(sub["District Area"].nunique()),
        "districts_used": k,
        "crashes_in_used_districts": int(
            g[("size", "N")].sum() + g[("size", "L")].sum()),
        "mean_d": d.mean(), "median_d": np.median(d), "sd_d": d.std(ddof=1),
        "se": se, "t": t.statistic, "df": k - 1, "p_value": t.pvalue,
        "ci95": [d.mean() - tcrit * se, d.mean() + tcrit * se],
        "positive_d": int((d > 0).sum()), "negative_d": int((d < 0).sum()),
        "zero_d": int((d == 0).sum()),
        "skew_d": stats.skew(d), "shapiro_p": stats.shapiro(d).pvalue,
        "matters": MATTERS,
    }

    w = stats.wilcoxon(d, alternative="greater")
    res["wilcoxon"] = {"W": w.statistic, "p_value": w.pvalue}
    nz = d[d != 0]
    sign = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5, alternative="greater")
    res["sign_test"] = {"positives": int((nz > 0).sum()), "n": len(nz),
                        "p_value": sign.pvalue}
    u = stats.ttest_ind(share_n, share_l, equal_var=False,
                        alternative="greater")
    res["unpaired_welch"] = {"t": u.statistic, "p_value": u.pvalue}
    res["pairing_correlation"] = np.corrcoef(share_n, share_l)[0, 1]

    # histogram of d_i (percentage points)
    edges = np.arange(-30, 32.5, 5.0)
    counts, _ = np.histogram(np.clip(d * 100, -29.99, 29.99), bins=edges)
    pd.DataFrame({"left": edges[:-1], "mid": edges[:-1] + 2.5,
                  "count": counts}).to_csv(FIGDATA / "district_diff_hist.csv",
                                           index=False)
    return res


def main():
    df = load_clean()
    base, sub, excluded = rural_night(df)
    out = {
        "rural_single_rows": len(base),
        "rows_used": len(sub),
        "excluded_light_levels": excluded.to_dict(),
        "test1": test1(sub),
        "test2": test2(sub),
    }
    save_json(out, "ps1.json")
    t1, t2 = out["test1"], out["test2"]
    print(f"Test 1: p_N={t1['p1']:.4f} p_L={t1['p2']:.4f} diff={t1['diff']:.4f}"
          f" z={t1['z']:.2f} p={t1['p_value']:.3g} CI={t1['ci95']}")
    print(f"  CMH z={t1['cmh_by_surface']['z']:.2f}"
          f" p={t1['cmh_by_surface']['p_value']:.3g}"
          f" MH RD={t1['cmh_by_surface']['mh_risk_difference']:.4f}")
    print(f"Test 2: k={t2['districts_used']} mean d={t2['mean_d']:.4f}"
          f" t={t2['t']:.2f} df={t2['df']} p={t2['p_value']:.3g}"
          f" CI={t2['ci95']}")
    print(f"  Wilcoxon p={t2['wilcoxon']['p_value']:.3g}"
          f" sign p={t2['sign_test']['p_value']:.3g}"
          f" unpaired p={t2['unpaired_welch']['p_value']:.3g}")


if __name__ == "__main__":
    main()
