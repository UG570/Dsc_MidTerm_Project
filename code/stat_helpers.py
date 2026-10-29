import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDATA = ROOT / "report" / "figdata"
RESULTS.mkdir(exist_ok=True)
FIGDATA.mkdir(parents=True, exist_ok=True)

Z975 = stats.norm.ppf(0.975)


def _p_from_z(z, alternative):
    if alternative == "greater":
        return stats.norm.sf(z)
    if alternative == "less":
        return stats.norm.cdf(z)
    return 2 * stats.norm.sf(abs(z))


def two_prop_ztest(x1, n1, x2, n2, alternative="greater"):
    """Two-proportion z-test for p1 - p2.

    The statistic uses the pooled SE (the SE under H0: p1 = p2);
    the 95% CI uses the unpooled SE (no assumption that H0 is true).
    """
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se_pooled = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    z = diff / se_pooled
    return {
        "x1": int(x1), "n1": int(n1), "p1": p1,
        "x2": int(x2), "n2": int(n2), "p2": p2,
        "diff": diff, "z": z, "p_value": _p_from_z(z, alternative),
        "alternative": alternative,
        "se_pooled": se_pooled, "se_unpooled": se_unpooled,
        "ci95": [diff - Z975 * se_unpooled, diff + Z975 * se_unpooled],
        "chi2_equiv": z ** 2,
    }


def cmh_test(tables, alternative="greater"):
    """Cochran-Mantel-Haenszel test over K strata of 2x2 tables.

    Each table is [[a, b], [c, d]]: rows = group 1 / group 2,
    columns = event / no event. Returns the signed CMH z (no continuity
    correction), the MH common odds ratio and the MH risk difference.
    """
    sum_a = sum_e = sum_v = 0.0
    or_num = or_den = 0.0
    rd_num = rd_w = 0.0
    for (a, b), (c, d) in tables:
        n1, n2 = a + b, c + d
        m1, m0 = a + c, b + d
        n = n1 + n2
        if n1 == 0 or n2 == 0 or n < 2:
            continue
        sum_a += a
        sum_e += n1 * m1 / n
        sum_v += n1 * n2 * m1 * m0 / (n ** 2 * (n - 1))
        or_num += a * d / n
        or_den += b * c / n
        w = n1 * n2 / n
        rd_num += w * (a / n1 - c / n2)
        rd_w += w

    z = (sum_a - sum_e) / np.sqrt(sum_v)
    
    return {
        "z": z, "chi2": z ** 2, "df": 1,
        "p_value": _p_from_z(z, alternative),
        "mh_odds_ratio": or_num / or_den,
        "mh_risk_difference": rd_num / rd_w,
        "strata": len(tables),
    }


def welch_ttest(x, y, alternative="less"):
    """Welch two-sample t-test for mean(x) - mean(y) with a 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    diff = x.mean() - y.mean()
    res = stats.ttest_ind(x, y, equal_var=False, alternative=alternative)
    tcrit = stats.t.ppf(0.975, df)
    return {
        "n1": n1, "n2": n2, "mean1": x.mean(), "mean2": y.mean(),
        "sd1": np.sqrt(v1), "sd2": np.sqrt(v2),
        "diff": diff, "se": se, "t": res.statistic, "df": df,
        "p_value": res.pvalue, "alternative": alternative,
        "ci95": [diff - tcrit * se, diff + tcrit * se],
    }


def holm(pvalues, alpha=0.05):
    """Holm step-down correction. `pvalues` is a dict name -> p."""
    names = sorted(pvalues, key=pvalues.get)
    m = len(names)
    out, running_max, still_rejecting = {}, 0.0, True
    for i, name in enumerate(names):
        adj = min(1.0, (m - i) * pvalues[name])
        running_max = max(running_max, adj)
        still_rejecting = still_rejecting and pvalues[name] <= alpha / (m - i)
        out[name] = {"p_raw": pvalues[name], "p_holm": running_max,
                     "threshold": alpha / (m - i),
                     "reject_holm": bool(still_rejecting)}
    return out


def _to_builtin(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(obj, name):
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, default=_to_builtin))
    print(f"  wrote {path.relative_to(ROOT)}")


def load_json(name):
    return json.loads((RESULTS / name).read_text())
