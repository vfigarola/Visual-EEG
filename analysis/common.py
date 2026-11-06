"""Shared helpers for the analysis scripts (behavior.py, erp_analysis.py, itc_analysis.py, stats).

Groups (day-1 sessions):  NT  |  ADHD-off (no stimulant)  |  ADHD-on (stimulant)
Within-person:            ADHD participants with both visits usable (on vs off).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # python/ folder (config, trials)
import config as C                                                 # noqa: E402

ANALYSIS_DIR = C.ROOT / "derivatives" / "analysis"
FIG_DIR = C.ROOT / "derivatives" / "figures"

GROUPS = ["NT", "ADHD-off", "ADHD-on"]
# categorical slots 1-3 of the validated default palette (pass all-pairs CVD checks)
COLORS = {"NT": "#2a78d6", "ADHD-off": "#eb6834", "ADHD-on": "#1baf7a",
          "ADHD (all)": "#e34948"}   # pooled ADHD: only ever shown next to NT (validated pair)
COMPARISONS = [("NT", "ADHD-off"), ("NT", "ADHD-on"), ("ADHD-off", "ADHD-on"), ("NT", "ADHD (all)")]
ATTEND = ["focal", "broad"]
INTERRUPTER = ["none", "early", "late"]


def sessions(include_excluded=False):
    """One row per session, built from the RESULTS file names (works without the BDFs)."""
    rows = {}
    for f in sorted(Path(C.BEH_DIR).glob("JKSA*_RESULTS.mat")):
        m = re.match(r"(JKSA\d{3})_([XM][XC])_([12])_(\d{4}-\d\d-\d\d)", f.name)
        if not m:
            continue
        subj, cond, half, date = m.groups()
        ses = f"{subj}_{cond}"
        d = rows.setdefault(ses, dict(session=ses, subject=subj, cond=cond, date=date))
        d[f"beh{half}"] = str(f)
    S = pd.DataFrame(rows.values())
    S["nt"] = S.subject.str[4:].isin(C.NEUROTYPICAL)
    S["stimulant"] = np.where(S.cond.str[0] == "M", "on", "off")
    S["caffeine"] = np.where(S.cond.str[1] == "C", "yes", "no")
    S["n_visits"] = S.groupby("subject").session.transform("count")
    S["day"] = S.groupby("subject").date.rank(method="first").astype(int)
    S["group"] = np.where(S.nt, "NT", np.where(S.stimulant == "on", "ADHD-on", "ADHD-off"))
    S["excluded"] = ""
    S.loc[S.session.isin(C.EXCLUDE_SESSIONS), "excluded"] = "group unknown"
    for s, why in C.ANALYSIS_EXCLUDE.items():
        S.loc[S.session == s, "excluded"] = why
    if not include_excluded:
        S = S[S.excluded == ""]
    return S.sort_values("session").reset_index(drop=True)


def within_pairs(S):
    """ADHD participants with both an on- and an off-stimulant session usable."""
    A = S[~S.nt]
    piv = A.pivot_table(index="subject", columns="stimulant", values="session", aggfunc="first")
    piv = piv.dropna()
    day = A.set_index("session").day
    piv["on_day"] = piv["on"].map(day)
    return piv.reset_index()


# ---------------------------------------------------------------- statistics
def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (b.mean() - a.mean()) / sp if sp > 0 else np.nan
    return d * (1 - 3 / (4 * (na + nb) - 9))


def _ols_coef(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


def group_test(y, is_b, covar=None, n_perm=10000, seed=0):
    """Difference B - A in y, adjusted for a binary covariate (OLS), with a permutation p-value.
    Group labels are permuted *within* covariate strata, so the covariate stays balanced."""
    y, is_b = np.asarray(y, float), np.asarray(is_b, int)
    cov = np.zeros_like(y) if covar is None else np.asarray(covar, float)
    X = np.c_[np.ones_like(y), is_b, cov] if covar is not None else np.c_[np.ones_like(y), is_b]
    obs = _ols_coef(y, X)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(cov == v) for v in np.unique(cov)]
    null = np.empty(n_perm)
    for k in range(n_perm):
        g = is_b.copy()
        for idx in strata:
            g[idx] = rng.permutation(g[idx])
        X[:, 1] = g
        null[k] = _ols_coef(y, X)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def signflip_test(d, n_perm=10000, seed=0):
    """One-sample / paired test of mean(d) != 0 by random sign flips."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    obs = d.mean()
    null = (rng.choice([-1, 1], size=(n_perm, len(d))) * d).mean(1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    dz = obs / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
    return obs, p, dz


def fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# ---------------------------------------------------------------- figures
def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def save(fig, name, sub):
    out = FIG_DIR / sub
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    return out / f"{name}.png"


def within_order_test(d, on_second, n_perm=10000, seed=0):
    """Within-person on - off difference d, adjusted for visit order.
    Model: d = b0 + b1*order, order = +1 if the on-stimulant visit was day 2, -1 if day 1.
    b0 = stimulant effect with practice (visit-order) effect cancelled. p via sign-flipping the
    residuals of the order-only model (Freedman-Lane)."""
    d = np.asarray(d, float)
    o = np.where(np.asarray(on_second, bool), 1.0, -1.0)
    ok = ~np.isnan(d)
    d, o = d[ok], o[ok]
    X = np.c_[np.ones_like(d), o]
    b = np.linalg.lstsq(X, d, rcond=None)[0]
    b1_red = np.linalg.lstsq(o[:, None], d, rcond=None)[0]
    fit, res = o * b1_red[0], d - o * b1_red[0]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        dp = fit + res * rng.choice([-1, 1], size=len(d))
        null[k] = np.linalg.lstsq(X, dp, rcond=None)[0][0]
    p = (np.sum(np.abs(null) >= abs(b[0])) + 1) / (n_perm + 1)
    return b[0], b[1], p


# task timing (s, re: trigger 1 = syllable-stream onset), from the RESULTS files
TIMING = dict(
    cue=-1.0,
    target=[0.0, 0.5, 1.25],
    distractor_common=[0.1, 0.6, 1.85, 2.6], distractor_rhythm=[1.35, 1.6],   # 3rd syllable: 1.35 or 1.6
    interrupter={"early": [1.0, 1.5, 2.25], "late": [1.5, 2.0, 2.75]},
)
CLUSTERS = {"left": ["PO7", "PO3", "O1"], "right": ["PO8", "PO4", "O2"]}
CLUSTER_LABEL = {"left": "Left PO7/PO3/O1 (contra to distractor)",
                 "right": "Right PO8/PO4/O2 (contra to interrupter)"}
