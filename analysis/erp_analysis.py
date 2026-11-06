"""ERPs: attentional modulation in the visual attention-switching task.

Run from the python/ folder (MNE environment active):
    python analysis/erp_analysis.py              # export averages (MNE) + measures, stats, figures
    python analysis/erp_analysis.py --skip-export  # reuse exported averages (numpy only)

Step 1 (MNE): per session, average the cleaned epochs for each of the 6 conditions, for two trial sets
        ("correct" = recalled correctly, main; "all" = all clean trials, check). Saved to
        derivatives/analysis/erp/evoked/<session>_evoked.npz (64 scalp channels, uV).
Step 2 (numpy): measures, statistics, figures.

Measures (one pooled PO/O cluster = PO7/PO3/O1 + PO8/PO4/O2; broad - focal = "delta"; µV). Left and right
cluster values are still written to erp_subject_measures.csv; the only per-hemisphere measure tested is
  N1 lateralization  (right - left) of the collapsed interrupter-syllable N1 delta; negative = larger
          modulation over the hemisphere contralateral to the interrupter (left visual field). Own family.
  N1      mean amplitude in a 50-ms window at ONE syllable-N1 latency, found on the response averaged over
          every target and interrupter syllable, both conditions, both clusters and all participants (so
          the latency is blind to every contrast), minus the mean of the 50 ms before that syllable.
          Cue N1: same, at its own latency (the cue is a different stimulus).
  N1 (collapsed) same N1, measured on the syllable-locked ERP averaged over syllables: target syllables
          pooled across all 3 interrupter conditions up to the interrupter (more trials), interrupter syllables
          pooled over early + late. Own FDR family ("N1_collapsed").
  cue     broad - focal at the cue N1. The two cues are different pictures, so this is a cue-processing
          difference, not attention to a stream.
  prep    mean 0-1.0 s after stream onset (before any interrupter, so the 3 interrupter conditions are
          pooled), re the pre-cue baseline: slow shift while preparing to switch.
  P3      mean 0.25-0.45 s after the first interrupter syllable, minus the 100 ms before it.
  lateneg mean 1.0-1.4 s after the first interrupter syllable, re the pre-cue baseline.
Each component is its own FDR family (Benjamini-Hochberg across clusters / interrupter timings).

Comparisons: NT vs ADHD-off, NT vs ADHD-on, ADHD-off vs ADHD-on, NT vs pooled ADHD (day 1; caffeine-
adjusted permutation tests), and within-person on vs off (12 pairs; visit order modelled).
Figures show participant means only (no error shading), low-pass 20 Hz for display; all measures and
cluster tests use the unsmoothed 0.5-40 Hz data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt

import common as K
import config as C

OUT = K.ANALYSIS_DIR / "erp"
EVK = OUT / "evoked"
SUB = "erp"
N_PERM = 5000
VERSIONS = ["correct", "all"]
MAIN = "correct"
CONDS = [f"{a}_{i}" for a in K.ATTEND for i in K.INTERRUPTER]
SIDES = ("pooled",)            # primary measures and figures use one pooled PO/O cluster
POOLED_LABEL = "PO7/PO3/O1 + PO8/PO4/O2"
T_WIN = (-1.6, 3.6)            # stored / plotted window
N1_SEARCH = (0.10, 0.25)       # s after onset
N1_HALF = 0.025                # 50-ms measurement window
PRE = 0.05                     # local pre-onset reference for N1
PREP_WIN = (0.0, 1.0)          # s re stream onset
P3_WIN, P3_PRE = (0.25, 0.45), 0.10   # s re first interrupter syllable
LATENEG_WIN = (1.0, 1.4)       # s re first interrupter syllable
DISPLAY_LP = 20.0              # Hz, figures only
SYL_WIN = (-0.10, 0.45)        # s, syllable-locked window (collapsed-syllable ERPs)
STREAMS = {"target": "Target syllables", "interrupter": "Interrupter syllables"}
MAIN_GROUPS = ["NT", "ADHD (all)"]   # main figures; ADHD-off / ADHD-on / within-person go to supplementary
FOCAL_COLOR = "#e34948"        # focal (ignore interrupter) = red
BROAD_COLOR = "#2a78d6"        # broad (prepare to switch)  = blue   (validated colour pair)


# ================================================================= step 1: export (MNE)
def export(S, overwrite=False):
    import mne
    EVK.mkdir(parents=True, exist_ok=True)
    pre_dir = Path(C.OUT_DIR)
    for ses in S.session:
        out = EVK / f"{ses}_evoked.npz"
        if out.exists() and not overwrite:
            continue
        ep = mne.read_epochs(pre_dir / ses / f"{ses}-epo.fif", preload=True, verbose=False)
        ep.pick("eeg").crop(*T_WIN)
        md = ep.metadata.reset_index(drop=True)
        X = ep.get_data() * 1e6
        correct = md["correct"].astype(str).str.lower().isin(["true", "1", "1.0"]).values
        d = {}
        for ver, sel in (("correct", correct), ("all", np.ones(len(md), bool))):
            for code, (a, i) in C.CUE_CODES.items():
                m = sel & (md["cue"].values == code)
                d[f"{ver}__{a}_{i}"] = (X[m].mean(0) if m.any() else np.full(X.shape[1:], np.nan)).astype(np.float32)
                d[f"n__{ver}__{a}_{i}"] = np.int32(m.sum())
        np.savez_compressed(out, times=ep.times, ch_names=np.array(ep.ch_names), **d)
        print(f"  exported {ses}: {len(md)} epochs ({correct.sum()} correct)")


# ================================================================= step 2: load
def load(S):
    data, ns = {}, []
    for ses in S.session:
        z = np.load(EVK / f"{ses}_evoked.npz", allow_pickle=False)
        times, ch = z["times"], list(z["ch_names"])
        data[ses] = {ver: np.stack([z[f"{ver}__{c}"] for c in CONDS]) for ver in VERSIONS}  # cond x ch x t
        ns.append({"session": ses, **{f"n_{ver}_{c}": int(z[f"n__{ver}__{c}"]) for ver in VERSIONS for c in CONDS}})
    return data, times, ch, pd.DataFrame(ns)


def cluster_ts(E, ch):
    """cond x ch x t -> {cluster: cond x t}; left, right, and pooled (all 6 channels)."""
    out = {k: np.nanmean(E[:, [ch.index(c) for c in v], :], axis=1) for k, v in K.CLUSTERS.items()}
    out["pooled"] = np.nanmean(E[:, [ch.index(c) for v in K.CLUSTERS.values() for c in v], :], axis=1)
    return out


def syllable_onsets():
    """(name, onset, conditions in which that syllable is measured)."""
    o = [(f"target{k}", t, ["focal_none", "broad_none"]) for k, t in enumerate(K.TIMING["target"], 1)]
    for typ in ("early", "late"):
        o += [(f"int_{typ}{k}", t, [f"focal_{typ}", f"broad_{typ}"])
              for k, t in enumerate(K.TIMING["interrupter"][typ], 1)]
    return o


def local(ts, times, t0, pre=PRE):
    """Re-reference a time series to the `pre` s before onset t0."""
    m = (times >= t0 - pre) & (times < t0)
    return ts - np.nanmean(ts[..., m], axis=-1, keepdims=True)


def win_mean(ts, times, a, b):
    w = (times >= a) & (times <= b)
    return np.nanmean(ts[..., w], axis=-1)


def _locked(data, times, ch, ver, t0, conds, rel):
    """Participant-average response to an onset (both clusters, given conditions), local baseline."""
    ci = [CONDS.index(c) for c in conds]
    acc = []
    for D in data.values():
        cl = cluster_ts(D[ver], ch)
        x = local(np.nanmean(cl["pooled"][ci], axis=0), times, t0)
        acc.append(np.interp(t0 + rel, times, x))
    return np.nanmean(acc, 0)


def n1_latencies(data, times, ch, ver):
    """Two latencies (s after onset): the cue N1 and one syllable N1 for every syllable."""
    dt = times[1] - times[0]
    rel = np.arange(-PRE, 0.45, dt)
    w = (rel >= N1_SEARCH[0]) & (rel <= N1_SEARCH[1])
    cue = _locked(data, times, ch, ver, K.TIMING["cue"], CONDS, rel)
    syl = np.mean([_locked(data, times, ch, ver, t0, conds, rel) for _, t0, conds in syllable_onsets()], 0)
    lat = {"cue": float(rel[w][np.argmin(cue[w])]), "syllable": float(rel[w][np.argmin(syl[w])])}
    for k, v in lat.items():
        if min(abs(v - N1_SEARCH[0]), abs(v - N1_SEARCH[1])) < 1.5 * dt:
            print(f"  WARNING: {k} N1 latency {v:.3f} s is at the edge of the search window {N1_SEARCH}")
    return lat


def n1(ts, times, t0, lat):
    return float(np.nanmean(local(ts, times, t0)[(times >= t0 + lat - N1_HALF) & (times <= t0 + lat + N1_HALF)]))


def syllable_items(stream, attend):
    """(onset, condition) pairs pooled into one syllable-locked average.
    target: every target syllable whose window ends before any interrupter starts, from all three
            interrupter conditions (they are identical until the interrupter appears).
    interrupter: the 3 syllables of the early and of the late interrupter."""
    if stream == "target":
        out = []
        for t in K.TIMING["target"]:
            for i in K.INTERRUPTER:
                if i == "none" or t + SYL_WIN[1] <= K.TIMING["interrupter"][i][0]:
                    out.append((t, f"{attend}_{i}"))
        return out
    return [(t, f"{attend}_{typ}") for typ in ("early", "late") for t in K.TIMING["interrupter"][typ]]


def syl_rel(times):
    dt = times[1] - times[0]
    return np.arange(SYL_WIN[0], SYL_WIN[1] + dt / 2, dt)


def syllable_wave(X, times, stream, attend):
    """Collapsed syllable-locked ERP (cond x t cluster array -> rel time), 50-ms pre-onset baseline each."""
    rel = syl_rel(times)
    return np.mean([np.interp(t0 + rel, times, local(X[CONDS.index(c)], times, t0))
                    for t0, c in syllable_items(stream, attend)], 0)


def syllable_n1(w, rel, lat):
    return float(np.nanmean(w[(rel >= lat - N1_HALF) & (rel <= lat + N1_HALF)]))


def subject_measures(data, times, ch, S, lat):
    rows = []
    info = S.set_index("session")
    for ses, D in data.items():
        for ver in VERSIONS:
            cl = cluster_ts(D[ver], ch)
            L = lat[ver]
            r = dict(session=ses, version=ver)
            for k, X in cl.items():
                ci = lambda c: CONDS.index(c)
                # --- N1 to each syllable, its mean over the 3 syllables, and delta
                for name, t0, conds in syllable_onsets():
                    for c in conds:
                        r[f"{k}_{name}_{c.split('_')[0]}"] = n1(X[ci(c)], times, t0, L["syllable"])
                for kind, names in (("target", ["target1", "target2", "target3"]),
                                    ("int_early", ["int_early1", "int_early2", "int_early3"]),
                                    ("int_late", ["int_late1", "int_late2", "int_late3"])):
                    for a in K.ATTEND:
                        r[f"{k}_{kind}_{a}"] = np.mean([r[f"{k}_{n}_{a}"] for n in names])
                    r[f"{k}_{kind}_delta"] = r[f"{k}_{kind}_broad"] - r[f"{k}_{kind}_focal"]
                    for n in names:
                        r[f"{k}_{n}_delta"] = r[f"{k}_{n}_broad"] - r[f"{k}_{n}_focal"]
                # --- cue response and preparatory shift (3 interrupter conditions pooled per cue type)
                for a in K.ATTEND:
                    pooled = np.nanmean(X[[ci(f"{a}_{i}") for i in K.INTERRUPTER]], 0)
                    r[f"{k}_cue_{a}"] = n1(pooled, times, K.TIMING["cue"], L["cue"])
                    r[f"{k}_prep_{a}"] = float(win_mean(pooled, times, *PREP_WIN))
                # --- P3 and late negativity after the first interrupter syllable
                for typ in ("early", "late"):
                    t0 = K.TIMING["interrupter"][typ][0]
                    for a in K.ATTEND:
                        x = X[ci(f"{a}_{typ}")]
                        r[f"{k}_p3_{typ}_{a}"] = float(win_mean(local(x, times, t0, P3_PRE), times,
                                                                t0 + P3_WIN[0], t0 + P3_WIN[1]))
                        r[f"{k}_lateneg_{typ}_{a}"] = float(win_mean(x, times, t0 + LATENEG_WIN[0], t0 + LATENEG_WIN[1]))
                # --- N1 on the collapsed syllable-locked ERPs
                for stream, tag in (("target", "tsyl"), ("interrupter", "isyl")):
                    for a in K.ATTEND:
                        r[f"{k}_{tag}_{a}"] = syllable_n1(syllable_wave(X, times, stream, a), syl_rel(times),
                                                          L["syllable"])
                for m in ["cue", "prep", "tsyl", "isyl"] + [f"{c}_{t}" for c in ("p3", "lateneg") for t in ("early", "late")]:
                    r[f"{k}_{m}_delta"] = r[f"{k}_{m}_broad"] - r[f"{k}_{m}_focal"]
            # N1 lateralization: (right - left) of the collapsed interrupter-syllable delta; negative = larger
            # modulation over the hemisphere contralateral to the interrupter (left visual field)
            r["lat_isyl_delta"] = r["right_isyl_delta"] - r["left_isyl_delta"]
            rows.append(r)
    M = pd.DataFrame(rows)
    return M.merge(info[["subject", "group", "stimulant", "caffeine", "day"]], left_on="session", right_index=True)


# primary measures, one FDR family per component
FAMILIES = {
    "N1": [f"{k}_{m}_delta" for k in SIDES for m in ("target", "int_early", "int_late")],
    "N1_collapsed": [f"{k}_{m}_delta" for k in SIDES for m in ("tsyl", "isyl")],
    "N1_lateralization": ["lat_isyl_delta"],
    "cue": [f"{k}_cue_delta" for k in SIDES],
    "prep": [f"{k}_prep_delta" for k in SIDES],
    "P3": [f"{k}_p3_{t}_delta" for k in SIDES for t in ("early", "late")],
    "lateneg": [f"{k}_lateneg_{t}_delta" for k in SIDES for t in ("early", "late")],
}
SECONDARY = [f"{k}_{m}_{a}" for k in SIDES for m in ("int_early", "int_late", "target", "cue", "prep", "tsyl", "isyl",
                                                      "p3_early", "p3_late", "lateneg_early", "lateneg_late")
             for a in K.ATTEND] + \
            [f"{k}_int_{t}{n}_delta" for k in SIDES for t in ("early", "late") for n in (1, 2, 3)]


def _families(secondary=True):
    out = list(FAMILIES.items())
    return out + [("secondary", SECONDARY)] if secondary else out


# ================================================================= statistics
def group_stats(M, fams=None):
    rows = []
    for ver, Mv in M[M.day == 1].groupby("version"):
        for ga, gb in K.COMPARISONS:
            A = Mv[Mv.group == ga]
            B = Mv[Mv.group.str.startswith("ADHD")] if gb == "ADHD (all)" else Mv[Mv.group == gb]
            X = pd.concat([A, B])
            isb = np.r_[np.zeros(len(A)), np.ones(len(B))]
            caf = (X.caffeine == "yes").astype(float).values
            for fam, meas in (fams if fams is not None else _families()):
                for m in meas:
                    diff, p = K.group_test(X[m].values, isb, caf, n_perm=N_PERM)
                    rows.append(dict(version=ver, family=fam, test="group (day 1)",
                                     comparison=f"{gb} - {ga}", measure=m, n_a=len(A), n_b=len(B),
                                     mean_a=A[m].mean(), mean_b=B[m].mean(), diff_raw=B[m].mean() - A[m].mean(),
                                     diff_adj_caffeine=diff, hedges_g=K.hedges_g(A[m], B[m]), p_perm=p))
    R = pd.DataFrame(rows)
    R["p_fdr"] = R.groupby(["version", "family", "comparison"]).p_perm.transform(K.fdr)
    return R


def condition_effects(M, fams=None):
    """Is there attentional modulation at all (delta != 0) within each group?"""
    rows = []
    for ver, Mv in M[M.day == 1].groupby("version"):
        for grp in ["all"] + K.GROUPS + ["ADHD (all)"]:
            G = Mv if grp == "all" else (Mv[Mv.group.str.startswith("ADHD")] if grp == "ADHD (all)" else Mv[Mv.group == grp])
            for fam, meas in (fams if fams is not None else _families(secondary=False)):
                if fam == "secondary":
                    continue
                for m in meas:
                    mean, p, dz = K.signflip_test(G[m], n_perm=N_PERM)
                    rows.append(dict(version=ver, family=fam, test="delta vs 0 (day 1)",
                                     comparison=grp, measure=m, n_a=len(G), mean_a=mean, dz=dz, p_perm=p))
    R = pd.DataFrame(rows)
    R["p_fdr"] = R.groupby(["version", "family", "comparison"]).p_perm.transform(K.fdr)
    return R


def within_stats(M, P, fams=None):
    rows = []
    for ver, Mv in M.groupby("version"):
        idx = Mv.set_index("session")
        on, off = idx.loc[P["on"]], idx.loc[P["off"]]
        on_second = (P.on_day.values == 2)
        for fam, meas in (fams if fams is not None else _families(secondary=False)):
            if fam == "secondary":
                continue
            for m in meas:
                d = on[m].values - off[m].values
                b0, b1, p = K.within_order_test(d, on_second, n_perm=N_PERM)
                mraw, praw, dz = K.signflip_test(d, n_perm=N_PERM)
                rows.append(dict(version=ver, family=fam, test="within-person (on - off)",
                                 comparison="on - off", measure=m, n_a=len(d), mean_a=off[m].mean(),
                                 mean_b=on[m].mean(), diff_raw=mraw, p_raw=praw, diff_order_adj=b0,
                                 order_effect_day2_minus_day1=2 * b1, dz=dz, p_perm=p))
    R = pd.DataFrame(rows)
    R["p_fdr"] = R.groupby(["version", "family"]).p_perm.transform(K.fdr)
    return R


def cluster_test(A, B, covA, covB, times, tmin=0.0, tmax=3.5, thresh=2.0, n_perm=2000, seed=0):
    """Two-group cluster-mass permutation test on time series (subjects x time), Welch t,
    labels permuted within caffeine strata. Returns list of (t_start, t_end, mass, p)."""
    w = (times >= tmin) & (times <= tmax)
    X = np.r_[A[:, w], B[:, w]]
    lab = np.r_[np.zeros(len(A)), np.ones(len(B))].astype(int)
    cov = np.r_[covA, covB]
    tw = times[w]

    def tstat(lb):
        a, b = X[lb == 0], X[lb == 1]
        return (b.mean(0) - a.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))

    def clusters(t):
        out = []
        for sign in (1, -1):
            above = sign * t > thresh
            i = 0
            while i < len(t):
                if above[i]:
                    j = i
                    while j + 1 < len(t) and above[j + 1]:
                        j += 1
                    out.append((i, j, float(np.abs(t[i:j + 1]).sum())))
                    i = j + 1
                else:
                    i += 1
        return out

    obs = clusters(tstat(lab))
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(cov == v) for v in np.unique(cov)]
    null = np.zeros(n_perm)
    for k in range(n_perm):
        lb = lab.copy()
        for s in strata:
            lb[s] = rng.permutation(lb[s])
        c = clusters(tstat(lb))
        null[k] = max([m for *_, m in c], default=0)
    return [(float(tw[i]), float(tw[j]), m, float((np.sum(null >= m) + 1) / (n_perm + 1))) for i, j, m in obs]


def _clusters(t, thresh):
    out = []
    for sign in (1, -1):
        above = sign * t > thresh
        i = 0
        while i < len(t):
            if above[i]:
                j = i
                while j + 1 < len(t) and above[j + 1]:
                    j += 1
                out.append((i, j, float(np.abs(t[i:j + 1]).sum())))
                i = j + 1
            else:
                i += 1
    return out


def signflip_cluster(D, times, tmin=0.0, tmax=3.5, thresh=2.0, n_perm=2000, seed=0):
    """Within-group cluster-mass test of broad - focal (subjects x time) against 0, by sign flipping.
    Returns list of (t_start, t_end, mass, p)."""
    w = (times >= tmin) & (times <= tmax)
    X = D[:, w]
    X = X[np.all(np.isfinite(X), 1)]
    n = len(X)
    tw = times[w]
    tstat = lambda Y: Y.mean(0) / (Y.std(0, ddof=1) / np.sqrt(n))
    obs = _clusters(tstat(X), thresh)
    rng = np.random.default_rng(seed)
    null = np.array([max([m for *_, m in _clusters(tstat(X * rng.choice([-1, 1], size=(n, 1))), thresh)], default=0)
                     for _ in range(n_perm)])
    return [(float(tw[i]), float(tw[j]), m, float((np.sum(null >= m) + 1) / (n_perm + 1))) for i, j, m in obs]


# ================================================================= figures
def _timeline(ax, typ, first_col):
    """Stimulus timeline strip: when each stream's syllables came on (one row per stream)."""
    rows = [("target (above)", K.TIMING["target"], "#0b0b0b", 2),
            ("distractor (right)", K.TIMING["distractor_common"], "#7d7b74", 1),
            ("interrupter (left)", K.TIMING["interrupter"].get(typ, []), "#0b0b0b", 0)]
    for name, ts, col, y in rows:
        for t in ts:
            ax.plot([t, t], [y - 0.32, y + 0.32], color=col, lw=1.8, solid_capstyle="butt",
                    ls="-" if not name.startswith("interrupter") else (0, (3, 1)))
    for t in K.TIMING["distractor_rhythm"]:          # 3rd distractor: 1.35 or 1.6 s (two rhythms)
        ax.plot([t, t], [1 - 0.32, 1 + 0.32], color="#7d7b74", lw=1.0, ls=(0, (1.5, 1)))
    ax.plot([K.TIMING["cue"]] * 2, [-0.5, 2.5], color="#52514e", lw=0.8, ls="--")
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows] if first_col else [], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(*T_WIN)


def _onset_lines(ax, typ):
    """Faint vertical lines at every syllable onset, styled like the timeline rows."""
    for t in K.TIMING["target"]:
        ax.axvline(t, color="#0b0b0b", lw=0.7, alpha=0.35, zorder=0)
    for t in K.TIMING["distractor_common"]:
        ax.axvline(t, color="#7d7b74", lw=0.7, alpha=0.35, ls=(0, (1.5, 1.5)), zorder=0)
    for t in K.TIMING["interrupter"].get(typ, []):
        ax.axvline(t, color="#0b0b0b", lw=0.9, alpha=0.5, ls=(0, (5, 2)), zorder=0)


def _cue_line(ax):
    ax.axvline(K.TIMING["cue"], color="#52514e", lw=0.8, ls="--")
    ax.text(K.TIMING["cue"], 0.98, " cue", transform=ax.get_xaxis_transform(), fontsize=6.5, color="#52514e", va="top")


def _layout(fig_h=4.0, nrows=1):
    fig = plt.figure(figsize=(11.5, fig_h), layout="constrained")
    gs = fig.add_gridspec(nrows + 1, 3, height_ratios=[1] * nrows + [0.32])
    axes = np.empty((nrows, 3), object)
    tl = []
    for c in range(3):
        for r in range(nrows):
            axes[r, c] = fig.add_subplot(gs[r, c], sharex=axes[0, 0] if (r or c) else None,
                                         sharey=axes[r, 0] if c else None)
            plt.setp(axes[r, c].get_xticklabels(), visible=False)
        tl.append(fig.add_subplot(gs[nrows, c], sharex=axes[0, 0]))
    return fig, axes, tl


def _smooth(x, times):
    """Zero-phase low-pass for display only."""
    b, a = butter(4, DISPLAY_LP / (0.5 / (times[1] - times[0])))
    return filtfilt(b, a, x)


def _line(ax, t, X, color, label, lw=1.6, sem=True):
    """Mean ± SEM across participants; each participant low-passed for display first."""
    Xs = np.array([_smooth(x, t) for x in np.atleast_2d(X)])
    m = np.nanmean(Xs, 0)
    if sem and len(Xs) > 1:
        se = np.nanstd(Xs, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Xs[:, 0])))
        ax.fill_between(t, m - se, m + se, color=color, alpha=0.18, lw=0)
    ax.plot(t, m, color=color, lw=lw, label=label)


def group_members(S1, g):
    return S1[S1.group.str.startswith("ADHD")] if g == "ADHD (all)" else S1[S1.group == g]


def cl_stack(data, ch, sessions, ver):
    """sessions -> {cluster: subj x cond x t}"""
    out = {k: [] for k in list(K.CLUSTERS) + ["pooled"]}
    for s in sessions:
        cl = cluster_ts(data[s][ver], ch)
        for k in out:
            out[k].append(cl[k])
    return {k: np.array(v) for k, v in out.items()}


def _vlabel(ver):
    return "correct trials" if ver == "correct" else "all clean trials"


def fig_grand(data, times, ch, S1, g, ver, wclus=None, R=None):
    """wclus: {interrupter type: [(t0, t1, mass, p)]} within-group broad vs focal cluster tests.
    R: stats table, for the broad vs focal P3 test of this group."""
    G = group_members(S1, g)
    X = cl_stack(data, ch, G.session, ver)
    fig, axes, tl = _layout()
    for r, k in enumerate(SIDES):
        for c, typ in enumerate(K.INTERRUPTER):
            ax = axes[r, c]
            _line(ax, times, X[k][:, CONDS.index(f"focal_{typ}")], FOCAL_COLOR, "focal (ignore left)")
            _line(ax, times, X[k][:, CONDS.index(f"broad_{typ}")], BROAD_COLOR, "broad (prepare to switch)")
            ax.axhline(0, color="#a3a29c", lw=0.6)
            tr = ax.get_xaxis_transform()
            for t0, t1, _, pv in (wclus or {}).get(typ, []):
                if pv < 0.05:
                    ax.plot([t0, t1], [0.965, 0.965], transform=tr, color="#0b0b0b", lw=3, solid_capstyle="butt")
            if typ != "none":                         # P3 window after the first interrupter syllable
                i0 = K.TIMING["interrupter"][typ][0]
                ax.axvspan(i0 + P3_WIN[0], i0 + P3_WIN[1], color="#e9e8e3", lw=0, zorder=0)
                if R is not None:
                    q = R[(R.version == ver) & (R.test == "delta vs 0 (day 1)") & (R.comparison == g) &
                          (R.measure == f"pooled_p3_{typ}_delta")].p_perm
                    st = _stars(float(q.iloc[0])) if len(q) else ""
                    ax.text(i0 + np.mean(P3_WIN), 0.03, f"P3\n{st or 'n.s.'}", transform=tr, ha="center",
                            va="bottom", fontsize=7.5)
            _cue_line(ax)
            _onset_lines(ax, typ)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title({"none": "No interrupter", "early": "Early interrupter", "late": "Late interrupter"}[typ])
            if c == 0:
                ax.set_ylabel(f"{POOLED_LABEL}\nµV", fontsize=8)
            ax.set_xlim(*T_WIN)
    for c, typ in enumerate(K.INTERRUPTER):
        _timeline(tl[c], typ, c == 0)
        tl[c].set_xlabel("Time from syllable-stream onset (s)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle(f"Grand-average ERP — {g} (n={len(G)}), day 1 · {_vlabel(ver)} · mean ± SEM, low-pass "
                 f"{DISPLAY_LP:.0f} Hz for display\nBlack bars = broad ≠ focal (within-group cluster permutation, p < .05); "
                 f"grey band = P3 window ({P3_WIN[0] * 1000:.0f}–{P3_WIN[1] * 1000:.0f} ms after the first interrupter "
                 f"syllable), stars = broad vs focal P3 (sign-flip permutation)", x=0.01, ha="left", fontsize=9)
    return fig


def fig_diffwaves(data, times, ch, S1, groups, ver, clus):
    fig, axes, tl = _layout(4.3)
    comps = list(next(iter(clus.values())).keys()) if clus else []
    for r, k in enumerate(SIDES):
        for c, typ in enumerate(K.INTERRUPTER):
            ax = axes[r, c]
            for g in groups:
                G = group_members(S1, g)
                X = cl_stack(data, ch, G.session, ver)[k]
                dW = X[:, CONDS.index(f"broad_{typ}")] - X[:, CONDS.index(f"focal_{typ}")]
                _line(ax, times, dW, K.COLORS[g], f"{g} (n={len(G)})")
            ax.axhline(0, color="#a3a29c", lw=0.6)
            _cue_line(ax)
            _onset_lines(ax, typ)
            ax.grid(axis="x", visible=False)
            # significant clusters (p < .05): one row of bars per comparison, labelled at the right edge
            tr = ax.get_xaxis_transform()
            for j, comp in enumerate(comps):
                y = 0.97 - 0.055 * j
                if c == 2:
                    ax.text(1.01, y, comp, transform=ax.transAxes, fontsize=6.5, va="center", color="#52514e")
                for t0, t1, _, pv in clus.get((k, typ), {}).get(comp, []):
                    if pv < 0.05:
                        ax.plot([t0, t1], [y, y], transform=tr, color="#0b0b0b", lw=3, solid_capstyle="butt")
            if r == 0:
                ax.set_title({"none": "No interrupter", "early": "Early interrupter", "late": "Late interrupter"}[typ])
            if c == 0:
                ax.set_ylabel(f"{POOLED_LABEL}\nbroad − focal (µV)", fontsize=8)
            ax.set_xlim(*T_WIN)
        lo, hi = axes[r, 0].get_ylim()                 # shared y per row: pad once for the cluster bars
        axes[r, 0].set_ylim(lo, hi + (hi - lo) * 0.09 * (len(comps) + 1))
    for c, typ in enumerate(K.INTERRUPTER):
        _timeline(tl[c], typ, c == 0)
        tl[c].set_xlabel("Time from syllable-stream onset (s)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=len(groups), fontsize=8)
    fig.suptitle(f"Attentional modulation (broad − focal), day 1 · {_vlabel(ver)} · mean ± SEM, low-pass "
                 f"{DISPLAY_LP:.0f} Hz for display; black bars = cluster-permutation p < .05 (unsmoothed data)",
                 x=0.01, ha="left", fontsize=9)
    return fig


def fig_syllables(data, times, ch, S1, ver, R, groups=None):
    """Collapsed syllable-locked ERPs. Rows = stream (target, interrupter); columns = focal, broad, broad - focal,
    with the groups overlaid in each panel (mean ± SEM)."""
    groups = groups or MAIN_GROUPS
    rel = syl_rel(times)
    L = R.attrs["lat_syllable"][ver]
    Rv = R[R.version == ver]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), layout="constrained", sharex=True)
    for r, st in enumerate(STREAMS):
        tag = "tsyl" if st == "target" else "isyl"
        W = {g: {a: np.array([syllable_wave(cluster_ts(data[s][ver], ch)["pooled"], times, st, a)
                              for s in group_members(S1, g).session]) for a in K.ATTEND} for g in groups}
        for c, a in enumerate(("focal", "broad", "delta")):
            ax = axes[r, c]
            for g in groups:
                X = W[g]["broad"] - W[g]["focal"] if a == "delta" else W[g][a]
                _line(ax, rel, X, K.COLORS[g], f"{g} (n={len(X)})")
            ax.axhline(0, color="#a3a29c", lw=0.6)
            ax.axvline(0, color="#0b0b0b", lw=0.8)
            ax.axvspan(L - N1_HALF, L + N1_HALF, color="#e9e8e3", lw=0, zorder=0)
            if st == "target":
                ax.axvline(0.1, color="#7d7b74", lw=0.7, ls=(0, (1.5, 1.5)), zorder=0)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title({"focal": "Focal (ignore left)", "broad": "Broad (prepare to switch)",
                              "delta": "Broad − focal"}[a], fontsize=9.5)
            if a == "delta":                       # N1 stats: broad vs focal within each group, and group difference
                m = f"pooled_{tag}_delta"
                lines = []
                for g in groups:
                    q = Rv[(Rv.test == "delta vs 0 (day 1)") & (Rv.comparison == g) & (Rv.measure == m)].p_perm
                    lines.append(f"{g}: {_stars(float(q.iloc[0])) or 'n.s.'}" if len(q) else "")
                cmps = ["ADHD (all) - NT"] if groups == MAIN_GROUPS else \
                       ["ADHD-off - NT", "ADHD-on - NT", "ADHD-on - ADHD-off"]
                for cmp in cmps:
                    q = Rv[(Rv.test == "group (day 1)") & (Rv.comparison == cmp) & (Rv.measure == m)].p_perm
                    if len(q):
                        lines.append(f"{cmp.replace(' - ', ' vs ')}: {_stars(float(q.iloc[0])) or 'n.s.'}")
                ax.text(1.03, 0.5, "N1 (grey band),\nbroad vs focal\n\n" + "\n".join(lines), transform=ax.transAxes,
                        ha="left", va="center", fontsize=7, color="#3b3a36")
        lo = min(axes[r, 0].get_ylim()[0], axes[r, 1].get_ylim()[0])      # focal and broad share one y range
        hi = max(axes[r, 0].get_ylim()[1], axes[r, 1].get_ylim()[1])
        for ax in axes[r, :2]:
            ax.set_ylim(lo, hi)
        axes[r, 0].set_ylabel(f"{STREAMS[st]}\n{POOLED_LABEL} (µV)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time from syllable onset (s)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=len(groups), fontsize=8)
    fig.suptitle(f"Collapsed syllable-locked ERPs, day 1 · {_vlabel(ver)} · mean ± SEM, low-pass {DISPLAY_LP:.0f} Hz for "
                 f"display\nTarget syllables: {len(syllable_items('target', 'focal'))} onset × condition averages (pooled "
                 f"across interrupter conditions until the interrupter starts); interrupter syllables: "
                 f"{len(syllable_items('interrupter', 'focal'))} (early + late).\nGrey band = N1 window ({L * 1000:.0f} ± "
                 f"{N1_HALF * 1000:.0f} ms); dotted = distractor syllable (+0.1 s). Stars: within group = sign-flip "
                 f"permutation; between groups = caffeine-adjusted permutation on broad − focal (uncorrected).",
                 x=0.01, ha="left", fontsize=8)
    return fig


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _box(ax, data_by_group, colors, rng):
    pos = np.arange(len(data_by_group))
    vals = [np.asarray(v, float)[~np.isnan(v)] for v in data_by_group]
    bp = ax.boxplot(vals, positions=pos, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#0b0b0b", lw=1.4), whiskerprops=dict(color="#52514e"),
                    capprops=dict(color="#52514e"), boxprops=dict(lw=0.8, color="#52514e"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    for x, v, c in zip(pos, vals, colors):
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color=c, edgecolor="white", lw=0.4, zorder=3)


def _sig_bars(ax, pairs):
    """pairs: list of (x1, x2, p). Draw a bracket with stars for p < .05 only."""
    sig = [(a, b, p) for a, b, p in pairs if p < 0.05]
    if not sig:
        return
    lo, hi = ax.get_ylim()
    h = (hi - lo)
    for j, (a, b, p) in enumerate(sorted(sig, key=lambda s: abs(s[1] - s[0]))):
        y = hi + h * (0.04 + 0.09 * j)
        ax.plot([a, a, b, b], [y - h * 0.02, y, y, y - h * 0.02], color="#0b0b0b", lw=0.8)
        ax.text((a + b) / 2, y, _stars(p), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(lo, hi + h * (0.12 + 0.09 * len(sig)))


N1_PANELS = [("target", "N1 · targets, no interrupter\n(preparing to switch)"), ("int_early", "N1 · early interrupter"),
             ("int_late", "N1 · late interrupter"), ("cue", "Cue response\n(broad vs focal cue picture)")]
SYL_PANELS = [("tsyl", "N1 · target syllables\n(collapsed, all trials)"),
              ("isyl", "N1 · interrupter syllables\n(collapsed, early + late)")]
LAT_PANELS = [("lat_isyl", "N1 lateralization\nright − left cluster\n(interrupter syllables)")]
COMP_PANELS = [("prep", "Preparatory shift\n0–1.0 s"), ("p3_early", "P3 · early interrupter"),
               ("p3_late", "P3 · late interrupter"), ("lateneg_early", "Late negativity\nearly interrupter"),
               ("lateneg_late", "Late negativity\nlate interrupter")]
def _col(k, m):
    """Column name of a panel's measure (lateralization panels are not per cluster)."""
    return f"{m}_delta" if m.startswith("lat_") else f"{k}_{m}_delta"


STATS_NAME = "erp_stats.csv"   # figure footers (itc_analysis.py reuses these figure functions)


def _foot():
    return f"* p < .05, ** p < .01, *** p < .001 (caffeine-adjusted permutation, uncorrected; full stats in {STATS_NAME})"


def fig_box(M, R, ver, pooled, panels, title, unit="µV"):
    rng = np.random.default_rng(3)
    Mv = M[(M.version == ver) & (M.day == 1)].copy()
    Rv = R[(R.version == ver) & (R.test == "group (day 1)")]
    if pooled:
        Mv["pg"] = np.where(Mv.group == "NT", "NT", "ADHD (all)")
        groups = ["NT", "ADHD (all)"]
        comps = {"ADHD (all) - NT": (0, 1)}
    else:
        Mv["pg"] = Mv.group
        groups = K.GROUPS
        comps = {"ADHD-off - NT": (0, 1), "ADHD-on - ADHD-off": (1, 2), "ADHD-on - NT": (0, 2)}
    fig, axes = plt.subplots(len(SIDES), len(panels), figsize=(2.6 * len(panels) + 0.4, 3.7), layout="constrained",
                             squeeze=False)
    for r, k in enumerate(SIDES):
        for c, (m, ttl) in enumerate(panels):
            ax = axes[r, c]
            col = _col(k, m)
            _box(ax, [Mv.loc[Mv.pg == g, col].values for g in groups], [K.COLORS[g] for g in groups], rng)
            ax.axhline(0, color="#52514e", lw=0.8, zorder=0)
            ps = Rv[Rv.measure == col].set_index("comparison").p_perm
            _sig_bars(ax, [(a, b, ps[cmp]) for cmp, (a, b) in comps.items()])
            ax.set_title(ttl, fontsize=8.5)
            ax.set_xticks(range(len(groups)),
                          [g.replace("ADHD-", "ADHD\n").replace(" (all)", "\n(all)") for g in groups], fontsize=7.5)
            ax.grid(axis="x", visible=False)
            if c == 0:
                ax.set_ylabel(f"{POOLED_LABEL}\nbroad − focal ({unit})", fontsize=8)
    fig.suptitle(f"{title}, day 1 · {_vlabel(ver)}\n{_foot()}", x=0.01, ha="left", fontsize=8.5)
    return fig


def fig_p3(M, R, ver, groups):
    """Interrupter P3, focal vs broad (see fig_paired)."""
    return fig_paired(M, R, ver, groups, [("pooled_p3_early", "Early interrupter"), ("pooled_p3_late", "Late interrupter")],
                      f"P3, {POOLED_LABEL} (µV)\n{P3_WIN[0] * 1000:.0f}–{P3_WIN[1] * 1000:.0f} ms after 1st interrupter "
                      f"syllable", "Interrupter P3, focal vs broad")


def fig_paired(M, R, ver, groups, panels, ylabel, title):
    """Focal vs broad: paired values per participant, broad vs focal within each group (sign-flip permutation on
    broad - focal) and group differences in broad - focal (= group x condition). panels: [(column prefix, title)];
    columns <prefix>_focal / _broad / _delta."""
    Mv = M[(M.version == ver) & (M.day == 1)].copy()
    Mv["pg"] = np.where(Mv.group == "NT", "NT", "ADHD (all)") if groups == MAIN_GROUPS else Mv.group
    Rv = R[R.version == ver]
    comps = {"ADHD (all) - NT": (0, 1)} if groups == MAIN_GROUPS else \
            {"ADHD-off - NT": (0, 1), "ADHD-on - ADHD-off": (1, 2), "ADHD-on - NT": (0, 2)}
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(groups) * len(panels) + 1.2, 3.9), layout="constrained",
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (pre, ttl) in zip(axes, panels):
        f_col, b_col, d_col = (f"{pre}_{a}" for a in ("focal", "broad", "delta"))
        for i, g in enumerate(groups):
            G = Mv[Mv.pg == g]
            x0, x1 = i - 0.17, i + 0.17
            j = rng.uniform(-0.04, 0.04, len(G))
            for yf, yb, jj in zip(G[f_col], G[b_col], j):
                ax.plot([x0 + jj, x1 + jj], [yf, yb], color="#c9c8c2", lw=0.8, zorder=1)
            ax.scatter(x0 + j, G[f_col], s=14, color=FOCAL_COLOR, edgecolor="white", lw=0.4, zorder=2)
            ax.scatter(x1 + j, G[b_col], s=14, color=BROAD_COLOR, edgecolor="white", lw=0.4, zorder=2)
            for x, c, col in ((x0, f_col, FOCAL_COLOR), (x1, b_col, BROAD_COLOR)):
                m, se = G[c].mean(), G[c].std(ddof=1) / np.sqrt(len(G))
                ax.errorbar(x - 0.09 if c == f_col else x + 0.09, m, yerr=se, fmt="o", ms=6, color=col, mec="#0b0b0b",
                            mew=0.6, elinewidth=1.6, capsize=0, zorder=3)
            q = Rv[(Rv.test == "delta vs 0 (day 1)") & (Rv.comparison == g) & (Rv.measure == d_col)]
            ax.text(i, 1.0, (_stars(float(q.p_perm.iloc[0])) or "n.s.") if len(q) else "", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(groups)), [f"{g}\n(n={int((Mv.pg == g).sum())})" for g in groups], fontsize=8)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.axhline(0, color="#a3a29c", lw=0.6, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.set_title(ttl, fontsize=9, pad=16)
    # group x condition brackets (drawn after both panels so the shared y-range includes every point)
    lo, hi = axes[0].get_ylim()
    h = hi - lo
    for ax, (pre, _) in zip(axes, panels):
        ps = Rv[(Rv.test == "group (day 1)") & (Rv.measure == f"{pre}_delta")].set_index("comparison").p_perm
        for j, (cmp, (a, b)) in enumerate(sorted(comps.items(), key=lambda kv: abs(kv[1][1] - kv[1][0]))):
            y = hi + h * (0.04 + 0.09 * j)
            ax.plot([a, a, b, b], [y - h * 0.02, y, y, y - h * 0.02], color="#0b0b0b", lw=0.8)
            ax.text((a + b) / 2, y, _stars(ps[cmp]) or "n.s.", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_ylim(lo, hi + h * (0.12 + 0.09 * len(comps)))
    axes[0].set_ylabel(ylabel, fontsize=8)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=FOCAL_COLOR, label="focal (ignore left)"),
                        Line2D([], [], marker="o", ls="", color=BROAD_COLOR, label="broad (prepare to switch)")],
               loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle(f"{title}, day 1 · {_vlabel(ver)} · mean ± SEM\nStars above each group = broad vs "
                 f"focal (sign-flip permutation, shown above the plot); brackets = group difference in broad − focal, i.e. group × condition "
                 f"(caffeine-adjusted permutation); uncorrected, full stats in {STATS_NAME}", x=0.01, ha="left", fontsize=8)
    return fig


def fig_within(M, P, R, ver, panels, title, unit="µV"):
    Mv = M[M.version == ver].set_index("session")
    Rw = R[(R.version == ver) & (R.test == "within-person (on - off)")].set_index("measure")
    fig, axes = plt.subplots(len(SIDES), len(panels), figsize=(2.2 * len(panels) + 1.6, 3.6), layout="constrained",
                             squeeze=False)
    for r, k in enumerate(SIDES):
        for c, (m, ttl) in enumerate(panels):
            ax = axes[r, c]
            col = _col(k, m)
            for _, p in P.iterrows():
                y0, y1 = Mv.loc[p["off"], col], Mv.loc[p["on"], col]
                ax.plot([0, 1], [y0, y1], color="#a3a29c", lw=1, ls="-" if p.on_day == 1 else "--", zorder=1)
                ax.scatter([0], [y0], s=22, color=K.COLORS["ADHD-off"], edgecolor="white", lw=0.6, zorder=2)
                ax.scatter([1], [y1], s=22, color=K.COLORS["ADHD-on"], edgecolor="white", lw=0.6, zorder=2)
            ax.axhline(0, color="#52514e", lw=0.8)
            ax.set_title(f"{ttl} {_stars(Rw.loc[col].p_perm)}".strip() if r == 0 else _stars(Rw.loc[col].p_perm),
                         fontsize=8.5)
            ax.set_xticks([0, 1], ["off", "on"])
            ax.set_xlim(-0.4, 1.4)
            if c == 0:
                ax.set_ylabel(f"{POOLED_LABEL}\nbroad − focal ({unit})", fontsize=8)
    fig.legend(handles=[Line2D([], [], color="#a3a29c", ls="-", label="on-stimulant visit first"),
                        Line2D([], [], color="#a3a29c", ls="--", label="off-stimulant visit first")],
               loc="outside right center", fontsize=7.5)
    fig.suptitle(f"Within-person stimulant effect on {title} (n={len(P)} ADHD) · {_vlabel(ver)}\n"
                 f"Stars = on vs off, visit-order adjusted: * p < .05, ** p < .01, *** p < .001 "
                 f"(full stats in {STATS_NAME})", x=0.01, ha="left", fontsize=8.5)
    return fig


# ---- topographies (drawn without MNE so the figure step runs anywhere)
def _locs_xy(ch):
    rows = [l.split() for l in Path(C.LOCS_FILE).read_text().splitlines() if l.strip()]
    pol = {r[3]: (float(r[1]), float(r[2])) for r in rows}
    th = np.deg2rad([pol[c][0] for c in ch])
    rad = np.array([pol[c][1] for c in ch])
    return np.c_[rad * np.sin(th), rad * np.cos(th)]          # nose up (+y), right ear +x


def _topo(ax, vals, xy, vlim, cmap="RdBu_r"):
    from scipy.interpolate import griddata
    gx, gy = np.meshgrid(np.linspace(-0.55, 0.55, 120), np.linspace(-0.55, 0.55, 120))
    z = griddata(xy, vals, (gx, gy), method="cubic")
    z[np.hypot(gx, gy) > 0.52] = np.nan
    im = ax.contourf(gx, gy, z, levels=np.linspace(-vlim, vlim, 13), cmap=cmap, extend="both")
    ax.contour(gx, gy, z, levels=np.linspace(-vlim, vlim, 7), colors="#52514e", linewidths=0.3)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.52 * np.cos(th), 0.52 * np.sin(th), color="#0b0b0b", lw=0.8)
    ax.plot([-0.05, 0, 0.05], [0.515, 0.58, 0.515], color="#0b0b0b", lw=0.8)
    ax.scatter(xy[:, 0], xy[:, 1], s=1.5, color="#0b0b0b", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    return im


def _topo_amp(E, times, comp, typ, cond, lat):
    """Channel amplitudes (ch,) for one condition at one component after the first interrupter syllable."""
    t0 = K.TIMING["interrupter"][typ][0]
    x = E[CONDS.index(cond)]
    if comp == "N1":
        L = t0 + lat["syllable"]
        return win_mean(local(x, times, t0), times, L - N1_HALF, L + N1_HALF)
    if comp == "P3":
        return win_mean(local(x, times, t0, P3_PRE), times, t0 + P3_WIN[0], t0 + P3_WIN[1])
    return win_mean(x, times, t0 + LATENEG_WIN[0], t0 + LATENEG_WIN[1])


TOPO_DESC = {
    "N1": lambda lat: f"N1 of the first interrupter syllable ({lat['syllable'] * 1000:.0f} ms after onset, "
                      f"re 50 ms before it)",
    "P3": lambda lat: f"P3 after the first interrupter syllable ({P3_WIN[0] * 1000:.0f}–{P3_WIN[1] * 1000:.0f} ms, "
                      f"re 100 ms before it)",
    "lateneg": lambda lat: f"late negativity after the first interrupter syllable ({LATENEG_WIN[0]:.1f}–"
                           f"{LATENEG_WIN[1]:.1f} s, re pre-cue baseline)",
}


def fig_topo(data, times, ch, S1, groups, ver, lat, comp):
    xy = _locs_xy(ch)
    cl_idx = [ch.index(c) for c in K.CLUSTERS["left"] + K.CLUSTERS["right"]]
    cols = [("early", "focal"), ("early", "broad"), ("early", "delta"), ("late", "focal"), ("late", "broad"), ("late", "delta")]
    vals = {}
    for g in groups:
        G = group_members(S1, g)
        for typ, a in cols:
            per = []
            for s in G.session:
                E = data[s][ver]
                amp = lambda c: _topo_amp(E, times, comp, typ, c, lat)
                per.append(amp(f"broad_{typ}") - amp(f"focal_{typ}") if a == "delta" else amp(f"{a}_{typ}"))
            vals[(g, typ, a)] = np.nanmean(per, 0)
    vabs = np.nanpercentile(np.abs([v for (g, t, a), v in vals.items() if a != "delta"]), 98)
    vdel = np.nanpercentile(np.abs([v for (g, t, a), v in vals.items() if a == "delta"]), 98)
    fig, axes = plt.subplots(len(groups), 6, figsize=(10.5, 1.9 * len(groups) + 1.2))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.12, wspace=0.05, hspace=0.12)
    axes = np.atleast_2d(axes)
    for r, g in enumerate(groups):
        for c, (typ, a) in enumerate(cols):
            im = _topo(axes[r, c], vals[(g, typ, a)], xy, vdel if a == "delta" else vabs)
            axes[r, c].scatter(xy[cl_idx, 0], xy[cl_idx, 1], s=10, facecolor="none", edgecolor="#0b0b0b", lw=0.6, zorder=4)
            if r == 0:
                axes[r, c].set_title(f"{typ} · {'broad − focal' if a == 'delta' else a}", fontsize=8)
            if c == 0:
                axes[r, c].text(-0.75, 0, g, rotation=90, va="center", ha="center", fontsize=8.5, color=K.COLORS[g],
                                fontweight="bold")
            if r == len(groups) - 1 and c in (0, 2):
                v = vdel if c == 2 else vabs
                b0, b1 = (axes[r, 0].get_position(), axes[r, 1].get_position()) if c == 0 else \
                         (axes[r, 2].get_position(), axes[r, 2].get_position())
                cax = fig.add_axes([b0.x0 + 0.01, 0.06, b1.x1 - b0.x0 - 0.02, 0.015])
                cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-v, 0, v], format="%.1f")
                cb.set_label(("focal & broad maps" if c == 0 else "broad − focal maps") + " (µV)", fontsize=7)
                cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Scalp maps: {TOPO_DESC[comp](lat)} · {_vlabel(ver)}; circles = PO clusters; "
                 f"late columns share the early colour scales", x=0.01, ha="left", fontsize=8.5)
    return fig


# ================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--overwrite-export", action="store_true")
    a = ap.parse_args()
    K.style()
    plt.rcParams["figure.max_open_warning"] = 0
    OUT.mkdir(parents=True, exist_ok=True)
    S = K.sessions()
    if not a.skip_export:
        export(S, overwrite=a.overwrite_export)
    data, times, ch, Ntr = load(S)
    lat = {ver: n1_latencies(data, times, ch, ver) for ver in VERSIONS}
    print("N1 latencies (ms after onset):", json.dumps({v: {k: round(x * 1000) for k, x in L.items()}
                                                        for v, L in lat.items()}))
    M = subject_measures(data, times, ch, S, lat)
    P = K.within_pairs(S)
    R = pd.concat([group_stats(M), condition_effects(M), within_stats(M, P)], ignore_index=True)
    R.attrs["lat_syllable"] = {v: lat[v]["syllable"] for v in VERSIONS}

    S1 = S[S.day == 1]
    clus_rows, clus = [], {}
    for ver in VERSIONS:
        clus[ver] = {}
        for k in SIDES:
            for typ in K.INTERRUPTER:
                res = {}
                for ga, gb in K.COMPARISONS:
                    A, B = group_members(S1, ga), group_members(S1, gb)
                    XA, XB = cl_stack(data, ch, A.session, ver)[k], cl_stack(data, ch, B.session, ver)[k]
                    dA = XA[:, CONDS.index(f"broad_{typ}")] - XA[:, CONDS.index(f"focal_{typ}")]
                    dB = XB[:, CONDS.index(f"broad_{typ}")] - XB[:, CONDS.index(f"focal_{typ}")]
                    out = cluster_test(dA, dB, (A.caffeine == "yes").values, (B.caffeine == "yes").values, times,
                                       n_perm=2000 if ver == MAIN else 1000)
                    short = f"{gb.replace('ADHD-', '').replace('ADHD (all)', 'ADHD')} vs {ga.replace('ADHD-', '')}"
                    res[short] = out
                    for t0, t1, mass, p in out:
                        clus_rows.append(dict(version=ver, cluster=k, interrupter=typ,
                                              comparison=f"{gb} - {ga}", t_start=t0, t_end=t1, mass=mass, p=p))
                clus[ver][(k, typ)] = res

    # within-group broad vs focal time-course tests (shown on the grand averages)
    wclus = {}
    for g in K.GROUPS + ["ADHD (all)"]:
        G = group_members(S1, g)
        X = cl_stack(data, ch, G.session, MAIN)["pooled"]
        wclus[g] = {}
        for typ in K.INTERRUPTER:
            out = signflip_cluster(X[:, CONDS.index(f"broad_{typ}")] - X[:, CONDS.index(f"focal_{typ}")], times)
            wclus[g][typ] = out
            for t0, t1, mass, p in out:
                clus_rows.append(dict(version=MAIN, cluster="pooled", interrupter=typ, test="broad vs focal within group",
                                      comparison=g, t_start=t0, t_end=t1, mass=mass, p=p))

    Ntr.merge(S[["session", "group", "day"]], on="session").to_csv(OUT / "erp_trial_counts.csv", index=False)
    M.to_csv(OUT / "erp_subject_measures.csv", index=False)
    R.to_csv(OUT / "erp_stats.csv", index=False)
    pd.DataFrame(clus_rows).to_csv(OUT / "erp_cluster_tests.csv", index=False)
    pd.DataFrame(lat).to_csv(OUT / "erp_n1_latencies.csv")

    # main figures: NT vs ADHD (all), correct trials. Supplementary: ADHD-off / ADHD-on, within-person, all trials.
    # Old erp_* figures are removed first so files from earlier layouts don't linger.
    for d in (K.FIG_DIR / SUB, K.FIG_DIR / SUB / "supplementary"):
        for f in list(d.glob("erp_*.png")) + list(d.glob("erp_*.pdf")) if d.exists() else []:
            f.unlink()
    written = []
    for ver in VERSIONS:
        figs = []                                   # (name, figure, supplementary?)
        main_v = ver == MAIN
        sfx = "" if main_v else f"__{ver}"
        if main_v:
            for g in K.GROUPS + ["ADHD (all)"]:
                figs.append((f"erp_grand_{g.replace(' (all)', '_pooled')}",
                             fig_grand(data, times, ch, S1, g, ver, wclus[g], R), g not in MAIN_GROUPS))
            for comp in ("N1", "P3", "lateneg"):
                figs.append((f"erp_topomaps_{comp}", fig_topo(data, times, ch, S1, MAIN_GROUPS, ver, lat[ver], comp), False))
                figs.append((f"erp_topomaps_{comp}_4groups",
                             fig_topo(data, times, ch, S1, ["NT", "ADHD (all)", "ADHD-off", "ADHD-on"], ver, lat[ver], comp), True))
        c3 = {kk: {c: v for c, v in d.items() if "ADHD vs" not in c} for kk, d in clus[ver].items()}
        cp = {kk: {c: v for c, v in d.items() if "ADHD vs" in c} for kk, d in clus[ver].items()}
        figs += [("erp_diffwaves", fig_diffwaves(data, times, ch, S1, MAIN_GROUPS, ver, cp), False),
                 ("erp_diffwaves_3groups", fig_diffwaves(data, times, ch, S1, K.GROUPS, ver, c3), True),
                 ("erp_syllables_collapsed", fig_syllables(data, times, ch, S1, ver, R, MAIN_GROUPS), False),
                 ("erp_syllables_collapsed_3groups", fig_syllables(data, times, ch, S1, ver, R, K.GROUPS), True),
                 ("erp_P3_focal_vs_broad", fig_p3(M, R, ver, MAIN_GROUPS), False),
                 ("erp_P3_focal_vs_broad_3groups", fig_p3(M, R, ver, K.GROUPS), True)]
        for pooled, gname in ((True, ""), (False, "_3groups")):
            figs += [(f"erp_deltaN1{gname}", fig_box(M, R, ver, pooled, N1_PANELS,
                                                     "N1 attentional modulation (more negative = larger)"), not pooled),
                     (f"erp_deltaN1_collapsed{gname}", fig_box(M, R, ver, pooled, SYL_PANELS + LAT_PANELS,
                                                               "N1 modulation on collapsed syllable ERPs (more negative = larger)"),
                      not pooled),
                     (f"erp_components{gname}", fig_box(M, R, ver, pooled, COMP_PANELS,
                                                        "Preparation, P3 and late negativity (broad − focal)"), not pooled)]
        figs += [("erp_within_person_N1", fig_within(M, P, R, ver, N1_PANELS[:3] + SYL_PANELS + LAT_PANELS, "N1 modulation"), True),
                 ("erp_within_person_components", fig_within(M, P, R, ver, COMP_PANELS, "preparation, P3 and late negativity"), True)]
        for name, f, supp in figs:
            written.append(K.save(f, name + sfx, f"{SUB}/supplementary" if (supp or not main_v) else SUB))
            plt.close(f)
    print(f"wrote {len(written)} figures to {K.FIG_DIR / SUB} (main) and {K.FIG_DIR / SUB / 'supplementary'}")

    p3 = R[(R.version == MAIN) & (R.family == "P3") & (R.test != "within-person (on - off)") &
           (R.comparison.isin(["NT", "ADHD (all)", "ADHD (all) - NT"]))]
    print("\nInterrupter P3, broad - focal (correct trials, pooled cluster):")
    print(p3[["test", "comparison", "measure", "mean_a", "mean_b", "dz", "hedges_g", "p_perm", "p_fdr"]].round(3).to_string(index=False))
    for g in MAIN_GROUPS:
        for typ in ("early", "late"):
            sig = [f"{t0:.2f}–{t1:.2f} s (p = {p:.3f})" for t0, t1, _, p in wclus[g][typ] if p < 0.05]
            print(f"  {g}, {typ} interrupter, broad ≠ focal clusters: {', '.join(sig) if sig else 'none'}")

    pd.set_option("display.width", 220)
    show = R[(R.version == MAIN) & (R.family != "secondary") & (R.p_perm < 0.05) &
             R.comparison.isin(["ADHD (all) - NT"])]
    print("\nNT vs ADHD (all), correct trials, primary measures with p < .05 (uncorrected; everything else in erp_stats.csv):")
    cols = ["test", "comparison", "family", "measure", "mean_a", "mean_b", "diff_raw", "p_perm", "p_fdr"]
    print(show[cols].round(3).to_string(index=False) if len(show) else "  none")
    ct = Ntr.set_index("session")[[f"n_correct_{c}" for c in CONDS]].min(axis=1)
    print(f"\nCorrect trials per condition: median {int(ct.median())}, min {int(ct.min())} ({ct.idxmin()})")


if __name__ == "__main__":
    main()
