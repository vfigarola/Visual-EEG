"""Behavior: recall accuracy in the visual attention-switching task.

Run from the python/ folder:   python analysis/behavior.py

Accuracy = all 3 syllables reported in the correct order (same definition as results.score).
Uses all visual trials in the RESULTS files (behavior does not depend on EEG quality).
Sessions excluded from post-processing (config.ANALYSIS_EXCLUDE, JKSA200) are left out so every
analysis uses the same sample.

Outputs
  derivatives/analysis/behavior/behavior_trials.csv         one row per trial
  derivatives/analysis/behavior/behavior_by_condition.csv   one row per session x condition
  derivatives/analysis/behavior/behavior_subject_measures.csv
  derivatives/analysis/behavior/behavior_stats.csv           group / within-group / within-person tests
  derivatives/figures/behavior/*.png|pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common as K
import trials as T

OUT = K.ANALYSIS_DIR / "behavior"
N_PERM = 10000


# ---------------------------------------------------------------- data
def load_trials(S):
    rows = []
    for _, s in S.iterrows():
        for half in (1, 2):
            f = s.get(f"beh{half}")
            if not isinstance(f, str):
                continue
            b = T.behavior_trials(f, half)
            b.insert(0, "session", s.session)
            rows.append(b)
    D = pd.concat(rows, ignore_index=True)
    D = D.merge(S[["session", "subject", "group", "stimulant", "caffeine", "day", "n_visits"]], on="session")
    D["syll_acc"] = D.n_syll_correct / 3
    return D


def check_against_score(S, D):
    """Sanity check: our per-block correct counts must equal results.score."""
    bad = []
    for _, s in S.iterrows():
        for half in (1, 2):
            f = s.get(f"beh{half}")
            if not isinstance(f, str):
                continue
            r = T._load_results(f)
            score = np.asarray(r["score"]).ravel()
            got = D[(D.session == s.session) & (D.half == half)].groupby("block").correct.sum().values
            if not np.array_equal(got, score[:len(got)]):
                bad.append((s.session, half, got.tolist(), score[:len(got)].tolist()))
    return bad


def condition_table(D):
    g = D.groupby(["session", "subject", "group", "stimulant", "caffeine", "day", "attend", "interrupter"])
    T_ = g.agg(n_trials=("correct", "size"), pct_correct=("correct", "mean"), syll_acc=("syll_acc", "mean"))
    rt = D[D.correct].groupby(["session", "attend", "interrupter"]).rt.median().rename("rt_correct_median")
    T_ = T_.reset_index().merge(rt.reset_index(), on=["session", "attend", "interrupter"], how="left")
    T_["pct_correct"] *= 100
    T_["syll_acc"] *= 100
    return T_


def subject_measures(D, Tc):
    M = D.groupby(["session", "subject", "group", "stimulant", "caffeine", "day"]).correct.mean().mul(100) \
         .rename("overall").reset_index()
    w = Tc.pivot_table(index="session", columns=["attend", "interrupter"], values="pct_correct")
    for a in K.ATTEND:
        for i in K.INTERRUPTER:
            M[f"{a}_{i}"] = M.session.map(w[(a, i)])
    att = D.groupby(["session", "attend"]).correct.mean().mul(100).unstack()
    M["focal"] = M.session.map(att["focal"])
    M["broad"] = M.session.map(att["broad"])
    M["broad_minus_focal"] = M.broad - M.focal
    for i in ("early", "late"):
        M[f"{i}_minus_none"] = M.session.map(D[D.interrupter == i].groupby("session").correct.mean().mul(100)
                                       - D[D.interrupter == "none"].groupby("session").correct.mean().mul(100))
        for a in K.ATTEND:
            M[f"{i}_minus_none_{a}"] = M[f"{a}_{i}"] - M[f"{a}_none"]
    return M


MEASURES = (["overall", "focal", "broad", "broad_minus_focal", "early_minus_none", "late_minus_none",
             "early_minus_none_focal", "early_minus_none_broad", "late_minus_none_focal", "late_minus_none_broad"]
            + [f"{a}_{i}" for a in K.ATTEND for i in K.INTERRUPTER])


# ---------------------------------------------------------------- statistics
def group_stats(M1):
    rows = []
    for ga, gb in K.COMPARISONS:
        A = M1[M1.group == ga]
        B = M1[M1.group.str.startswith("ADHD")] if gb == "ADHD (all)" else M1[M1.group == gb]
        X = pd.concat([A, B])
        is_b = np.r_[np.zeros(len(A)), np.ones(len(B))]
        caf = (X.caffeine == "yes").astype(float).values
        for m in MEASURES:
            diff, p = K.group_test(X[m].values, is_b, caf, n_perm=N_PERM)
            rows.append(dict(test="group (day 1)", comparison=f"{gb} - {ga}", measure=m,
                             n_a=len(A), n_b=len(B), mean_a=A[m].mean(), mean_b=B[m].mean(),
                             diff_raw=B[m].mean() - A[m].mean(), diff_adj_caffeine=diff,
                             hedges_g=K.hedges_g(A[m], B[m]), p_perm=p))
    R = pd.DataFrame(rows)
    R["p_fdr"] = R.groupby("comparison").p_perm.transform(K.fdr)
    return R


def condition_effects(M1):
    """Within-group effects of attend and interrupter (is there a cost at all?)."""
    rows = []
    for grp in ["all"] + K.GROUPS:
        G = M1 if grp == "all" else M1[M1.group == grp]
        for m in ["broad_minus_focal", "early_minus_none", "late_minus_none",
                  "early_minus_none_focal", "early_minus_none_broad", "late_minus_none_focal", "late_minus_none_broad"]:
            mean, p, dz = K.signflip_test(G[m], n_perm=N_PERM)
            rows.append(dict(test="condition effect (day 1)", comparison=grp, measure=m, n_a=len(G),
                             mean_a=mean, dz=dz, p_perm=p))
    R = pd.DataFrame(rows)
    R["p_fdr"] = R.groupby("comparison").p_perm.transform(K.fdr)
    return R


def within_person(M, P):
    rows = []
    idx = M.set_index("session")
    on, off = idx.loc[P["on"]], idx.loc[P["off"]]
    later = np.where((P.on_day.values == 2)[:, None], on[MEASURES].values, off[MEASURES].values)
    earlier = np.where((P.on_day.values == 2)[:, None], off[MEASURES].values, on[MEASURES].values)
    for k, m in enumerate(MEASURES):
        d = on[m].values - off[m].values
        mean, p, dz = K.signflip_test(d, n_perm=N_PERM)
        rows.append(dict(test="within-person (12 pairs)", comparison="on - off", measure=m, n_a=len(d),
                         mean_a=off[m].mean(), mean_b=on[m].mean(), diff_raw=mean, dz=dz, p_perm=p))
        d2 = later[:, k] - earlier[:, k]
        mean, p, dz = K.signflip_test(d2, n_perm=N_PERM)
        rows.append(dict(test="within-person (12 pairs)", comparison="day 2 - day 1", measure=m, n_a=len(d2),
                         mean_a=np.mean(earlier[:, k]), mean_b=np.mean(later[:, k]), diff_raw=mean, dz=dz, p_perm=p))
    R = pd.DataFrame(rows)
    R["p_fdr"] = R.groupby("comparison").p_perm.transform(K.fdr)
    return R


# ---------------------------------------------------------------- figures
def _dots(ax, x, vals, color, rng, width=0.10):
    vals = np.asarray(vals, float)
    ax.scatter(x + rng.uniform(-width, width, len(vals)), vals, s=14, color=color, alpha=0.45,
               edgecolor="white", linewidth=0.5, zorder=2)
    m, se = np.nanmean(vals), np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals)))
    ax.errorbar(x, m, yerr=se, fmt="o", ms=7, color=color, mec="white", mew=1.2, elinewidth=2,
                capsize=0, zorder=3)
    return m


def fig_conditions(M1, pooled=False, R=None):
    """Accuracy per condition. pooled=True: NT vs all ADHD (day 1), with permutation p per condition."""
    rng = np.random.default_rng(1)
    M1 = M1.copy()
    if pooled:
        M1["plot_group"] = np.where(M1.group == "NT", "NT", "ADHD (all)")
        groups, off = ["NT", "ADHD (all)"], {"NT": -0.15, "ADHD (all)": 0.15}
    else:
        M1["plot_group"] = M1.group
        groups, off = K.GROUPS, {"NT": -0.25, "ADHD-off": 0.0, "ADHD-on": 0.25}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True, layout="constrained")
    for ax, a in zip(axes, K.ATTEND):
        for g in groups:
            G = M1[M1.plot_group == g]
            means = [_dots(ax, xi + off[g], G[f"{a}_{i}"], K.COLORS[g], rng, 0.07)
                     for xi, i in enumerate(K.INTERRUPTER)]
            ax.plot(np.arange(3) + off[g], means, color=K.COLORS[g], lw=2, zorder=1,
                    label=f"{g} (n={len(G)})")
        if pooled and R is not None:
            ps = R[(R.test == "group (day 1)") & (R.comparison == "ADHD (all) - NT")].set_index("measure").p_perm
            for xi, i in enumerate(K.INTERRUPTER):
                ax.text(xi, 31, f"p = {ps[f'{a}_{i}']:.2f}", ha="center", va="bottom", fontsize=7, color="#52514e")
        ax.set_xticks(range(3), ["no interrupter", "early", "late"])
        ax.set_title(f"{a.capitalize()} attention")
    axes[0].set_ylabel("Trials correct (%)")
    axes[0].set_ylim(30, 101)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside right center", fontsize=8)
    if pooled:
        fig.suptitle("Recall accuracy by condition: NT vs all ADHD (day 1)\n"
                     "mean ± SEM, dots = participants; p = NT vs ADHD, caffeine-adjusted permutation (uncorrected)",
                     x=0.01, ha="left", fontsize=9)
    else:
        fig.suptitle("Recall accuracy by condition (day 1; mean ± SEM, dots = participants)", x=0.01, ha="left")
    return fig


def fig_effects(M1, R):
    rng = np.random.default_rng(2)
    meas = [("broad_minus_focal", "Broad − focal\n(all trials)"),
            ("early_minus_none_focal", "Focal: early − none\n(cost of ignoring interrupter)"),
            ("late_minus_none_focal", "Focal: late − none\n(cost of ignoring interrupter)"),
            ("early_minus_none_broad", "Broad: early − none\n(switching vs. prepared-only)"),
            ("late_minus_none_broad", "Broad: late − none\n(switching vs. prepared-only)")]
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.6), layout="constrained")
    for ax, (m, title) in zip(axes, meas):
        for gi, g in enumerate(K.GROUPS):
            _dots(ax, gi, M1.loc[M1.group == g, m], K.COLORS[g], rng)
        ax.axhline(0, color="#52514e", lw=0.8)
        ps = R[(R.test == "group (day 1)") & (R.measure == m)].set_index("comparison").p_perm
        short = {"ADHD-off - NT": "off vs NT", "ADHD-on - NT": "on vs NT", "ADHD-on - ADHD-off": "on vs off"}
        ax.set_title(title, fontsize=9, pad=30)
        ax.text(0.5, 1.01, "\n".join(f"{short[c]}: p = {ps[c]:.2f}" for c in short), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.5, color="#52514e", linespacing=1.1)
        ax.set_xticks(range(3), ["NT", "ADHD\noff", "ADHD\non"], fontsize=8)
    axes[0].set_ylabel("Δ trials correct (percentage points)")
    fig.suptitle("Attention and interrupter effects (day 1; group permutation p, caffeine-adjusted, uncorrected)",
                 x=0.01, ha="left")
    return fig


def fig_within(M, P):
    idx = M.set_index("session")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True, layout="constrained")
    for ax, (col_a, col_b, la, lb, title) in zip(axes, [
            ("off", "on", "off stimulant", "on stimulant", "Stimulant (within person)"),
            ("first", "second", "day 1", "day 2", "Visit order (same people)")]):
        for _, r in P.iterrows():
            first, second = (r["on"], r["off"]) if r.on_day == 1 else (r["off"], r["on"])
            sess = {"off": r["off"], "on": r["on"], "first": first, "second": second}
            ya, yb = idx.loc[sess[col_a], "overall"], idx.loc[sess[col_b], "overall"]
            ax.plot([0, 1], [ya, yb], color="#a3a29c", lw=1, zorder=1)
            ca = K.COLORS[idx.loc[sess[col_a], "group"]]
            cb = K.COLORS[idx.loc[sess[col_b], "group"]]
            ax.scatter([0], [ya], s=22, color=ca, edgecolor="white", lw=0.6, zorder=2)
            ax.scatter([1], [yb], s=22, color=cb, edgecolor="white", lw=0.6, zorder=2)
        ax.set_xticks([0, 1], [la, lb])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(title)
    axes[0].set_ylabel("Trials correct, all conditions (%)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=K.COLORS["ADHD-off"], label="off-stimulant session"),
                            Line2D([], [], marker="o", ls="", color=K.COLORS["ADHD-on"], label="on-stimulant session")],
               loc="outside right center", fontsize=7.5)
    fig.suptitle(f"ADHD participants with both visits (n={len(P)}); grey lines join the same person", x=0.01, ha="left")
    return fig


# ---------------------------------------------------------------- main
def main():
    K.style()
    OUT.mkdir(parents=True, exist_ok=True)
    S = K.sessions()
    D = load_trials(S)
    mismatch = check_against_score(S, D)
    print(f"sessions: {len(S)}; trials: {len(D)}; score check mismatches: {len(mismatch)}")
    for m in mismatch[:5]:
        print("   ", m)
    Tc = condition_table(D)
    M = subject_measures(D, Tc)
    M1 = M[M.day == 1].copy()
    P = K.within_pairs(S)
    print("day-1 groups:", M1.group.value_counts().to_dict(), "| within-person pairs:", len(P))

    R = pd.concat([group_stats(M1), condition_effects(M1), within_person(M, P)], ignore_index=True)
    D.to_csv(OUT / "behavior_trials.csv", index=False)
    Tc.to_csv(OUT / "behavior_by_condition.csv", index=False)
    M.to_csv(OUT / "behavior_subject_measures.csv", index=False)
    R.to_csv(OUT / "behavior_stats.csv", index=False)
    P.to_csv(OUT / "within_person_pairs.csv", index=False)

    figs = {"behavior_conditions": fig_conditions(M1),
            "behavior_conditions_ADHD_pooled": fig_conditions(M1, pooled=True, R=R), "behavior_effects": fig_effects(M1, R),
            "behavior_within_person": fig_within(M, P)}
    for name, f in figs.items():
        print("wrote", K.save(f, name, "behavior"))
        plt.close(f)

    # console summary
    show = R[(R.p_perm < 0.05)].copy()
    pd.set_option("display.width", 200)
    print("\nEffects with p < .05 (uncorrected):")
    print(show[["test", "comparison", "measure", "mean_a", "mean_b", "diff_raw", "p_perm", "p_fdr"]]
          .round(3).to_string(index=False) if len(show) else "  none")


if __name__ == "__main__":
    main()
