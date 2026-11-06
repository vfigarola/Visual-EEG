"""ITC: phase locking to the syllables in the visual attention-switching task.

Run from the python/ folder (MNE environment active):
    python analysis/itc_analysis.py               # export phase locking (MNE) + measures, stats, figures
    python analysis/itc_analysis.py --skip-export # reuse the export (numpy only)

Step 1 (MNE reads the epochs; the rest is numpy): per session, complex Morlet wavelets at 2-20 Hz (1-Hz
        steps; 3 cycles at 2 Hz rising linearly to 15 at 20 Hz, like EEGLAB newtimef [3 0.5]) on every
        clean epoch, then phase locking across trials for each condition and trial set ("correct" = main,
        "all" = check). Saved to derivatives/analysis/itc/tf/<session>_itc.npz at 64 Hz:
          ppc__  PO/O channels x frequency x time   (pairwise phase consistency)
          itc__  PO/O channels x frequency x time   (classic ITC = |mean unit phase vector|, supplementary)
          band__ all 64 channels x time             (PPC averaged over 2-20 Hz, for scalp maps)
Step 2 (numpy): measures, statistics, figures.

Why PPC: classic ITC is biased upward when there are fewer trials (pure noise gives ITC ~ 1/sqrt(N)), and
correct-trial counts range from 6 to ~35 per condition. PPC (Vinck et al., 2010) is the unbiased estimate
of squared phase locking: (|sum of unit vectors|^2 - N) / (N (N - 1)). It is ~0 for random phases whatever
N is, so every trial is used and nothing is equalized or dropped.

Measures (PPC averaged over 2-20 Hz and over the pooled PO/O cluster; broad - focal = "delta"):
  per syllable: mean PPC in a 100-ms window at ONE syllable latency (peak of the response pooled over every
          target and interrupter syllable, both conditions, all participants: blind to every contrast).
          target / int_early / int_late = mean of their 3 syllables (same trials as the ERP N1 measures).
  collapsed: target syllables pooled across all 3 interrupter conditions until the interrupter starts
          (tsyl), interrupter syllables pooled over early + late (isyl).
  lateralization: (right - left cluster) of the isyl delta.
  cue:    PPC after the cue (its own latency); the two cues are different pictures.
Families, statistics and figure layouts are those of erp_analysis.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

import common as K
import config as C
import erp_analysis as E

OUT = K.ANALYSIS_DIR / "itc"
TFD = OUT / "tf"
SUB = "itc"
FREQS = np.arange(2.0, 21.0, 1.0)
CYCLES = np.linspace(3.0, 15.0, len(FREQS))
DECIM = 4                       # 256 Hz -> 64 Hz (phase locking is smooth in time)
PO = [c for v in K.CLUSTERS.values() for c in v]          # left 3 then right 3
CL_IDX = {"left": [0, 1, 2], "right": [3, 4, 5], "pooled": [0, 1, 2, 3, 4, 5]}
PEAK_SEARCH = (0.05, 0.35)      # s after onset
HALF = 0.05                     # 100-ms measurement window
SYL_WIN = (-0.10, 0.45)
CONDS = E.CONDS
VERSIONS, MAIN, SIDES = E.VERSIONS, E.MAIN, E.SIDES
UNIT = "PPC"


# ================================================================= step 1: phase locking
def morlet(f, n_cycles, sfreq):
    sd = n_cycles / (2 * np.pi * f)
    t = np.arange(-5 * sd, 5 * sd + 0.5 / sfreq, 1 / sfreq)
    w = np.exp(2j * np.pi * f * t) * np.exp(-t ** 2 / (2 * sd ** 2))
    return (w / np.abs(w).sum()).astype(np.complex64)


def phase_locking(X, sfreq, masks, po_idx):
    """X: trials x channels x time (float). masks: {name: bool over trials}.
    Returns {name: (ppc_po, itc_po, band_all, n)} with ppc_po/itc_po = PO channels x freq x time(decimated),
    band_all = all channels x time (PPC averaged over frequency)."""
    X = np.asarray(X, np.float32)
    n_ch, n_t = X.shape[1], X[..., ::DECIM].shape[-1]
    out = {g: [np.zeros((len(po_idx), len(FREQS), n_t), np.float32), np.zeros((len(po_idx), len(FREQS), n_t), np.float32),
               np.zeros((n_ch, n_t), np.float64), int(m.sum())] for g, m in masks.items()}
    for fi, (f, cyc) in enumerate(zip(FREQS, CYCLES)):
        W = fftconvolve(X, morlet(f, cyc, sfreq)[None, None, :], mode="same", axes=-1)[..., ::DECIM]
        U = W / np.maximum(np.abs(W), 1e-30)
        for g, m in masks.items():
            n = int(m.sum())
            if n < 2:
                out[g][0][:, fi] = np.nan
                out[g][1][:, fi] = np.nan
                out[g][2][:] = np.nan
                continue
            S = U[m].sum(0)                                   # channels x time
            ppc = (np.abs(S) ** 2 - n) / (n * (n - 1))
            out[g][0][:, fi] = ppc[po_idx]
            out[g][1][:, fi] = np.abs(S[po_idx]) / n
            out[g][2] += ppc / len(FREQS)
    return {g: (a, b, c.astype(np.float32), n) for g, (a, b, c, n) in out.items()}


def export(S, overwrite=False):
    import mne
    TFD.mkdir(parents=True, exist_ok=True)
    for ses in S.session:
        out = TFD / f"{ses}_itc.npz"
        if out.exists() and not overwrite:
            continue
        ep = mne.read_epochs(Path(C.OUT_DIR) / ses / f"{ses}-epo.fif", preload=True, verbose=False)
        ep.pick("eeg")
        md = ep.metadata.reset_index(drop=True)
        correct = md["correct"].astype(str).str.lower().isin(["true", "1", "1.0"]).values
        masks = {}
        for ver, sel in (("correct", correct), ("all", np.ones(len(md), bool))):
            for code, (a, i) in C.CUE_CODES.items():
                masks[f"{ver}__{a}_{i}"] = sel & (md["cue"].values == code)
        res = phase_locking(ep.get_data(), ep.info["sfreq"], masks, [ep.ch_names.index(c) for c in PO])
        times = ep.times[::DECIM]
        keep = (times >= E.T_WIN[0] - 1e-9) & (times <= E.T_WIN[1] + 1e-9)
        d = {}
        for g, (ppc, itc, band, n) in res.items():
            d[f"ppc__{g}"], d[f"itc__{g}"], d[f"band__{g}"] = ppc[..., keep], itc[..., keep], band[..., keep]
            d[f"n__{g}"] = np.int32(n)
        np.savez_compressed(out, times=times[keep], freqs=FREQS, ch_names=np.array(ep.ch_names), po=np.array(PO), **d)
        print(f"  exported {ses}: {len(md)} epochs ({correct.sum()} correct)")


# ================================================================= step 2: load + measures
def load(S):
    data = {}
    for ses in S.session:
        z = np.load(TFD / f"{ses}_itc.npz", allow_pickle=False)
        times, ch = z["times"], list(z["ch_names"])
        data[ses] = {ver: {kind: np.stack([z[f"{kind}__{ver}__{c}"] for c in CONDS]) for kind in ("ppc", "itc", "band")}
                     for ver in VERSIONS}
    return data, times, ch


def tc(D, k, kind="ppc"):
    """cond x time: phase locking averaged over 2-20 Hz and over cluster k."""
    return np.nanmean(D[kind][:, CL_IDX[k]], axis=(1, 2))


def _idx(times, t):
    return int(np.argmin(np.abs(times - t)))


def syl_rel(times):
    dt = times[1] - times[0]
    return np.arange(int(round(SYL_WIN[0] / dt)), int(round(SYL_WIN[1] / dt)) + 1), dt


def locked(x, times, t0):
    """x[..., time] -> x[..., syllable-locked window]; onsets fall on the 64-Hz grid."""
    r, _ = syl_rel(times)
    return x[..., _idx(times, t0) + r]


def syllable_tc(x, times, stream, attend):
    """Collapsed syllable-locked time course(s) from a cond x ... x time array."""
    return np.nanmean([locked(x[CONDS.index(c)], times, t0) for t0, c in E.syllable_items(stream, attend)], 0)


def latencies(data, times, ver):
    r, dt = syl_rel(times)
    rel = r * dt
    w = (rel >= PEAK_SEARCH[0]) & (rel <= PEAK_SEARCH[1])
    syl, cue = [], []
    for D in data.values():
        x = tc(D[ver], "pooled")
        syl.append(np.nanmean([locked(x[[CONDS.index(c) for c in conds]].mean(0), times, t0)
                               for _, t0, conds in E.syllable_onsets()], 0))
        cue.append(locked(np.nanmean(x, 0), times, K.TIMING["cue"]))
    return {"syllable": float(rel[w][np.nanargmax(np.nanmean(syl, 0)[w])]),
            "cue": float(rel[w][np.nanargmax(np.nanmean(cue, 0)[w])])}


def win(x, times, t):
    return float(np.nanmean(x[(times >= t - HALF) & (times <= t + HALF)]))


def subject_measures(data, times, S, lat):
    rows = []
    info = S.set_index("session")
    r_idx, dt = syl_rel(times)
    rel = r_idx * dt
    for ses, Dv in data.items():
        for ver in VERSIONS:
            L = lat[ver]
            r = dict(session=ses, version=ver)
            for k in CL_IDX:
                x = tc(Dv[ver], k)
                for name, t0, conds in E.syllable_onsets():
                    for c in conds:
                        r[f"{k}_{name}_{c.split('_')[0]}"] = win(x[CONDS.index(c)], times, t0 + L["syllable"])
                for kind, names in (("target", ["target1", "target2", "target3"]),
                                    ("int_early", ["int_early1", "int_early2", "int_early3"]),
                                    ("int_late", ["int_late1", "int_late2", "int_late3"])):
                    for a in K.ATTEND:
                        r[f"{k}_{kind}_{a}"] = np.mean([r[f"{k}_{n}_{a}"] for n in names])
                    r[f"{k}_{kind}_delta"] = r[f"{k}_{kind}_broad"] - r[f"{k}_{kind}_focal"]
                    for n in names:
                        r[f"{k}_{n}_delta"] = r[f"{k}_{n}_broad"] - r[f"{k}_{n}_focal"]
                for stream, tag in (("target", "tsyl"), ("interrupter", "isyl")):
                    for a in K.ATTEND:
                        r[f"{k}_{tag}_{a}"] = win(syllable_tc(x, times, stream, a), rel, L["syllable"])
                for a in K.ATTEND:
                    pooled = np.nanmean(x[[CONDS.index(f"{a}_{i}") for i in K.INTERRUPTER]], 0)
                    r[f"{k}_cue_{a}"] = win(pooled, times, K.TIMING["cue"] + L["cue"])
                for m in ("tsyl", "isyl", "cue"):
                    r[f"{k}_{m}_delta"] = r[f"{k}_{m}_broad"] - r[f"{k}_{m}_focal"]
            r["lat_isyl_delta"] = r["right_isyl_delta"] - r["left_isyl_delta"]
            rows.append(r)
    M = pd.DataFrame(rows)
    return M.merge(info[["subject", "group", "stimulant", "caffeine", "day"]], left_on="session", right_index=True)


FAMS = [
    ("ITC", [f"{k}_{m}_delta" for k in SIDES for m in ("target", "int_early", "int_late")]),
    ("ITC_collapsed", [f"{k}_{m}_delta" for k in SIDES for m in ("tsyl", "isyl")]),
    ("ITC_lateralization", ["lat_isyl_delta"]),
    ("cue", [f"{k}_cue_delta" for k in SIDES]),
    ("secondary", [f"{k}_{m}_{a}" for k in SIDES for m in ("target", "int_early", "int_late", "tsyl", "isyl", "cue")
                   for a in K.ATTEND] + [f"{k}_int_{t}{n}_delta" for k in SIDES for t in ("early", "late") for n in (1, 2, 3)]),
]
ITC_PANELS = [("target", "ITC · targets, no interrupter\n(preparing to switch)"), ("int_early", "ITC · early interrupter"),
              ("int_late", "ITC · late interrupter"), ("cue", "Cue\n(broad vs focal cue picture)")]
PAIRED_PANELS = [("pooled_tsyl", "Target syllables (collapsed)"), ("pooled_isyl", "Interrupter syllables (collapsed)"),
                 ("pooled_int_early", "Early interrupter (3 syllables)"), ("pooled_int_late", "Late interrupter (3 syllables)")]
SYL_PANELS = [("tsyl", "ITC · target syllables\n(collapsed, all trials)"),
              ("isyl", "ITC · interrupter syllables\n(collapsed, early + late)"),
              ("lat_isyl", "ITC lateralization\nright − left cluster\n(interrupter syllables)")]


# ================================================================= figures
def _stack(data, sessions, ver, k="pooled", kind="ppc"):
    return np.array([tc(data[s][ver], k, kind) for s in sessions])          # subj x cond x time


def _kind_label(kind):
    return "PPC" if kind == "ppc" else "ITC"


def fig_grand(data, times, S1, g, ver, kind="ppc", wclus=None):
    """Whole-trial phase locking, focal vs broad, mean ± SEM; black bars = broad ≠ focal (within-group cluster test)."""
    G = E.group_members(S1, g)
    X = _stack(data, G.session, ver, kind=kind)
    fig, axes, tl = E._layout()
    for c, typ in enumerate(K.INTERRUPTER):
        ax = axes[0, c]
        for a, col, lab in (("focal", E.FOCAL_COLOR, "focal (ignore left)"), ("broad", E.BROAD_COLOR, "broad (prepare to switch)")):
            E._line(ax, times, X[:, CONDS.index(f"{a}_{typ}")], col, lab)
        ax.axhline(0, color="#a3a29c", lw=0.6)
        tr = ax.get_xaxis_transform()
        for t0, t1, _, pv in (wclus or {}).get(typ, []):
            if pv < 0.05:
                ax.plot([t0, t1], [0.965, 0.965], transform=tr, color="#0b0b0b", lw=3, solid_capstyle="butt")
        E._cue_line(ax)
        E._onset_lines(ax, typ)
        ax.grid(axis="x", visible=False)
        ax.set_title({"none": "No interrupter", "early": "Early interrupter", "late": "Late interrupter"}[typ])
        ax.set_xlim(*E.T_WIN)
        _timeline_and_label(tl[c], typ, c)
    axes[0, 0].set_ylabel(f"{E.POOLED_LABEL}\n{_kind_label(kind)}, 2–20 Hz", fontsize=8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8)
    bars = "; black bars = broad ≠ focal (within-group cluster permutation, p < .05)" if wclus else ""
    fig.suptitle(f"Phase locking ({_kind_label(kind)}) — {g} (n={len(G)}), day 1 · {E._vlabel(ver)} · mean ± SEM{bars}",
                 x=0.01, ha="left", fontsize=10)
    return fig


def _timeline_and_label(ax, typ, c):
    E._timeline(ax, typ, c == 0)
    ax.set_xlabel("Time from syllable-stream onset (s)")


def fig_diffwaves(data, times, S1, groups, ver, clus):
    fig, axes, tl = E._layout(4.3)
    comps = list(next(iter(clus.values())).keys()) if clus else []
    for c, typ in enumerate(K.INTERRUPTER):
        ax = axes[0, c]
        for g in groups:
            G = E.group_members(S1, g)
            X = _stack(data, G.session, ver)
            E._line(ax, times, X[:, CONDS.index(f"broad_{typ}")] - X[:, CONDS.index(f"focal_{typ}")], K.COLORS[g],
                    f"{g} (n={len(G)})")
        ax.axhline(0, color="#a3a29c", lw=0.6)
        E._cue_line(ax)
        E._onset_lines(ax, typ)
        ax.grid(axis="x", visible=False)
        tr = ax.get_xaxis_transform()
        for j, comp in enumerate(comps):
            y = 0.97 - 0.055 * j
            if c == 2:
                ax.text(1.01, y, comp, transform=ax.transAxes, fontsize=6.5, va="center", color="#52514e")
            for t0, t1, _, pv in clus.get(typ, {}).get(comp, []):
                if pv < 0.05:
                    ax.plot([t0, t1], [y, y], transform=tr, color="#0b0b0b", lw=3, solid_capstyle="butt")
        ax.set_title({"none": "No interrupter", "early": "Early interrupter", "late": "Late interrupter"}[typ])
        ax.set_xlim(*E.T_WIN)
        _timeline_and_label(tl[c], typ, c)
    lo, hi = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(lo, hi + (hi - lo) * 0.09 * (len(comps) + 1))
    axes[0, 0].set_ylabel(f"{E.POOLED_LABEL}\nbroad − focal PPC", fontsize=8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=len(groups), fontsize=8)
    fig.suptitle(f"Attentional modulation of phase locking (broad − focal PPC, 2–20 Hz), day 1 · {E._vlabel(ver)} · "
                 f"mean ± SEM; black bars = group cluster-permutation p < .05", x=0.01, ha="left", fontsize=9)
    return fig


def fig_syllables(data, times, S1, ver, R, lat, groups=None):
    """Collapsed syllable-locked PPC. Rows = stream; columns = focal, broad, broad - focal, groups overlaid."""
    groups = groups or E.MAIN_GROUPS
    r_idx, dt = syl_rel(times)
    rel = r_idx * dt
    L = lat[ver]["syllable"]
    Rv = R[R.version == ver]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), layout="constrained", sharex=True)
    for r, st in enumerate(E.STREAMS):
        tag = "tsyl" if st == "target" else "isyl"
        W = {g: {a: np.array([syllable_tc(tc(data[s][ver], "pooled"), times, st, a) for s in E.group_members(S1, g).session])
                 for a in K.ATTEND} for g in groups}
        for c, a in enumerate(("focal", "broad", "delta")):
            ax = axes[r, c]
            for g in groups:
                X = W[g]["broad"] - W[g]["focal"] if a == "delta" else W[g][a]
                E._line(ax, rel, X, K.COLORS[g], f"{g} (n={len(X)})")
            ax.axhline(0, color="#a3a29c", lw=0.6)
            ax.axvline(0, color="#0b0b0b", lw=0.8)
            ax.axvspan(L - HALF, L + HALF, color="#e9e8e3", lw=0, zorder=0)
            if st == "target":
                ax.axvline(0.1, color="#7d7b74", lw=0.7, ls=(0, (1.5, 1.5)), zorder=0)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title({"focal": "Focal (ignore left)", "broad": "Broad (prepare to switch)",
                              "delta": "Broad − focal"}[a], fontsize=9.5)
            if a == "delta":
                m = f"pooled_{tag}_delta"
                lines = []
                for g in groups:
                    q = Rv[(Rv.test == "delta vs 0 (day 1)") & (Rv.comparison == g) & (Rv.measure == m)].p_perm
                    lines.append(f"{g}: {E._stars(float(q.iloc[0])) or 'n.s.'}" if len(q) else "")
                cmps = ["ADHD (all) - NT"] if groups == E.MAIN_GROUPS else \
                       ["ADHD-off - NT", "ADHD-on - NT", "ADHD-on - ADHD-off"]
                for cmp in cmps:
                    q = Rv[(Rv.test == "group (day 1)") & (Rv.comparison == cmp) & (Rv.measure == m)].p_perm
                    if len(q):
                        lines.append(f"{cmp.replace(' - ', ' vs ')}: {E._stars(float(q.iloc[0])) or 'n.s.'}")
                ax.text(1.03, 0.5, "PPC (grey band),\nbroad vs focal\n\n" + "\n".join(lines), transform=ax.transAxes,
                        ha="left", va="center", fontsize=7, color="#3b3a36")
        lo = min(axes[r, 0].get_ylim()[0], axes[r, 1].get_ylim()[0])
        hi = max(axes[r, 0].get_ylim()[1], axes[r, 1].get_ylim()[1])
        for ax in axes[r, :2]:
            ax.set_ylim(lo, hi)
        axes[r, 0].set_ylabel(f"{E.STREAMS[st]}\nPPC, 2–20 Hz", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time from syllable onset (s)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=len(groups), fontsize=8)
    fig.suptitle(f"Collapsed syllable-locked phase locking (PPC, 2–20 Hz, {E.POOLED_LABEL}), day 1 · {E._vlabel(ver)} · "
                 f"mean ± SEM\nGrey band = measurement window ({L * 1000:.0f} ± {HALF * 1000:.0f} ms); dotted = distractor "
                 f"syllable (+0.1 s). Stars: within group = sign-flip permutation; between groups = caffeine-adjusted "
                 f"permutation on broad − focal (uncorrected).", x=0.01, ha="left", fontsize=8)
    return fig


def _syl_tf(D, times, stream, attend):
    """pooled-cluster TF (freq x syllable time) for one participant."""
    x = np.nanmean(D["ppc"][:, CL_IDX["pooled"]], axis=1)          # cond x freq x time
    return syllable_tc(x, times, stream, attend)


def fig_tf(data, times, S1, ver, lat, groups=None):
    """Time-frequency maps of collapsed syllable-locked PPC: all day-1 participants (focal, broad, broad - focal)
    and broad - focal per group."""
    groups = groups or E.MAIN_GROUPS
    r_idx, dt = syl_rel(times)
    rel = r_idx * dt
    fig, axes = plt.subplots(2, 3 + len(groups), figsize=(2.1 * (3 + len(groups)) + 0.6, 5.2), layout="constrained",
                             sharex=True, sharey=True)
    for r, st in enumerate(E.STREAMS):
        T = {a: {s: _syl_tf(data[s][ver], times, st, a) for s in S1.session} for a in K.ATTEND}
        allf = np.nanmean([T["focal"][s] for s in S1.session], 0)
        allb = np.nanmean([T["broad"][s] for s in S1.session], 0)
        vmax = np.nanpercentile(np.r_[allf.ravel(), allb.ravel()], 99)
        panels = [("all · focal", allf, "abs"), ("all · broad", allb, "abs"), ("all · broad − focal", allb - allf, "d")]
        for g in groups:
            G = E.group_members(S1, g).session
            panels.append((f"{g} · broad − focal", np.nanmean([T["broad"][s] - T["focal"][s] for s in G], 0), "d"))
        dmax = np.nanpercentile(np.abs(np.concatenate([p[1].ravel() for p in panels if p[2] == "d"])), 99)
        for c, (ttl, Z, typ) in enumerate(panels):
            ax = axes[r, c]
            if typ == "abs":
                im_a = ax.pcolormesh(rel, FREQS, Z, cmap="viridis", vmin=0, vmax=vmax, shading="nearest")
            else:
                im_d = ax.pcolormesh(rel, FREQS, Z, cmap="RdBu_r", vmin=-dmax, vmax=dmax, shading="nearest")
            ax.axvline(0, color="white" if typ == "abs" else "#0b0b0b", lw=0.8)
            ax.axvline(lat[ver]["syllable"], color="white" if typ == "abs" else "#52514e", lw=0.6, ls=":")
            if r == 0:
                ax.set_title(ttl, fontsize=8)
            ax.grid(False)
        axes[r, 0].set_ylabel(f"{E.STREAMS[st]}\nfrequency (Hz)", fontsize=8)
        fig.colorbar(im_a, ax=axes[r, :2], shrink=0.8, label="PPC", pad=0.01)
        fig.colorbar(im_d, ax=axes[r, 2:], shrink=0.8, label="broad − focal PPC", pad=0.01)
    for ax in axes[-1]:
        ax.set_xlabel("Time from syllable (s)")
    fig.suptitle(f"Collapsed syllable-locked phase locking by frequency ({E.POOLED_LABEL}), day 1 · {E._vlabel(ver)}; "
                 f"dotted = measurement latency", x=0.01, ha="left", fontsize=9)
    return fig


def _topo_levels(ax, vals, xy, lo, hi, cmap):
    from scipy.interpolate import griddata
    gx, gy = np.meshgrid(np.linspace(-0.55, 0.55, 120), np.linspace(-0.55, 0.55, 120))
    z = griddata(xy, vals, (gx, gy), method="cubic")
    z[np.hypot(gx, gy) > 0.52] = np.nan
    im = ax.contourf(gx, gy, z, levels=np.linspace(lo, hi, 13), cmap=cmap, extend="both")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.52 * np.cos(th), 0.52 * np.sin(th), color="#0b0b0b", lw=0.8)
    ax.plot([-0.05, 0, 0.05], [0.515, 0.58, 0.515], color="#0b0b0b", lw=0.8)
    ax.scatter(xy[:, 0], xy[:, 1], s=1.5, color="#0b0b0b", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    return im


def fig_topo(data, times, ch, S1, ver, lat, groups=None):
    groups = groups or E.MAIN_GROUPS
    xy = E._locs_xy(ch)
    cl = [ch.index(c) for c in PO]
    r_idx, dt = syl_rel(times)
    rel = r_idx * dt
    L = lat[ver]["syllable"]
    cols = [(st, a) for st in ("target", "interrupter") for a in ("focal", "broad", "delta")]
    vals = {}
    for g in groups:
        G = E.group_members(S1, g).session
        for st, a in cols:
            per = []
            for s in G:
                B = data[s][ver]["band"]                                     # cond x ch x time
                amp = lambda at: np.array([win(w, rel, L) for w in syllable_tc(B, times, st, at)])
                per.append(amp("broad") - amp("focal") if a == "delta" else amp(a))
            vals[(g, st, a)] = np.nanmean(per, 0)
    vabs = np.nanpercentile([v for (g, s, a), v in vals.items() if a != "delta"], 99)
    vdel = np.nanpercentile(np.abs([v for (g, s, a), v in vals.items() if a == "delta"]), 98)
    fig, axes = plt.subplots(len(groups), 6, figsize=(10.5, 1.9 * len(groups) + 1.2), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92 if len(groups) > 2 else 0.86,
                        bottom=0.12 if len(groups) > 2 else 0.2, wspace=0.05, hspace=0.12)
    for r, g in enumerate(groups):
        for c, (st, a) in enumerate(cols):
            ax = axes[r, c]
            if a == "delta":
                im_d = _topo_levels(ax, vals[(g, st, a)], xy, -vdel, vdel, "RdBu_r")
            else:
                im_a = _topo_levels(ax, vals[(g, st, a)], xy, 0, vabs, "viridis")
            ax.scatter(xy[cl, 0], xy[cl, 1], s=10, facecolor="none", edgecolor="#0b0b0b", lw=0.6, zorder=4)
            if r == 0:
                ax.set_title(f"{st} syll. · {'broad − focal' if a == 'delta' else a}", fontsize=8)
            if c == 0:
                ax.text(-0.75, 0, g, rotation=90, va="center", ha="center", fontsize=8.5, color=K.COLORS[g], fontweight="bold")
    for im, (c0, c1), lab, ticks in ((im_a, (0, 1), "focal & broad PPC", [0, vabs / 2, vabs]),
                                     (im_d, (2, 2), "broad − focal PPC", [-vdel, 0, vdel])):
        b0, b1 = axes[-1, c0].get_position(), axes[-1, c1].get_position()
        cax = fig.add_axes([b0.x0 + 0.01, 0.06 if len(groups) > 2 else 0.09, b1.x1 - b0.x0 - 0.02,
                            0.015 if len(groups) > 2 else 0.025])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks, format="%.3f")
        cb.set_label(lab, fontsize=7)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Scalp maps of collapsed syllable phase locking (PPC 2–20 Hz, {L * 1000:.0f} ± {HALF * 1000:.0f} ms after "
                 f"each syllable) · {E._vlabel(ver)}; circles = PO clusters", x=0.01, ha="left", fontsize=8.5)
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
    E.STATS_NAME = "itc_stats.csv"
    S = K.sessions()
    if not a.skip_export:
        export(S, overwrite=a.overwrite_export)
    data, times, ch = load(S)
    lat = {ver: latencies(data, times, ver) for ver in VERSIONS}
    print("Phase-locking peak latencies (ms after onset):",
          json.dumps({v: {k: round(x * 1000) for k, x in L.items()} for v, L in lat.items()}))
    for v, L in lat.items():
        for k, x in L.items():
            if min(abs(x - PEAK_SEARCH[0]), abs(x - PEAK_SEARCH[1])) < 1.5 * (times[1] - times[0]):
                print(f"  WARNING: {v} {k} peak {x:.3f} s is at the edge of the search window {PEAK_SEARCH}")
    M = subject_measures(data, times, S, lat)
    P = K.within_pairs(S)
    R = pd.concat([E.group_stats(M, FAMS), E.condition_effects(M, FAMS), E.within_stats(M, P, FAMS)], ignore_index=True)

    S1 = S[S.day == 1]
    clus_rows, clus = [], {}
    for ver in VERSIONS:
        clus[ver] = {}
        for typ in K.INTERRUPTER:
            res = {}
            for ga, gb in K.COMPARISONS:
                A, B = E.group_members(S1, ga), E.group_members(S1, gb)
                XA, XB = _stack(data, A.session, ver), _stack(data, B.session, ver)
                dA = XA[:, CONDS.index(f"broad_{typ}")] - XA[:, CONDS.index(f"focal_{typ}")]
                dB = XB[:, CONDS.index(f"broad_{typ}")] - XB[:, CONDS.index(f"focal_{typ}")]
                out = E.cluster_test(dA, dB, (A.caffeine == "yes").values, (B.caffeine == "yes").values, times,
                                     n_perm=2000 if ver == MAIN else 1000)
                short = f"{gb.replace('ADHD-', '').replace('ADHD (all)', 'ADHD')} vs {ga.replace('ADHD-', '')}"
                res[short] = out
                for t0, t1, mass, p in out:
                    clus_rows.append(dict(version=ver, interrupter=typ, comparison=f"{gb} - {ga}",
                                          t_start=t0, t_end=t1, mass=mass, p=p))
            clus[ver][typ] = res

    M.to_csv(OUT / "itc_subject_measures.csv", index=False)
    R.to_csv(OUT / "itc_stats.csv", index=False)
    pd.DataFrame(lat).to_csv(OUT / "itc_latencies.csv")

    # within-group broad vs focal time-course tests (shown on the whole-trial figures)
    wclus = {}
    for g in K.GROUPS + ["ADHD (all)"]:
        X = _stack(data, E.group_members(S1, g).session, MAIN)
        wclus[g] = {typ: E.signflip_cluster(X[:, CONDS.index(f"broad_{typ}")] - X[:, CONDS.index(f"focal_{typ}")], times)
                    for typ in K.INTERRUPTER}
        for typ, out in wclus[g].items():
            for t0, t1, mass, p in out:
                clus_rows.append(dict(version=MAIN, interrupter=typ, test="broad vs focal within group", comparison=g,
                                      t_start=t0, t_end=t1, mass=mass, p=p))
    pd.DataFrame(clus_rows).to_csv(OUT / "itc_cluster_tests.csv", index=False)

    # main = NT vs ADHD (all), correct trials; supplementary = ADHD-off / ADHD-on, within-person, all trials, classic ITC.
    # Old itc_* figures are removed first so files from earlier layouts don't linger.
    for d in (K.FIG_DIR / SUB, K.FIG_DIR / SUB / "supplementary"):
        for f in list(d.glob("itc_*.png")) + list(d.glob("itc_*.pdf")) if d.exists() else []:
            f.unlink()
    written = []
    for ver in VERSIONS:
        main_v = ver == MAIN
        sfx = "" if main_v else f"__{ver}"
        figs = []                                   # (name, figure, supplementary?)
        if main_v:
            for g in K.GROUPS + ["ADHD (all)"]:
                figs.append((f"itc_grand_{g.replace(' (all)', '_pooled')}", fig_grand(data, times, S1, g, ver, wclus=wclus[g]),
                             g not in E.MAIN_GROUPS))
            figs += [("itc_tf_syllables", fig_tf(data, times, S1, ver, lat), False),
                     ("itc_tf_syllables_3groups", fig_tf(data, times, S1, ver, lat, ["NT", "ADHD-off", "ADHD-on"]), True),
                     ("itc_topomaps", fig_topo(data, times, ch, S1, ver, lat), False),
                     ("itc_topomaps_4groups", fig_topo(data, times, ch, S1, ver, lat, ["NT", "ADHD (all)", "ADHD-off", "ADHD-on"]),
                      True)]
            for g in ("NT", "ADHD (all)"):      # classic ITC (trial-count biased), for comparison with the original analysis
                figs.append((f"itc_grand_classicITC_{g.replace(' (all)', '_pooled')}",
                             fig_grand(data, times, S1, g, ver, kind="itc"), True))
        c3 = {t: {c: v for c, v in d.items() if "ADHD vs" not in c} for t, d in clus[ver].items()}
        cp = {t: {c: v for c, v in d.items() if "ADHD vs" in c} for t, d in clus[ver].items()}
        figs += [("itc_diffwaves", fig_diffwaves(data, times, S1, E.MAIN_GROUPS, ver, cp), False),
                 ("itc_diffwaves_3groups", fig_diffwaves(data, times, S1, K.GROUPS, ver, c3), True),
                 ("itc_syllables_collapsed", fig_syllables(data, times, S1, ver, R, lat), False),
                 ("itc_syllables_collapsed_3groups", fig_syllables(data, times, S1, ver, R, lat, K.GROUPS), True),
                 ("itc_focal_vs_broad", E.fig_paired(M, R, ver, E.MAIN_GROUPS, PAIRED_PANELS,
                                                     f"PPC, {E.POOLED_LABEL}\n{lat[ver]['syllable'] * 1000:.0f} ± "
                                                     f"{HALF * 1000:.0f} ms after each syllable",
                                                     "Phase locking, focal vs broad"), False),
                 ("itc_focal_vs_broad_3groups", E.fig_paired(M, R, ver, K.GROUPS, PAIRED_PANELS, f"PPC, {E.POOLED_LABEL}",
                                                             "Phase locking, focal vs broad"), True)]
        for pooled, gname in ((True, ""), (False, "_3groups")):
            figs += [(f"itc_delta{gname}", E.fig_box(M, R, ver, pooled, ITC_PANELS, "Attentional modulation of phase locking "
                                                     "(PPC, larger = more locking on broad)", unit=UNIT), not pooled),
                     (f"itc_delta_collapsed{gname}", E.fig_box(M, R, ver, pooled, SYL_PANELS,
                                                               "Phase locking on collapsed syllables (PPC)", unit=UNIT), not pooled)]
        figs.append(("itc_within_person", E.fig_within(M, P, R, ver, ITC_PANELS[:3] + SYL_PANELS, "phase locking", unit=UNIT),
                     True))
        for name, f, supp in figs:
            written.append(K.save(f, name + sfx, f"{SUB}/supplementary" if (supp or not main_v) else SUB))
            plt.close(f)
    print(f"wrote {len(written)} figures to {K.FIG_DIR / SUB} (main) and {K.FIG_DIR / SUB / 'supplementary'}")

    pd.set_option("display.width", 220)
    show = R[(R.version == MAIN) & (R.family != "secondary") & (R.p_perm < 0.05) & (R.comparison == "ADHD (all) - NT")]
    print("\nNT vs ADHD (all), correct trials, p < .05 (uncorrected; everything else in itc_stats.csv):")
    cols = ["test", "comparison", "family", "measure", "mean_a", "mean_b", "diff_raw", "p_perm", "p_fdr"]
    print(show[cols].round(4).to_string(index=False) if len(show) else "  none")
    eff = R[(R.version == MAIN) & (R.test == "delta vs 0 (day 1)") & R.comparison.isin(E.MAIN_GROUPS)]
    print("\nBroad - focal within NT and ADHD (all):")
    print(eff.pivot_table(index="measure", columns="comparison", values=["mean_a", "dz", "p_perm"]).round(4).to_string())


if __name__ == "__main__":
    main()
