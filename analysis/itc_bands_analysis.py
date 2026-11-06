"""ITC by frequency band: theta (4-8 Hz) and slow (2-4 Hz) phase locking, reported separately from the 2-20 Hz average.

Run from the python/ folder (MNE environment active):
    python analysis/itc_bands_analysis.py                 # export band scalp maps (MNE, quick) + measures, stats, figures
    python analysis/itc_bands_analysis.py --skip-export   # numpy only; scalp maps only if the band export exists

Nothing in the 2-20 Hz analysis is overwritten. Everything is written next to it:
    derivatives/analysis/itc_bands/<band>/   itc_subject_measures.csv, itc_stats.csv, itc_cluster_tests.csv, itc_latencies.csv
    derivatives/analysis/itc_bands/tf/       <session>_bands.npz (64-channel PPC per band, for scalp maps only)
    figures/itc_bands/<band>/ (+ supplementary/)

It reads the PO/O phase-locking export that itc_analysis.py already made (PPC at 1-Hz steps, 2-20 Hz) and averages
PPC over the frequencies of each band: theta = 4,5,6,7,8 Hz; slow = 2,3,4 Hz (4 Hz is in both). Same participants,
pooled PO/O cluster, families, statistics and figure layouts as the 2-20 Hz analysis.

Measures per band (broad - focal = "delta"):
  syllable-locked (as in itc_analysis.py): 100-ms window at the band's own syllable peak latency (blind to condition);
          target / int_early / int_late (3 syllables each), collapsed tsyl / isyl, lateralization, cue.
  stream following (new; mostly meant for the slow band, how well each stream is followed over time):
          prep         0-1.0 s mean, after the cue and before any interrupter (same trials, different cue picture)
          stream_early 1.0-2.65 s mean, early-interrupter trials (first interrupter syllable to 0.4 s after the last)
          stream_late  1.5-3.15 s mean, late-interrupter trials
Scalp maps need the 64-channel band PPC, so the (quick) MNE export runs once; the rest is numpy.
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
import itc_analysis as I

BANDS = {"theta": (4.0, 8.0), "slow": (2.0, 4.0)}
BAND_LABEL = {"theta": "theta 4–8 Hz", "slow": "slow 2–4 Hz"}
ROOT = K.ANALYSIS_DIR / "itc_bands"
TFD = ROOT / "tf"
CONDS, VERSIONS, MAIN = I.CONDS, I.VERSIONS, I.MAIN
STREAM_WIN = {"early": (1.0, 2.65), "late": (1.5, 3.15)}
STREAM_FAMS = ("ITC_stream", [f"pooled_{m}_delta" for m in ("prep", "stream_early", "stream_late")])
STREAM_PANELS = [("prep", "Preparation\n(0–1.0 s after cue)"), ("stream_early", "Early interrupter stream\n(1.0–2.65 s)"),
                 ("stream_late", "Late interrupter stream\n(1.5–3.15 s)")]
PAIRED_STREAM = [("pooled_prep", "Preparation (0–1.0 s)"), ("pooled_stream_early", "Early interrupter stream"),
                 ("pooled_stream_late", "Late interrupter stream")]


def fmask(band):
    lo, hi = BANDS[band]
    return (I.FREQS >= lo - 1e-9) & (I.FREQS <= hi + 1e-9)


# ================================================================= 64-channel band export (scalp maps)
def phase_locking_bands(X, sfreq, masks):
    """PPC averaged over the frequencies of each band, all channels: {(mask, band): channels x time}."""
    X = np.asarray(X, np.float32)
    n_ch, n_t = X.shape[1], X[..., ::I.DECIM].shape[-1]
    out = {(g, b): np.zeros((n_ch, n_t), np.float64) for g in masks for b in BANDS}
    for b in BANDS:
        fi = np.where(fmask(b))[0]
        for i in fi:
            W = fftconvolve(X, I.morlet(I.FREQS[i], I.CYCLES[i], sfreq)[None, None, :], mode="same", axes=-1)[..., ::I.DECIM]
            U = W / np.maximum(np.abs(W), 1e-30)
            for g, m in masks.items():
                n = int(m.sum())
                if n < 2:
                    out[(g, b)][:] = np.nan
                    continue
                s = U[m].sum(0)
                out[(g, b)] += (np.abs(s) ** 2 - n) / (n * (n - 1)) / len(fi)
    return out


def export_bands(S, overwrite=False):
    import mne
    TFD.mkdir(parents=True, exist_ok=True)
    for ses in S.session:
        out = TFD / f"{ses}_bands.npz"
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
        res = phase_locking_bands(ep.get_data(), ep.info["sfreq"], masks)
        times = ep.times[::I.DECIM]
        keep = (times >= E.T_WIN[0] - 1e-9) & (times <= E.T_WIN[1] + 1e-9)
        np.savez_compressed(out, times=times[keep], ch_names=np.array(ep.ch_names),
                            **{f"{b}__{g}": v[..., keep].astype(np.float32) for (g, b), v in res.items()})
        print(f"  exported bands {ses}")


def band_data(data, band, S):
    """Slice the 2-20 Hz export to one band; attach the band's 64-channel maps when they have been exported."""
    fm = fmask(band)
    out = {ses: {ver: {"ppc": D[ver]["ppc"][:, :, fm], "itc": D[ver]["itc"][:, :, fm], "band": None} for ver in VERSIONS}
           for ses, D in data.items()}
    files = [TFD / f"{s}_bands.npz" for s in S.session]
    if all(f.exists() for f in files):
        for ses, f in zip(S.session, files):
            z = np.load(f, allow_pickle=False)
            for ver in VERSIONS:
                out[ses][ver]["band"] = np.stack([z[f"{band}__{ver}__{c}"] for c in CONDS])
        return out, True
    return out, False


# ================================================================= stream-following measures
def stream_measures(data, times):
    def mean_win(x, t0, t1):
        return float(np.nanmean(x[(times >= t0) & (times <= t1)]))
    rows = []
    for ses, Dv in data.items():
        for ver in VERSIONS:
            x = I.tc(Dv[ver], "pooled")
            r = dict(session=ses, version=ver)
            for a in K.ATTEND:
                allc = [CONDS.index(f"{a}_{i}") for i in K.INTERRUPTER]
                r[f"pooled_prep_{a}"] = mean_win(np.nanmean(x[allc], 0), 0.0, 1.0)
                for typ, (t0, t1) in STREAM_WIN.items():
                    r[f"pooled_stream_{typ}_{a}"] = mean_win(x[CONDS.index(f"{a}_{typ}")], t0, t1)
            for m in ("prep", "stream_early", "stream_late"):
                r[f"pooled_{m}_delta"] = r[f"pooled_{m}_broad"] - r[f"pooled_{m}_focal"]
            rows.append(r)
    return pd.DataFrame(rows)


# ================================================================= one band
def run_band(band, data0, times, ch, S, P):
    I.BAND_LABEL = BAND_LABEL[band]
    OUT = ROOT / band
    SUB = f"itc_bands/{band}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n==================== {BAND_LABEL[band]} ====================")
    data, has_topo = band_data(data0, band, S)
    lat = {ver: I.latencies(data, times, ver) for ver in VERSIONS}
    print("Peak latencies (ms after onset):", json.dumps({v: {k: round(x * 1000) for k, x in L.items()} for v, L in lat.items()}))
    for v, L in lat.items():
        for k, x in L.items():
            if min(abs(x - I.PEAK_SEARCH[0]), abs(x - I.PEAK_SEARCH[1])) < 1.5 * (times[1] - times[0]):
                print(f"  WARNING: {v} {k} peak {x:.3f} s is at the edge of the search window {I.PEAK_SEARCH}")

    M = I.subject_measures(data, times, S, lat)
    M = M.merge(stream_measures(data, times), on=["session", "version"])
    fams = I.FAMS + [STREAM_FAMS]
    R = pd.concat([E.group_stats(M, fams), E.condition_effects(M, fams), E.within_stats(M, P, fams)], ignore_index=True)

    S1 = S[S.day == 1]
    clus_rows, clus = [], {}
    for ver in VERSIONS:
        clus[ver] = {}
        for typ in K.INTERRUPTER:
            res = {}
            for ga, gb in K.COMPARISONS:
                A, B = E.group_members(S1, ga), E.group_members(S1, gb)
                XA, XB = I._stack(data, A.session, ver), I._stack(data, B.session, ver)
                dA = XA[:, CONDS.index(f"broad_{typ}")] - XA[:, CONDS.index(f"focal_{typ}")]
                dB = XB[:, CONDS.index(f"broad_{typ}")] - XB[:, CONDS.index(f"focal_{typ}")]
                o = E.cluster_test(dA, dB, (A.caffeine == "yes").values, (B.caffeine == "yes").values, times,
                                   n_perm=2000 if ver == MAIN else 1000)
                res[f"{gb.replace('ADHD-', '').replace('ADHD (all)', 'ADHD')} vs {ga.replace('ADHD-', '')}"] = o
                for t0, t1, mass, p in o:
                    clus_rows.append(dict(version=ver, interrupter=typ, comparison=f"{gb} - {ga}",
                                          t_start=t0, t_end=t1, mass=mass, p=p))
            clus[ver][typ] = res
    wclus = {}
    for g in K.GROUPS + ["ADHD (all)"]:
        X = I._stack(data, E.group_members(S1, g).session, MAIN)
        wclus[g] = {typ: E.signflip_cluster(X[:, CONDS.index(f"broad_{typ}")] - X[:, CONDS.index(f"focal_{typ}")], times)
                    for typ in K.INTERRUPTER}
        for typ, o in wclus[g].items():
            for t0, t1, mass, p in o:
                clus_rows.append(dict(version=MAIN, interrupter=typ, test="broad vs focal within group", comparison=g,
                                      t_start=t0, t_end=t1, mass=mass, p=p))
    M.to_csv(OUT / "itc_subject_measures.csv", index=False)
    R.to_csv(OUT / "itc_stats.csv", index=False)
    pd.DataFrame(lat).to_csv(OUT / "itc_latencies.csv")
    pd.DataFrame(clus_rows).to_csv(OUT / "itc_cluster_tests.csv", index=False)

    for d in (K.FIG_DIR / SUB, K.FIG_DIR / SUB / "supplementary"):
        if d.exists():
            for f in list(d.glob("itc_*.png")) + list(d.glob("itc_*.pdf")):
                f.unlink()
    written = []
    box_panels = I.ITC_PANELS + STREAM_PANELS
    for ver in VERSIONS:
        main_v = ver == MAIN
        sfx = "" if main_v else f"__{ver}"
        figs = []
        if main_v:
            for g in K.GROUPS + ["ADHD (all)"]:
                figs.append((f"itc_grand_{g.replace(' (all)', '_pooled')}", I.fig_grand(data, times, S1, g, ver, wclus=wclus[g]),
                             g not in E.MAIN_GROUPS))
            if has_topo:
                figs += [("itc_topomaps", I.fig_topo(data, times, ch, S1, ver, lat), False),
                         ("itc_topomaps_4groups", I.fig_topo(data, times, ch, S1, ver, lat,
                                                             ["NT", "ADHD (all)", "ADHD-off", "ADHD-on"]), True)]
        c3 = {t: {c: v for c, v in d.items() if "ADHD vs" not in c} for t, d in clus[ver].items()}
        cp = {t: {c: v for c, v in d.items() if "ADHD vs" in c} for t, d in clus[ver].items()}
        ylab = f"PPC {BAND_LABEL[band]}, {E.POOLED_LABEL}"
        figs += [("itc_diffwaves", I.fig_diffwaves(data, times, S1, E.MAIN_GROUPS, ver, cp), False),
                 ("itc_diffwaves_3groups", I.fig_diffwaves(data, times, S1, K.GROUPS, ver, c3), True),
                 ("itc_syllables_collapsed", I.fig_syllables(data, times, S1, ver, R, lat), False),
                 ("itc_syllables_collapsed_3groups", I.fig_syllables(data, times, S1, ver, R, lat, K.GROUPS), True),
                 ("itc_focal_vs_broad", E.fig_paired(M, R, ver, E.MAIN_GROUPS, I.PAIRED_PANELS,
                                                     f"{ylab}\n{lat[ver]['syllable'] * 1000:.0f} ± {I.HALF * 1000:.0f} ms after each syllable",
                                                     f"Phase locking ({BAND_LABEL[band]}), focal vs broad"), False),
                 ("itc_focal_vs_broad_3groups", E.fig_paired(M, R, ver, K.GROUPS, I.PAIRED_PANELS, ylab,
                                                             f"Phase locking ({BAND_LABEL[band]}), focal vs broad"), True),
                 ("itc_stream_focal_vs_broad", E.fig_paired(M, R, ver, E.MAIN_GROUPS, PAIRED_STREAM, ylab,
                                                            f"Sustained phase locking ({BAND_LABEL[band]}), focal vs broad"), False),
                 ("itc_stream_focal_vs_broad_3groups", E.fig_paired(M, R, ver, K.GROUPS, PAIRED_STREAM, ylab,
                                                                    f"Sustained phase locking ({BAND_LABEL[band]})"), True)]
        for pooled, gname in ((True, ""), (False, "_3groups")):
            figs += [(f"itc_delta{gname}", E.fig_box(M, R, ver, pooled, I.ITC_PANELS,
                                                     f"Attentional modulation of phase locking ({BAND_LABEL[band]}, PPC)",
                                                     unit=I.UNIT), not pooled),
                     (f"itc_delta_collapsed{gname}", E.fig_box(M, R, ver, pooled, I.SYL_PANELS,
                                                               f"Collapsed syllables ({BAND_LABEL[band]}, PPC)", unit=I.UNIT), not pooled),
                     (f"itc_delta_stream{gname}", E.fig_box(M, R, ver, pooled, STREAM_PANELS,
                                                            f"Sustained phase locking ({BAND_LABEL[band]}, PPC)",
                                                            unit=I.UNIT), not pooled)]
        figs.append(("itc_within_person", E.fig_within(M, P, R, ver, I.ITC_PANELS[:3] + I.SYL_PANELS + STREAM_PANELS,
                                                      f"phase locking ({BAND_LABEL[band]})", unit=I.UNIT), True))
        for name, f, supp in figs:
            written.append(K.save(f, name + sfx, f"{SUB}/supplementary" if (supp or not main_v) else SUB))
            plt.close(f)
    print(f"wrote {len(written)} figures to {K.FIG_DIR / SUB}" + ("" if has_topo else " (no scalp maps: run without --skip-export)"))

    pd.set_option("display.width", 220)
    show = R[(R.version == MAIN) & (R.family != "secondary") & (R.p_perm < 0.05) & (R.comparison == "ADHD (all) - NT")]
    print("NT vs ADHD (all), correct trials, p < .05 (uncorrected):")
    print(show[["test", "family", "measure", "mean_a", "mean_b", "diff_raw", "p_perm", "p_fdr"]].round(4).to_string(index=False)
          if len(show) else "  none")
    eff = R[(R.version == MAIN) & (R.test == "delta vs 0 (day 1)") & R.comparison.isin(E.MAIN_GROUPS) & (R.family != "secondary")]
    print("Broad - focal within NT and ADHD (all):")
    print(eff.pivot_table(index="measure", columns="comparison", values=["mean_a", "dz", "p_perm"]).round(4).to_string())
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--overwrite-export", action="store_true")
    ap.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
    a = ap.parse_args()
    K.style()
    plt.rcParams["figure.max_open_warning"] = 0
    E.STATS_NAME = "itc_stats.csv"
    S = K.sessions()
    if not a.skip_export:
        export_bands(S, overwrite=a.overwrite_export)
    data, times, ch = I.load(S)
    P = K.within_pairs(S)
    for band in a.bands:
        run_band(band, data, times, ch, S, P)


if __name__ == "__main__":
    main()
