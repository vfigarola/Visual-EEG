"""Preprocess Jasmine's visual-attention EEG from raw BDFs.

Usage (from the python/ folder, with the env described in README.md active):
    python preprocess.py --list                      # show sessions found
    python preprocess.py JKSA019_XX                  # one session
    python preprocess.py --all                       # every session (skips ones already done)
    python preprocess.py --all --overwrite           # redo everything
    python preprocess.py JKSA019_XX --refit-ica      # refit ICA instead of reusing the saved one

Per session, written to derivatives/preproc_v1/<session>/:
    <session>-epo.fif     epochs (-2.5..4.5 s re: trigger 1), trial metadata attached (epochs.metadata)
    <session>-ica.fif     fitted ICA (reused on reruns so ica_overrides.json edits are cheap)
    <session>_trials.csv  every behavioral visual trial, with kept / rejected status
    <session>_qc.json     everything that happened (bad channels, ICs removed, trials per condition...)
    <session>_report.html ICA components + ERPs for the review step
"""
import argparse
import json
import sys
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import trials as T
from bdf_io import read_channels

import mne
from scipy.signal import resample_poly

mne.set_log_level("WARNING")


# ---------------------------------------------------------------- helpers
def montage():
    """64-ch Barb-lab locations + approximate mastoids (needed for interpolation and ICLabel)."""
    lines = [l for l in Path(C.LOCS_FILE).read_text().splitlines() if l.strip()]
    lines = [l for l in lines if l.split()[-1] not in C.EOG.values()]
    n = len(lines)
    for k, (name, (th, rad)) in enumerate(C.MASTOID_LOCS.items(), start=1):
        lines.append(f"{n + k}\t{th}\t{rad}\t{name}")
    tmp = Path(C.OUT_DIR) / "_montage.locs"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(lines) + "\n")
    return mne.channels.read_custom_montage(tmp)


def scalp_names():
    names = [l.split()[-1] for l in Path(C.LOCS_FILE).read_text().splitlines() if l.strip()]
    return names[:C.N_SCALP]


def load_half(bdf, beh, half):
    """Align triggers to behavior, crop to the visual blocks, resample to SFREQ, return RawArray + metadata."""
    meta, log, hd = T.align_half(bdf, beh, half)
    t0 = meta.t_start.min() - C.CROP_PAD_S
    t1 = np.nanmax(np.r_[meta.t_end.values, meta.t_stim.values + C.TMAX]) + C.CROP_PAD_S
    frac = Fraction(C.SFREQ / hd.sfreq).limit_denominator(64)
    src = hd.labels[:C.N_SCALP] + list(C.MASTOIDS) + list(C.EOG)
    data = []
    for name in src:                               # channel by channel to keep memory low
        x, s0 = read_channels(hd, [name], t0, t1)
        data.append(resample_poly(x[0], frac.numerator, frac.denominator))   # zero-phase anti-alias FIR
    data = np.vstack(data)
    data -= data.mean(axis=1, keepdims=True)       # remove BioSemi DC offsets (avoids filter edge transients)
    data *= 1e-6                                   # uV -> V
    crop_start = s0 / hd.sfreq
    # EOG channels referenced to the mastoid mean (EEG channels are referenced later, after bad-channel detection)
    m = data[C.N_SCALP:C.N_SCALP + 2].mean(0)
    data[C.N_SCALP + 2:] -= m
    names = scalp_names() + list(C.MASTOIDS.values()) + list(C.EOG.values())
    types = ["eeg"] * (C.N_SCALP + 2) + ["eog"] * len(C.EOG)
    info = mne.create_info(names, C.SFREQ, types)
    raw = mne.io.RawArray(data, info, verbose=False)
    meta["sample"] = np.round((meta.t_stim - crop_start) * C.SFREQ).astype(int)
    log.update(crop_s=[round(t0, 2), round(t1, 2)], resample=f"{frac.numerator}/{frac.denominator}")
    return raw, meta, log


def gross_windows(X_uv, sfreq):
    """X_uv: channels x samples (uV, high-passed). Returns list of (onset_s, duration_s) of whole-head junk."""
    w = int(sfreq)
    n = X_uv.shape[1] // w
    Xm = X_uv[:, :n * w] - np.median(X_uv[:, :n * w], axis=0, keepdims=True)
    sd = np.median(Xm.reshape(X_uv.shape[0], n, w).std(axis=2), axis=0)          # per window
    thr = max(C.GROSS_UV, C.GROSS_FACTOR * np.median(sd))
    bad = sd > thr
    segs, i = [], 0
    while i < n:
        if bad[i]:
            j = i
            while j + 1 < n and bad[j + 1]:
                j += 1
            on = max(0.0, i - C.GROSS_PAD_S)
            segs.append((on, (j + 1 + C.GROSS_PAD_S) - on))
            i = j + 1
        else:
            i += 1
    return segs, float(thr)


def detect_flat_noisy(raw_1hz):
    """Stage (a), before ICA: flat or very-high-variance channels (scalp + mastoids)."""
    picks = mne.pick_types(raw_1hz.info, eeg=True, exclude=[])
    names = [raw_1hz.ch_names[i] for i in picks]
    X = raw_1hz.get_data(picks=picks, reject_by_annotation="omit")[:, ::2] * 1e6
    X = X - np.median(X, axis=0, keepdims=True)                     # robust common reference
    sd = X.std(1)
    lv = np.log(sd + 1e-12)
    z = (lv - np.median(lv)) / (1.4826 * np.median(np.abs(lv - np.median(lv))) + 1e-12)
    why = {}
    for i, n in enumerate(names):
        if sd[i] < C.BAD_FLAT_UV:
            why[n] = f"flat (sd={sd[i]:.2f}uV) [pre-ICA]"
        elif z[i] > C.BAD_VAR_Z:
            why[n] = f"high variance (z={z[i]:.1f}) [pre-ICA]"
    return why


def detect_uncorrelated(raw_1hz, already_bad, max_new):
    """Stage (b), after ICA: iteratively flag the channel least correlated with the mean of its 6 nearest
    good neighbours (average reference over good channels, 1-s windows, median over windows)."""
    picks = mne.pick_types(raw_1hz.info, eeg=True, exclude=[])
    names = [raw_1hz.ch_names[i] for i in picks]
    pos = np.array([raw_1hz.info["chs"][i]["loc"][:3] for i in picks])
    X = raw_1hz.get_data(picks=picks, reject_by_annotation="omit") * 1e6
    w = int(raw_1hz.info["sfreq"])
    n = X.shape[1] // w
    W = X[:, :n * w].reshape(len(names), n, w)
    del X
    W -= W.mean(axis=2, keepdims=True)
    return uncorrelated_core(W, names, pos, already_bad, max_new)


def uncorrelated_core(W, names, pos, already_bad, max_new):
    """W: channels x windows x samples (uV, window-demeaned). Pure numpy so it can be tested without MNE."""
    bad, found = set(already_bad), {}
    skip = set(C.MASTOIDS.values())      # mastoids sit at the edge with no neighbours below -> not tested, not used
    for _ in range(max_new):
        good = [i for i in range(len(names)) if names[i] not in bad and names[i] not in skip]
        Wr = W - W[good].mean(axis=0)
        scores = {}
        for i in good:
            d = np.linalg.norm(pos[good] - pos[i], axis=1)
            nb = [good[j] for j in np.argsort(d)[1:7]]
            a, b = Wr[i], Wr[nb].mean(axis=0)
            r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)
            scores[i] = float(np.median(r))
        worst = min(scores, key=scores.get)
        if scores[worst] >= C.BAD_NEIGHBOR_R:
            break
        bad.add(names[worst])
        found[names[worst]] = f"low neighbour corr (r={scores[worst]:.2f}) [post-ICA]"
    return found


def make_epochs(raw, meta):
    events = np.c_[meta["sample"].values + raw.first_samp, np.zeros(len(meta), int), meta.cue.values]
    event_id = {f"{a}/{i}": code for code, (a, i) in C.CUE_CODES.items() if code in set(meta.cue)}
    return mne.Epochs(raw, events, event_id, tmin=C.TMIN, tmax=C.TMAX, baseline=C.BASELINE,
                      metadata=meta.reset_index(drop=True), picks=["eeg", "eog"], preload=True,
                      reject_by_annotation=True, verbose=False)


def exceed_matrix(epochs):
    """epochs x channels: exceeds threshold (bool) and peak |amplitude| (uV)."""
    ep = epochs.copy().pick("eeg").crop(*C.REJECT_WINDOW)
    peak = np.abs(ep.get_data() * 1e6).max(axis=2)
    return peak > C.REJECT_UV, np.array(ep.ch_names), peak


def load_overrides():
    p = Path(C.ICA_OVERRIDES)
    return json.loads(p.read_text()) if p.exists() else {}


def iclabel_probs(raw_ica, ica):
    from mne_icalabel.iclabel import iclabel_label_components
    P = iclabel_label_components(raw_ica, ica)           # (n_components, 7)
    classes = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]
    return pd.DataFrame(P, columns=classes)


# ---------------------------------------------------------------- one session
def process(ses_row, overwrite=False, refit_ica=False):
    ses = ses_row.session
    out = Path(C.OUT_DIR) / ses
    epo_f = out / f"{ses}-epo.fif"
    if epo_f.exists() and not overwrite and not refit_ica:
        print(f"{ses}: already done (use --overwrite)")
        return None
    out.mkdir(parents=True, exist_ok=True)
    qc = dict(session=ses, subject=ses_row.subject, group=ses_row.group, stimulant=ses_row.stimulant,
              caffeine=ses_row.caffeine, day=int(ses_row.day), n_visits=int(ses_row.n_visits),
              date=ses_row.date, params={k: (list(v) if isinstance(v, (tuple, set)) else v)
                                         for k, v in vars(C).items() if k.isupper() and not isinstance(v, Path)})
    ovr = load_overrides().get(ses, {})

    # 1. load both halves (visual blocks only), align with behavior, resample
    raws, metas, logs, offset = [], [], [], 0
    for half in (1, 2):
        bdf, beh = ses_row.get(f"bdf{half}"), ses_row.get(f"beh{half}")
        if not isinstance(bdf, str) or not isinstance(beh, str):
            logs.append(dict(half=half, missing_file=True))
            continue
        raw_h, meta_h, log_h = load_half(bdf, beh, half)
        meta_h["sample"] += offset
        offset += raw_h.n_times
        raws.append(raw_h); metas.append(meta_h); logs.append(log_h)
        print(f"  half {half}: {log_h['matched']}/{log_h['beh_visual_trials']} trials aligned, "
              f"fs={log_h['sfreq']:.0f}, usable={len(meta_h)}")
    qc["halves"] = logs
    raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
    raw.set_montage(montage(), on_missing="ignore")
    meta = pd.concat(metas, ignore_index=True)

    # 2. filtering (zero-phase FIR); ICA copy at 1 Hz
    raw.filter(C.L_FREQ, C.H_FREQ, picks=["eeg", "eog"], phase="zero", fir_design="firwin")
    raw_ica = raw.copy().filter(C.ICA_L_FREQ, None, picks=["eeg", "eog"], phase="zero", fir_design="firwin")

    # 3a. gross-artifact segments -> BAD_gross annotations (skipped by bad-channel detection, ICA, epochs)
    X = raw_ica.get_data(picks="eeg") * 1e6
    segs, thr = gross_windows(X, raw_ica.info["sfreq"])
    del X
    if segs:
        on, du = zip(*segs)
        ann = mne.Annotations(on, du, ["BAD_gross"] * len(segs), orig_time=raw.annotations.orig_time)
        raw.set_annotations(raw.annotations + ann)
        raw_ica.set_annotations(raw_ica.annotations + ann)
    qc["gross_artifact"] = dict(seconds=round(float(sum(d for _, d in segs)), 1), n_segments=len(segs),
                                threshold_uv=round(thr, 1))
    if segs:
        print(f"  gross-artifact segments: {len(segs)} ({qc['gross_artifact']['seconds']} s)")

    # 3b. bad channels, stage (a): flat / very noisy (before ICA)
    manual_bads = "bads" in ovr
    bads = {b: "manual override" for b in ovr["bads"]} if manual_bads else detect_flat_noisy(raw_ica)
    raw.info["bads"] = list(bads); raw_ica.info["bads"] = list(bads)
    print(f"  bad channels before ICA: {list(bads) or 'none'}")

    # 4. average reference (bads excluded) for ICA / ICLabel
    raw.set_eeg_reference("average", projection=False)
    raw_ica.set_eeg_reference("average", projection=False)

    # 5. ICA (fit once, reused unless --refit-ica or bad channels changed)
    ica_f = out / f"{ses}-ica.fif"
    ica = None
    if ica_f.exists() and not refit_ica:
        ica = mne.preprocessing.read_ica(ica_f)
        if sorted(ica.ch_names) != sorted([c for c in raw_ica.copy().pick("eeg", exclude="bads").ch_names]):
            ica = None                                   # channel set changed -> refit
    if ica is None:
        rank = mne.compute_rank(raw_ica, rank=None, tol=1e-4, tol_kind="relative")["eeg"]
        n_comp = int(min(rank - 1, C.ICA_MAX_COMPONENTS))
        ica = mne.preprocessing.ICA(n_components=n_comp, method=C.ICA_METHOD, random_state=C.ICA_SEED,
                                    fit_params=dict(extended=True, ortho=False), max_iter="auto")
        ica.fit(raw_ica, picks="eeg", reject_by_annotation=True)
    probs = iclabel_probs(raw_ica, ica)
    eog_idx, eog_scores = ica.find_bads_eog(raw_ica, ch_name=list(C.EOG.values()), threshold=C.EOG_CORR_Z,
                                            measure="zscore")
    eog_ok = [int(i) for i in eog_idx if probs.brain[int(i)] < C.ICLABEL_BRAIN_PROTECT]
    auto = sorted(set(np.flatnonzero(probs.eye >= C.ICLABEL_EYE_THRESH).tolist()) | set(eog_ok))
    exclude = sorted(ovr["exclude"]) if "exclude" in ovr else auto
    ica.exclude = exclude
    ica.save(ica_f, overwrite=True)
    qc["ica"] = dict(n_components=ica.n_components_, auto_excluded=auto, excluded=exclude,
                     override=("exclude" in ovr), override_note=ovr.get("note"),
                     iclabel=probs.round(3).to_dict(orient="index"),
                     eog_corr_flagged=list(map(int, eog_idx)))
    print(f"  ICA: {ica.n_components_} comps, removing {exclude} "
          f"({'override' if 'exclude' in ovr else 'auto'})")
    ica.apply(raw)

    # 6. bad channels, stage (b): poorly correlated with neighbours in 1-15 Hz (after ICA, so blinks don't count)
    if not manual_bads:
        chk = raw.copy().filter(*C.BAD_CORR_BAND, picks="eeg", phase="zero", fir_design="firwin")
        new = detect_uncorrelated(chk, raw.info["bads"], max(0, C.MAX_BAD_CHANNELS - len(bads)))
        del chk
        bads.update(new)
        raw.info["bads"] = list(bads)
        if new:
            print(f"  bad channels after ICA: {list(new)}")

    # 7. interpolate, re-reference to mastoids, epoch
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference(list(C.MASTOIDS.values()), projection=False)
    raw.drop_channels(list(C.MASTOIDS.values()))
    meta = meta.copy()
    meta.insert(0, "session", ses)
    for k in ("subject", "group", "stimulant", "caffeine", "day"):
        meta[k] = getattr(ses_row, k)
    epochs = make_epochs(raw, meta)

    # 7b. bad channels, stage (c): iteratively interpolate the channel that is the worst offender in more than
    #     EPOCH_BAD_FRAC of all epochs (ignoring channels already interpolated, which mirror their neighbours).
    #     A candidate whose nearest neighbours are also frequent offenders is a regional artifact, not a bad
    #     electrode -> not interpolated (those trials are rejected instead).
    regional = {}
    if not manual_bads:
        ep_pos = {ch["ch_name"]: ch["loc"][:3] for ch in epochs.info["chs"]}
        while len(bads) < C.MAX_BAD_CHANNELS:
            ex, chs, peak = exceed_matrix(epochs)
            interp = np.isin(chs, list(bads))
            counts = np.zeros(len(chs))
            for e in np.flatnonzero(ex.any(axis=1)):
                amp = peak[e].copy()
                amp[interp] = -np.inf
                counts[int(np.argmax(amp))] += 1
            frac = counts / len(epochs)
            P = np.array([ep_pos[c] for c in chs])
            pick = None
            for i in np.argsort(-frac):
                if frac[i] <= C.EPOCH_BAD_FRAC:
                    break
                nb = np.argsort(np.linalg.norm(P - P[i], axis=1))[1:7]
                nb = [j for j in nb if not interp[j]]
                if sum(frac[j] > C.EPOCH_BAD_FRAC for j in nb) >= C.REGIONAL_MIN_NEIGHBOURS:
                    regional[str(chs[i])] = f"{frac[i]:.0%} (neighbours also high: " + \
                        ",".join(f"{chs[j]} {frac[j]:.0%}" for j in nb if frac[j] > C.EPOCH_BAD_FRAC) + ")"
                    continue
                pick = i
                break
            if pick is None:
                break
            bads[str(chs[pick])] = f"worst channel in {frac[pick]:.0%} of epochs over threshold [post-epoch]"
            raw.info["bads"] = [str(chs[pick])]
            raw.interpolate_bads(reset_bads=True)
            epochs = make_epochs(raw, meta)
            print(f"  bad channel from epochs: {chs[pick]} ({frac[pick]:.0%} of epochs)")
    qc["regional_artifact_not_interpolated"] = regional
    if regional:
        print(f"  regional artifact (not interpolated): {list(regional)}")
    qc["bad_channels"] = bads
    qc["warnings"] = [f"{len(bads)} bad channels (> {C.MAX_BAD_CHANNELS}) - review"] if len(bads) > C.MAX_BAD_CHANNELS else []
    print(f"  bad channels total ({len(bads)}): {list(bads) or 'none'}")

    # 8. absolute-threshold rejection on scalp channels
    n_after_annot = len(epochs)
    X = epochs.copy().pick("eeg").crop(*C.REJECT_WINDOW).get_data() * 1e6
    bad_ep = np.flatnonzero(np.abs(X).max(axis=(1, 2)) > C.REJECT_UV)
    worst = [epochs.copy().pick("eeg").ch_names[i] for i in np.abs(X[bad_ep]).max(2).argmax(1)] if len(bad_ep) else []
    epochs.drop(bad_ep, reason=f"THRESH_{C.REJECT_UV:g}uV")
    epochs.save(epo_f, overwrite=True)

    # 9. trial table + QC
    all_tr = meta.drop(columns=["sample"]).copy()
    all_tr["kept"] = False
    all_tr.loc[epochs.selection, "kept"] = True
    all_tr.to_csv(out / f"{ses}_trials.csv", index=False)
    kept = all_tr[all_tr.kept]
    per_cond = kept.groupby(["attend", "interrupter"]).size() if len(kept) else pd.Series(dtype=int)
    qc["trials"] = dict(behavior_total=int(sum(l.get("beh_visual_trials", 0) for l in logs)),
                        eeg_aligned=len(meta), rejected_bad_segment=int(len(meta) - n_after_annot),
                        rejected_threshold=int(len(bad_ep)), kept=len(kept),
                        rejected_worst_channel={k: int(v) for k, v in pd.Series(worst, dtype=str).value_counts().head(5).items()},
                        kept_per_condition={f"{a}/{i}": int(v) for (a, i), v in per_cond.items()})
    (out / f"{ses}_qc.json").write_text(json.dumps(qc, indent=2, default=str))
    print(f"  epochs kept: {len(kept)}/{len(meta)} (threshold drops {len(bad_ep)})")

    # 10. review report
    try:
        rep = mne.Report(title=f"{ses} preprocessing")
        rep.add_html(f"<pre>{json.dumps({k: qc[k] for k in ('bad_channels', 'trials')}, indent=2, default=str)}</pre>",
                     title="Summary")
        rep.add_ica(ica, title="ICA (excluded components marked)", inst=raw_ica,
                    picks=list(range(ica.n_components_)), n_jobs=1)
        rep.add_html(probs.round(2).to_html(), title="ICLabel probabilities")
        po = [c for c in ("PO7", "PO3", "O1", "PO8", "PO4", "O2") if c in epochs.ch_names]
        evk = [epochs[a].average(picks=po) for a in ("focal", "broad")]
        for e, a in zip(evk, ("focal", "broad")):
            e.comment = a
        fig = mne.viz.plot_compare_evokeds(dict(zip(("focal", "broad"), evk)), picks=po, combine="mean",
                                           show=False, title=f"{ses} PO cluster")[0]
        rep.add_figure(fig, title="ERP focal vs broad (PO7/PO3/O1/PO8/PO4/O2)")
        rep.save(out / f"{ses}_report.html", overwrite=True, open_browser=False)
    except Exception as e:                                # report is a convenience; never fail the session on it
        print(f"  (report skipped: {e})")
    return qc


# ---------------------------------------------------------------- driver
def write_summary():
    rows = []
    for f in sorted(Path(C.OUT_DIR).glob("*/*_qc.json")):
        q = json.loads(f.read_text())
        r = {k: q[k] for k in ("session", "subject", "group", "stimulant", "caffeine", "day", "n_visits")}
        r.update(bad_channels=";".join(q["bad_channels"]), n_bad=len(q["bad_channels"]),
                 gross_artifact_s=q.get("gross_artifact", {}).get("seconds"),
                 warnings="; ".join(q.get("warnings", [])),
                 ics_removed=len(q["ica"]["excluded"]), ica_override=q["ica"]["override"],
                 trials_behavior=q["trials"]["behavior_total"], trials_eeg=q["trials"]["eeg_aligned"],
                 trials_rejected=q["trials"]["rejected_threshold"], trials_kept=q["trials"]["kept"])
        r.update({f"kept_{k}": v for k, v in q["trials"]["kept_per_condition"].items()})
        rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(Path(C.OUT_DIR) / "qc_summary.csv", index=False)
        print(f"wrote {Path(C.OUT_DIR) / 'qc_summary.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--refit-ica", action="store_true")
    a = ap.parse_args()
    S = T.find_sessions()
    if a.list:
        print(S[["session", "group", "stimulant", "caffeine", "day", "n_visits", "date"]].to_string())
        return
    todo = S[S.group != "excluded"] if a.all else S[S.session.isin(a.sessions)]
    if not len(todo):
        sys.exit("no sessions selected (give session names or --all; --list to see them)")
    failed = []
    for _, row in todo.iterrows():
        print(f"== {row.session}")
        try:
            process(row, overwrite=a.overwrite, refit_ica=a.refit_ica)
        except Exception:
            traceback.print_exc()
            failed.append(row.session)
    write_summary()
    if failed:
        print("FAILED:", failed)


if __name__ == "__main__":
    main()
