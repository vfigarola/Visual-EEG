"""Session discovery, behavioral RESULTS loading, and EEG-trigger <-> behavior alignment."""
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from bdf_io import read_header, read_triggers

try:
    from scipy.io import loadmat as _sp_loadmat

    def _load_results(path):
        r = _sp_loadmat(path, squeeze_me=False, struct_as_record=False)["results"][0, 0]
        return {k: getattr(r, k) for k in r._fieldnames}
except ImportError:                                   # numpy-only fallback (no scipy)
    import matv5

    def _load_results(path):
        r = matv5.loadmat(str(path))["results"]
        for k, v in r.items():
            if isinstance(v, dict) and "__cell__" in v:
                a = np.empty(len(v["__cell__"]), dtype=object)
                a[:] = [np.asarray(x if x is not None else "") for x in v["__cell__"]]
                r[k] = a
        return r


# ---------------------------------------------------------------- sessions
def find_sessions():
    """DataFrame: one row per session (subject + condition label), with both BDF halves and RESULTS files."""
    rows = {}
    for bdf in sorted(Path(C.RAW_DIR).rglob("*.bdf")):
        stem = C.BDF_RENAMES.get(bdf.stem, bdf.stem)
        m = re.fullmatch(r"(JKSA\d{3})_([XM][XC])_([12])", stem)
        if not m:
            print(f"  [skip] unrecognised BDF name: {bdf}")
            continue
        subj, cond, half = m.groups()
        ses = f"{subj}_{cond}"
        d = rows.setdefault(ses, dict(session=ses, subject=subj, cond=cond))
        d[f"bdf{half}"] = str(bdf)
        beh = sorted(Path(C.BEH_DIR).glob(f"{ses}_{half}_*_RESULTS.mat"))
        d[f"beh{half}"] = str(beh[0]) if beh else None
        if beh:
            d["date"] = re.search(r"_(\d{4}-\d\d-\d\d)_", beh[0].name).group(1)
    df = pd.DataFrame(rows.values())
    df["group"] = np.where(df.subject.str[4:].isin(C.NEUROTYPICAL), "NT", "ADHD")
    df.loc[df.session.isin(C.EXCLUDE_SESSIONS), "group"] = "excluded"
    df["stimulant"] = np.where(df.cond.str[0] == "M", "on", "off")
    df["caffeine"] = np.where(df.cond.str[1] == "C", "yes", "no")
    df["n_visits"] = df.groupby("subject").session.transform("count")
    df["day"] = df.groupby("subject").date.rank(method="first").astype(int)
    return df.sort_values("session").reset_index(drop=True)


# ---------------------------------------------------------------- EEG trials
def eeg_visual_trials(bdf_path):
    """Visual trials found in one BDF half: list of dicts with cue code and times (s from file start)."""
    hd = read_header(bdf_path)
    t, c = read_triggers(hd)
    T = C.TRIG
    i99 = list(np.flatnonzero(c == T["trial_start"])) + [len(c)]
    out = []
    for k in range(len(i99) - 1):
        seg, ts = c[i99[k]:i99[k + 1]], t[i99[k]:i99[k + 1]]
        if np.any(seg == T["aud_marker"]) or np.any(np.isin(seg, T["responses"])):
            continue                                          # auditory trial
        seg_l = list(seg)
        cue = next((int(x) for x in seg if x in C.CUE_CODES), 0)
        out.append(dict(cue=cue, t_start=float(ts[0]),
                        t_cue=float(ts[seg_l.index(cue)]) if cue else np.nan,
                        t_stim=float(ts[seg_l.index(T["stim"])]) if T["stim"] in seg_l else np.nan,
                        t_end=float(ts[seg_l.index(T["trial_end"])]) if T["trial_end"] in seg_l else np.nan))
    return out, hd


# ---------------------------------------------------------------- behavior
def behavior_trials(results_path, half):
    """Visual trials from one RESULTS file (columns 1-5 of each matrix are the visual blocks)."""
    r = _load_results(results_path)
    stop = np.atleast_1d(np.squeeze(r["stopTime"]))
    n_blocks_done = sum(1 for s in stop if np.size(s) and str(np.squeeze(s)).strip())
    n_vis = min(5, n_blocks_done)
    cue = np.asarray(r["cuetrigger"])
    answ, resp = np.asarray(r["answ"]), np.asarray(r["resp"])
    rows = []
    for b in range(n_vis):
        for i in range(cue.shape[0]):
            hits = answ[i, :, b] == resp[i, :, b]
            att, intr = C.CUE_CODES[int(cue[i, b])]
            rows.append(dict(half=half, block=b + 1 + 5 * (half - 1), trial_in_block=i + 1,
                             cue=int(cue[i, b]), attend=att, interrupter=intr,
                             attend_code=int(np.asarray(r["attend"])[i, b]),
                             delayS=float(np.asarray(r["delayS"])[i, b]),
                             rhythm=int(np.asarray(r["rhythm"])[i, b]),
                             n_syll_correct=int(hits.sum()), correct=bool(hits.all()),
                             rt=float(np.asarray(r["rxnTime"])[i, b])))
    return pd.DataFrame(rows)


def align_half(bdf_path, results_path, half):
    """Match EEG visual trials to behavioral trials by cue sequence. Returns (metadata_df, log_dict)."""
    eeg, hd = eeg_visual_trials(bdf_path)
    beh = behavior_trials(results_path, half)
    sm = difflib.SequenceMatcher(a=beh.cue.tolist(), b=[e["cue"] for e in eeg], autojunk=False)
    pairs = [(blk.a + k, blk.b + k) for blk in sm.get_matching_blocks() for k in range(blk.size)]
    n_match = len(pairs)
    if n_match < 0.9 * min(len(beh), len(eeg)):
        raise RuntimeError(f"{bdf_path}: only {n_match} of {len(beh)} behavioral / {len(eeg)} EEG trials align")
    rows = []
    for ia, ib in pairs:
        e = eeg[ib]
        rows.append({**beh.iloc[ia].to_dict(), **{k: e[k] for k in ("t_start", "t_cue", "t_stim", "t_end")}})
    meta = pd.DataFrame(rows)
    usable = meta.t_stim.notna() & (meta.t_stim + C.TMAX < hd.n_rec * hd.rec_dur)
    log = dict(file=Path(bdf_path).name, sfreq=hd.sfreq, n_rec=hd.n_rec, n_rec_header=hd.n_rec_header,
               header_id=hd.patient_id, beh_visual_trials=len(beh), eeg_visual_trials=len(eeg),
               matched=n_match, unusable_no_stim_or_cut=int((~usable).sum()),
               behavior_trials_missing_in_eeg=len(beh) - n_match, eeg_trials_unmatched=len(eeg) - n_match)
    return meta[usable].reset_index(drop=True), log, hd
