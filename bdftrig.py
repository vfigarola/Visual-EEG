"""Fast BioSemi .bdf trigger reader (numpy only).

Reads just the Status channel and returns trigger onsets using the LOW BYTE only.
The upper byte of the BioSemi status word flickers during the session; EEGLAB's
pop_biosig + mod(type,256) turns each flicker into a duplicate event with the same
code. That is where the "extra" cue triggers come from in visual_epoching.m.
Using low-byte transitions removes them without any manual deletion.

Works on truncated recordings (header nrec = -1) by inferring record count from file size.
"""
import os
import numpy as np


def read_triggers(path):
    with open(path, 'rb') as fh:
        h = fh.read(256)
    nch = int(h[252:256])
    hdr = 256 * (nch + 1)
    with open(path, 'rb') as fh:
        H = fh.read(hdr)
    ns = [int(H[256 + nch * 216 + i * 8: 256 + nch * 216 + (i + 1) * 8]) for i in range(nch)]
    recb = sum(ns) * 3
    nrec = (os.path.getsize(path) - hdr) // recb
    mm = np.memmap(path, dtype=np.uint8, mode='r', offset=hdr, shape=(nrec, recb))
    st = sum(ns[:-1]) * 3
    raw = np.asarray(mm[:, st:st + ns[-1] * 3]).reshape(-1, 3)
    val = raw[:, 0].astype(np.int32)          # low byte = trigger code
    fs = ns[-1]
    on = np.flatnonzero(np.diff(val, prepend=0) != 0)
    codes = val[on]
    keep = codes != 0
    return on[keep] / fs, codes[keep], fs, nrec, int(H[236:244])


def visual_trials(path):
    """Split a file into trials (99 -> next 99). A trial is auditory if it has code 9 or a 24/25 response.
    Returns list of dicts for VISUAL trials: cue (2-7), t99, t_cue, t1 (stim onset), t10 (end)."""
    t, c, fs, *_ = read_triggers(path)
    i99 = list(np.flatnonzero(c == 99)) + [len(c)]
    out = []
    for k in range(len(i99) - 1):
        seg, ts = c[i99[k]:i99[k + 1]], t[i99[k]:i99[k + 1]]
        if np.any(seg == 9) or np.any((seg == 24) | (seg == 25)):
            continue
        g = lambda code: float(ts[list(seg).index(code)]) if code in seg else None
        cue = next((int(x) for x in seg if 2 <= x <= 7), 0)
        out.append(dict(cue=cue, t99=float(ts[0]), t_cue=g(cue) if cue else None, t1=g(1), t10=g(10)))
    return out
