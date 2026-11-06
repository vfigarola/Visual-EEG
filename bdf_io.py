"""BioSemi .bdf reading without relying on the header's record count.

* Triggers come from the LOW BYTE of the Status channel only. The upper byte flickers
  during these recordings; EEGLAB + mod(type,256) turns each flicker into a duplicate
  event (the "extra" cue triggers seen in visual_epoching.m).
* The number of records is computed from the file size, so crashed recordings whose
  header says nrec = -1 (JKSA093_MC_2, JKSA105_XX_1) load normally.
"""
from dataclasses import dataclass
import os
import numpy as np


@dataclass
class BDFHeader:
    path: str
    labels: list
    n_samp: list          # samples per record, per channel
    rec_dur: float
    n_rec: int            # from file size
    n_rec_header: int
    hdr_bytes: int
    gain: np.ndarray      # uV per bit, per channel
    offset: np.ndarray    # uV, per channel
    patient_id: str
    start: str

    @property
    def sfreq(self):
        return self.n_samp[0] / self.rec_dur

    @property
    def rec_bytes(self):
        return sum(self.n_samp) * 3


def read_header(path):
    with open(path, "rb") as fh:
        h = fh.read(256)
        nch = int(h[252:256])
        H = h + fh.read(256 * nch)

    def field(width, start):
        return [H[256 + start * nch + i * width: 256 + start * nch + (i + 1) * width].decode("latin1").strip()
                for i in range(nch)]
    labels = [l.split("-")[-1] for l in field(16, 0)]   # some files label channels '1-A1', '1-EXG1'
    physmin = np.array(field(8, 104), float)
    physmax = np.array(field(8, 112), float)
    digmin = np.array(field(8, 120), float)
    digmax = np.array(field(8, 128), float)
    n_samp = [int(x) for x in field(8, 216)]
    gain = (physmax - physmin) / (digmax - digmin)
    offset = physmin - digmin * gain
    hdr_bytes = 256 * (nch + 1)
    rec_bytes = sum(n_samp) * 3
    n_rec = (os.path.getsize(path) - hdr_bytes) // rec_bytes
    return BDFHeader(path, labels, n_samp, float(h[244:252]), int(n_rec), int(h[236:244]), hdr_bytes,
                     gain, offset, h[8:88].decode("latin1").strip(), h[168:184].decode("latin1"))


def _memmap(hd):
    return np.memmap(hd.path, dtype=np.uint8, mode="r", offset=hd.hdr_bytes, shape=(hd.n_rec, hd.rec_bytes))


def _chan_slice(hd, idx):
    start = sum(hd.n_samp[:idx]) * 3
    return start, start + hd.n_samp[idx] * 3


def read_triggers(hd):
    """Return (times_s, codes) for every low-byte change to a non-zero value."""
    mm = _memmap(hd)
    a, b = _chan_slice(hd, hd.labels.index("Status"))
    low = np.asarray(mm[:, a:b]).reshape(-1, 3)[:, 0].astype(np.int32)
    on = np.flatnonzero(np.diff(low, prepend=0) != 0)
    codes = low[on]
    keep = codes != 0
    return on[keep] / hd.sfreq, codes[keep]


def read_channels(hd, names, t0, t1):
    """Read channels (in uV, float64) for the time window [t0, t1) seconds. Returns (data, first_sample)."""
    fs = hd.sfreq
    s0, s1 = max(0, int(np.floor(t0 * fs))), min(hd.n_rec * hd.n_samp[0], int(np.ceil(t1 * fs)))
    r0, r1 = s0 // hd.n_samp[0], (s1 - 1) // hd.n_samp[0] + 1
    mm = _memmap(hd)
    out = np.empty((len(names), s1 - s0))
    for k, name in enumerate(names):
        i = hd.labels.index(name)
        a, b = _chan_slice(hd, i)
        raw = np.asarray(mm[r0:r1, a:b]).reshape(-1, 3).astype(np.int32)
        v = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)          # 24-bit two's complement
        v = v[s0 - r0 * hd.n_samp[0]: s1 - r0 * hd.n_samp[0]]
        out[k] = v * hd.gain[i] + hd.offset[i]
    return out, s0
