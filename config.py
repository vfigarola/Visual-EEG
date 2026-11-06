"""Settings for the Jasmine visual-EEG preprocessing pipeline.
Edit paths / parameters here; every run writes the values it used into each session's QC json."""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(os.environ.get("JASMINE_ROOT", Path.home() / "Documents/1_CMU/Barb_Lab/Projects/Jasmine-data"))
RAW_DIR = ROOT / "RawData"                  # searched recursively for *.bdf
BEH_DIR = ROOT / "RawBehavior"              # *_RESULTS.mat
LOCS_FILE = ROOT / "chanlocs_64_3_eye_chan.locs"
OUT_DIR = ROOT / "derivatives" / "preproc_v1"
ICA_OVERRIDES = Path(__file__).with_name("ica_overrides.json")

# ---------------------------------------------------------------- sessions
EXCLUDE_SESSIONS = {"JKSA200_XC"}           # group unknown (ADHD vs NT) -> left out for now
# excluded from post-processing analyses (>= 50% of EEG trials rejected in preprocessing, Nov 6 2026)
ANALYSIS_EXCLUDE = {"JKSA025_MX": "46% trials kept", "JKSA093_XX": "43% trials kept"}
# BDF files whose names don't follow JKSA###_@@_#.bdf
BDF_RENAMES = {"JKSA032_1": "JKSA032_MX_1", "JKSA032_2": "JKSA032_MX_2"}
NEUROTYPICAL = {"018", "019", "021", "022", "023", "039", "051", "053", "055", "057",
                "061", "063", "064", "067", "070", "075", "076", "080", "101"}

# ---------------------------------------------------------------- channels
N_SCALP = 64
MASTOIDS = {"EXG1": "M1", "EXG2": "M2"}     # final reference = mean(M1, M2)
EOG = {"EXG3": "VEOG", "EXG4": "HEOGl", "EXG6": "HEOGr"}   # EXG5/7/8 dropped
# approximate mastoid positions in EEGLAB polar coords (theta deg, radius) for interpolation / ICLabel
MASTOID_LOCS = {"M1": (-115.0, 0.64), "M2": (115.0, 0.64)}

# ---------------------------------------------------------------- triggers (low byte of the Status channel)
TRIG = dict(trial_start=99, stim=1, trial_end=10, aud_marker=9, responses=(24, 25), block=255)
CUE_CODES = {2: ("focal", "none"), 3: ("focal", "early"), 4: ("focal", "late"),
             5: ("broad", "none"), 6: ("broad", "early"), 7: ("broad", "late")}
CROP_PAD_S = 10.0         # keep this much data before the first / after the last visual trial

# ---------------------------------------------------------------- signal processing
SFREQ = 256.0
L_FREQ, H_FREQ = 0.5, 40.0      # analysis data (zero-phase FIR)
ICA_L_FREQ = 1.0                # ICA is fit on a 1-40 Hz copy, then applied to the 0.5-40 Hz data
LINE_FREQ = 60.0                # US mains; above H_FREQ so no notch needed

# gross-artifact segments (whole-head junk, e.g. first minutes of JKSA073_XX_1): per 1-s window, the
# median-across-channels SD (median-referenced, 1-40 Hz) above max(GROSS_UV, GROSS_FACTOR x session median)
GROSS_UV = 50.0
GROSS_FACTOR = 4.0
GROSS_PAD_S = 1.0

# bad-channel detection (scalp + mastoids) in three stages
# (a) before ICA, 1-40 Hz copy: flat or very high variance only (blinks make a correlation test unreliable here)
BAD_FLAT_UV = 0.5               # channel std below this (uV) -> flat
BAD_VAR_Z = 4.0                 # robust z of log-variance above this -> noisy
# (b) after ICA: correlation with the mean of the 6 nearest good neighbours (avg ref, 1-s windows, median
#     over windows); iterative - the worst channel is removed and everything is recomputed
BAD_NEIGHBOR_R = 0.3
BAD_CORR_BAND = (1.0, 15.0)     # low band: catches bad contacts, not temporalis muscle (which is >20 Hz)
# (c) after epoching: iteratively interpolate the channel that is the worst offender (largest peak among
#     not-yet-interpolated channels) in > EPOCH_BAD_FRAC of all epochs that exceed REJECT_UV
EPOCH_BAD_FRAC = 0.05
REGIONAL_MIN_NEIGHBOURS = 3     # candidate not interpolated if >= this many of its 6 nearest neighbours are also high
MAX_BAD_CHANNELS = 8            # stages (b)+(c) stop adding at this total; more than this -> QC warning

# ICA
ICA_METHOD = "picard"           # extended-infomax equivalent; needs `python-picard`
ICA_SEED = 97
ICA_MAX_COMPONENTS = 40
ICLABEL_EYE_THRESH = 0.70       # exclude if ICLabel 'eye blink' probability >= this
EOG_CORR_Z = 4.0                # ...or if |z| of correlation with an EOG channel >= this
ICLABEL_BRAIN_PROTECT = 0.50    # ...unless ICLabel brain prob >= this (EOG channels carry EEG too; correlation
                                #    alone once removed components ICLabel rated 100% brain). Requiring ICLabel eye
                                #    >= 0.3 instead was too strict: it kept eye artifacts ICLabel calls 'other'/'muscle'.

# epochs (time 0 = trigger 1 = syllable stream onset; cue at ~ -1.0 s, trial start at ~ -2.0 s)
TMIN, TMAX = -2.5, 4.5
BASELINE = (-1.5, -1.0)         # after trial start, before cue
REJECT_UV = 100.0               # absolute threshold on scalp channels after baseline
REJECT_WINDOW = (-1.5, 4.5)     # window in which the threshold is checked
