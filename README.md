# Jasmine visual-EEG preprocessing (Python / MNE)

Reprocesses every visual session from the raw BioSemi BDFs. JKSA200_XC is excluded (group unknown).
Sessions with missing trials (JKSA093_MC: 195 usable visual trials, JKSA029_XX: 192) are included; their counts are in the QC files.

## Setup (once, on your Mac)
```bash
cd ~/Documents/1_CMU/Barb_Lab/Projects/Jasmine-data/python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
source .venv/bin/activate
python preprocess.py --list            # 58 sessions found (JKSA200 marked excluded)
python preprocess.py JKSA019_XX        # try one first (~3-5 min)
python preprocess.py --all             # everything; skips sessions already done
```
Output goes to `Jasmine-data/derivatives/preproc_v1/<session>/`, plus `qc_summary.csv` (one row per session).

## Review step (ICA)
1. Open each `<session>_report.html`. It shows every ICA component (topography, spectrum, time course), the ICLabel probabilities, which components were removed, and the focal-vs-broad ERP at the parieto-occipital cluster.
2. If a session needs a change, add it to `ica_overrides.json` and rerun with `--overwrite`. The saved ICA is reused, so the decomposition is identical and only the removal changes:
```json
{
  "JKSA073_XX": {"exclude": [0, 1, 4], "note": "IC4 is a saccade ICLabel missed"},
  "JKSA019_XX": {"bads": ["PO4"], "note": "intermittent PO4"}
}
```
`exclude` replaces the automatic list. `bads` replaces all automatic bad-channel steps for that session (changing bad channels refits the ICA).

## What the pipeline does (per session) — usable as a methods draft
1. **Triggers**: read from the low byte of the BioSemi Status channel. The upper byte flickers during these recordings and, read naively, duplicates trigger events (the "extra" cue triggers in the old MATLAB scripts).
2. **Visual trials only**: in each BDF half, blocks 1–5 are visual and 6–10 auditory. Auditory trials are identified by code 9 / response codes 24–25. Each EEG visual trial is matched to the behavioral RESULTS file by its cue-code sequence. Trial metadata (attend focal/broad, interrupter none/early/late, accuracy, RT, block) comes from the RESULTS file. The data is cropped to the visual blocks ±10 s.
3. **Resampling** to 256 Hz with a zero-phase polyphase anti-alias filter. This also handles the four sessions recorded at 4096 Hz (018_XC, 022_XX, 030_MX, 031_XX). The two halves are concatenated, with a boundary marker that filters and epochs respect.
4. **Filtering**: 0.5–40 Hz zero-phase FIR (MNE `firwin`). A 1–40 Hz copy is used for artifact detection and ICA.
5. **Gross-artifact segments**: 1-s windows whose median channel SD exceeds max(50 µV, 4 × the session median) are marked `BAD_gross` and excluded from everything downstream (e.g., the first minutes of JKSA073_XX_1).
6. **ICA**: Components are manually removed. 
7. **Epochs**: −2.5 to 4.5 s around trigger 1 (syllable-stream onset). The cue is at ≈ −1.0 s and trial start at ≈ −2.0 s. Baseline is −1.5 to −1.0 s (before the cue).
8. **Bad channels from epochs**: each epoch that exceeds ±100 µV is attributed to its worst channel (largest peak, ignoring channels already interpolated, since those mirror their neighbours). If one channel is responsible for more than 5% of all epochs, and fewer than 3 of its 6 nearest neighbors are also above 5%, it is interpolated and the epochs are rebuilt; this repeats. When 3 or more neighbors are also high, it's a regional artifact (e.g., frontal), not a bad electrode, so those trials are rejected instead (logged as `regional_artifact_not_interpolated`). 
9. **Rejection**: drop epochs where any scalp channel exceeds ±100 µV between −1.5 and 4.5 s, or that overlap a `BAD_gross` segment.

## Files
| file | what |
|---|---|
| `config.py` | all paths and parameters (every run saves them into the QC json) |
| `bdf_io.py` | BDF header / trigger / channel reader (numpy; handles truncated headers and `1-A1`-style labels) |
| `trials.py` | session discovery, RESULTS loading, EEG↔behavior alignment |
| `preprocess.py` | the pipeline |
| `session_manifest.csv` | per-session inventory from the data audit |
| `matv5.py`, `bdftrig.py` | numpy-only helpers used for the audit |

## Known caveats
- EOG channel mapping assumes EXG3 = VEOG, EXG4 = HEOGl, EXG6 = HEOGr (from the old scripts and `chanlocs_64_3_eye_chan.locs`). JKSA110's header says "4 EOGs", so its layout may differ; check its report.
- Mastoid positions for interpolation / ICLabel are approximate (EEGLAB polar ±115°, radius 0.64). If a mastoid is flagged bad, the mastoid reference partly relies on its interpolation; such sessions show `M1`/`M2` in `bad_channels`, so check them.
- The low-pass at 40 Hz is below ICLabel's training band (1–100 Hz). That's fine for eye components, but ICLabel will print a warning.
