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
6. **Bad channels, before ICA** (64 scalp + 2 mastoids): flat (SD < 0.5 µV) or very high variance (robust z > 4). A neighbour-correlation test is not used here because blinks make frontal channels look uncorrelated.
7. **ICA**: Picard (extended-infomax equivalent) on the average-referenced 1–40 Hz copy, excluding bad channels. Components are removed if ICLabel's eye probability is ≥ 0.70, or if their EOG correlation has |z| ≥ 4 (VEOG, HEOGl, HEOGr) and ICLabel does not classify them as brain (brain probability < 0.50). The brain check protects components ICLabel rates as brain, because the EOG channels also pick up EEG. A stricter rule (requiring ICLabel eye ≥ 0.30) missed eye artifacts that ICLabel labels 'other' or 'muscle'. Removal is then applied to the 0.5–40 Hz data.
8. **Bad channels, after ICA**: correlation of each channel with the mean of its 6 nearest good neighbours in the **1–15 Hz** band (average reference, 1-s windows, median over windows). Run iteratively (flag the worst, recompute) until all r ≥ 0.3. The mastoids are excluded from this test: they sit at the edge with no electrodes below, so it falsely flags them. They are still checked for flat or noisy signal before ICA. The low band catches bad contacts but not temporalis muscle, which lives above 20 Hz and doesn't affect ERP/ITC. Flagged channels are interpolated (spherical spline), then data are **re-referenced to linked mastoids** (EXG1/EXG2, as in the original analysis).
9. **Epochs**: −2.5 to 4.5 s around trigger 1 (syllable-stream onset). The cue is at ≈ −1.0 s and trial start at ≈ −2.0 s. Baseline is −1.5 to −1.0 s (before the cue).
10. **Bad channels from epochs**: each epoch that exceeds ±100 µV is attributed to its worst channel (largest peak, ignoring channels already interpolated, since those mirror their neighbours). If one channel is responsible for more than 5% of all epochs, and fewer than 3 of its 6 nearest neighbours are also above 5%, it is interpolated and the epochs are rebuilt; this repeats. When 3 or more neighbours are also high, it's a regional artifact (e.g., frontal), not a bad electrode, so those trials are rejected instead (logged as `regional_artifact_not_interpolated`). At most 8 bad channels in total come from the automatic steps; more than that is flagged as a QC warning.
11. **Rejection**: drop epochs where any scalp channel exceeds ±100 µV between −1.5 and 4.5 s, or that overlap a `BAD_gross` segment.

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
