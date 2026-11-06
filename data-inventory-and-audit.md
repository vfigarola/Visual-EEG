# Jasmine visual EEG: data inventory & audit (Nov 2026)

Source folder: `~/Documents/1_CMU/Barb_Lab/Projects/Jasmine-data`
Per-session table: `Jasmine-data/python/session_manifest.csv` (58 sessions: group, stim/caffeine, date, day 1/2, sampling rate, visual trials in EEG vs behavior, % correct overall/focal/broad, issues).
Preprocessing QC: `Jasmine-data/derivatives/preproc_v1/qc_summary.csv` (final run Nov 6, 08:41–09:34).

## Decisions
- (Nov 5) **JKSA200_XC is excluded** for now (group unknown) → 57 sessions preprocessed: 19 NT, 25 ADHD subjects (38 sessions).
- (Nov 5) **Reprocess everything from raw BDFs.** Jasmine says her 2018 preprocessing was wrong; the `VSA*_DATA.mat` files are not used.
- (Nov 5) **Sessions with missing recorded trials stay in** and are reported in the paper: JKSA093_MC (195 usable visual trials) and JKSA029_XX (192).
- (Nov 5) **Pipeline**: Python + MNE; analysis band 0.5–40 Hz (zero-phase); ICA removes **eye components only** (Victoria declined muscle removal). See `Jasmine-data/python/README.md`.
- (Nov 6) **Exclusion rule: a session with ≥ 50% of its trials rejected is excluded from post-processing analysis.** Final run excludes **JKSA025_MX** (ADHD, on stimulant, single visit; 110/240 kept) and **JKSA093_XX** (ADHD, off stimulant, visit 2; 103/240 kept). That leaves **55 sessions**. JKSA093 keeps its day-1 session (093_MC) but loses its on/off pair.
- Borderline sessions that are included: **JKSA018_XC** (NT; 122/240 = 51%; min 14 trials in one condition) and **JKSA034_XC** (min 19).

## Final preprocessing results (Nov 6 run, 55 included sessions)
- Median trials kept: ADHD 213/240, NT 211/240 (~88–90%). Median per condition: 33–37.
- Median bad channels: 1 (ADHD), 3 (NT). Median ICA components removed: 3.
- No removed ICA component has ICLabel brain ≥ 0.5. One mastoid flagged (JKSA032_XX M2, noisy signal before ICA).

## Pipeline v1 (summary; full methods text in the README)
1. Low-byte triggers.
2. Visual blocks only, each trial matched to RESULTS by cue sequence.
3. Resample to 256 Hz (zero-phase polyphase) and concatenate the halves.
4. 0.5–40 Hz FIR (1–40 Hz copy for detection and ICA).
5. `BAD_gross` segments: 1-s windows with median channel SD > max(50 µV, 4× the session median).
6. Bad channels, three stages:
   - (a) before ICA: flat or robust variance z > 4 (mastoids included).
   - (b) after ICA: iterative correlation with the 6-neighbour mean in **1–15 Hz**, r < 0.3. Mastoids excluded.
   - (c) after epoching: iteratively interpolate the channel that is the worst offender in > 5% of epochs, unless ≥ 3 of its 6 nearest neighbours are also > 5% (that counts as a regional artifact, so the trials are rejected instead).
   - Cap of 8 bad channels.
7. Picard ICA on the average-referenced 1–40 Hz copy. Remove a component if ICLabel eye ≥ 0.70, or if EOG correlation |z| ≥ 4 **and** ICLabel brain < 0.50. Overrides go in `ica_overrides.json`.
8. Interpolate bad channels, then re-reference to the linked mastoids.
9. Epoch −2.5…4.5 s around trigger 1, baseline −1.5…−1.0 s (pre-cue).
10. Reject epochs over ±100 µV (−1.5…4.5 s) or overlapping a `BAD_gross` segment.

### How the pipeline was tuned (Nov 5–6)
- The single-pass neighbour-correlation detector flagged far too many channels. It depends heavily on the reference, blinks make frontal channels look uncorrelated before ICA, and temporalis muscle (>20 Hz) decorrelates temporal sites. That led to the 3-stage design, which uses 1–15 Hz after ICA.
- Mastoids were falsely flagged by the neighbour test in 12 sessions (edge position) → excluded from that test.
- ICA rule iterations:
  - EOG correlation alone removed components ICLabel rated 100% brain (JKSA080 IC4, JKSA091_XC IC7).
  - Requiring ICLabel eye ≥ 0.30 was too strict: it kept eye artifacts ICLabel calls "other/muscle". This collapsed JKSA057_XC from 178 to 4 kept trials.
  - Final rule: EOG-correlated components are removed unless ICLabel says brain ≥ 0.50. This affects ~32 sessions, mostly IC0.
- Regional guard: blocking interpolation whenever any neighbour was high stopped a 3-electrode posterior bad patch in JKSA022 (P6/P4/PO4) from being interpolated → now requires ≥ 3 neighbours.

Known data quirks the pipeline handles: JKSA061 channel labels have a `1-` prefix; the first minutes of JKSA073_XX_1 are whole-head junk (~93 s flagged as `BAD_gross`).

## Bottom line of the audit
- **58 sessions / 45 subjects + 1 unresolved**: 19 neurotypical (1 session each), 25 ADHD (13 with two sessions), and JKSA200 (group unclear).
- **The raw BDFs are clean.** Every BDF (all 116) matches its RESULTS file trial-for-trial: the cue sequence (2–7) is identical to `results.cuetrigger`. Only 4 recordings are truncated (below).
- **The "extra" cue triggers are an artifact, not missing/extra trials.** The upper byte of the BioSemi status word flips mid-session. `pop_biosig` + `mod(type,256)` makes each flip a duplicate event with the same code. Fix: keep only low-byte transitions.
- Jasmine's 2018–2020 preprocessed files used a 1–20 Hz **minimum-phase** (causal) filter, which shifts latency and phase. Not used.

## Corrections to the notes (what the data shows)
| Notes say | Data shows |
|---|---|
| Interleaved 5 aud, 5 vis, 5 aud, 5 vis | **Visual first** in every file: blocks 1–5 visual, 6–10 auditory. RESULTS columns 1–5 = visual. |
| `_1`/`_2` = day; day 2 has 10 blocks | `_1`/`_2` are the two **halves of one visit**. Every visit (day 1 or 2) = 2 files × (5 vis + 5 aud) × 24 trials = **240 visual trials**. The condition label (XX/XC/MX/MC) identifies the visit. |
| Interrupters in 1/3 of trials | Exactly **2/3 interrupted** (1/3 early + 1/3 late) and 1/3 uninterrupted: ~40 trials per cue type per session. |
| Response trigger 25/26 | Response triggers are **24/25** and appear **only in auditory trials**. Visual accuracy must come from RESULTS (`answ` vs `resp`). Code 9 = auditory-only marker; 255 = block boundary. |
| 3 different 1→10 timings | Visual 1→10 is ~4.51–4.53 s on every trial. The three timing configurations are the interrupter onsets (`tSup`): none, early (1.0/1.5/2.25 s), late (1.5/2.0/2.75 s). Targets are always at 0/0.5/1.25 s. Distractors use 2 rhythms. |

Codes (low byte): 99 trial start (≈ −2.0 s) → cue 2–7 (≈ −1.0 s, ±15 ms) → 1 stim onset (0 s) → 10 trial end (≈ +4.5 s). The next trial starts ≥ 1.8 s after 10. 2 FN, 3 FS1 (early), 4 FS2 (late), 5 BN, 6 BS1, 7 BS2. `results.attend` 1 = focal, 2 = broad; `delayS` 0/1/1.5.

## Recording problems
- **JKSA093_MC_2**: crashed (header nrec = −1). EEG has 76/120 visual trials and 75 are usable → **195** for the session.
- **JKSA029_XX_2**: stopped after 3 visual blocks (EEG and behavior agree). Session total is **192**.
- **JKSA074_MC_1**: only the visual half was recorded. Visual is complete.
- **JKSA105_XX_1**: header nrec = −1, but visual is complete (120/120).
- **4096 Hz sessions**: 018_XC, 022_XX, 030_MX, 031_XX.
- **Naming**: `JKSA032_1/2.bdf` = the MX session (6/23/17). Header/subjID typos where the file name is still correct: 027_XX (035), 029_XX (036), 030_MX (037), 075_XC (076), 096_MC (084), 100_MX (101), 091 MC/XC (swapped), 031_MC_1 (DEMO), 064_XC_1 (060). Always key on file names.

## Group / design cells (after excluding JKSA200, 025_MX and 093_XX)
- NT: 19 sessions (all one visit, unmedicated; 8 caffeinated XC).
- ADHD day 1: 16 on stimulant (025_MX excluded), 8 off.
- ADHD two-visit subjects with both sessions usable: 12 (027, 028, 029, 030, 031, 032, 034, 084, 091, 096, 103, 105); 093 lost its day 2. Off-stimulant on day 1: 030, 096, 103, 105. Visit order is confounded with medication state.
- **Caffeine varies too** (XC/MC). In 034 and 091 the caffeine state changes between visits → covariate.
- Spreadsheet subjects with no data here (dismissed/excluded): 020, 024, 033, 038, 048, 059, 060, 077, 095, 107. `Copy of SubjectDataJKSA.xlsx` is password-protected.

## Issues in the old MATLAB scripts (superseded)
1. **visual_preprocessing_FINAL.m filter**: `fir1(128,[1 40]/(fs/2),blackman)` at 256 Hz cannot make a 1 Hz high-pass, so drift passes through.
2. It hard-codes 120 end triggers and keeps the auditory EEG in the merged set.
3. It has no ICA or artifact rejection, and `pop_rmbase(EEG,[],[])` subtracts the whole-epoch mean.
4. **visual_epoching.m**: manual `EEG.event(...)=[]` deletions; references an undefined variable.
5. **individualITC.m**: hard-codes `subjID='VSA040_MC'`; the `setdiff` bad-trial exclusion does not exclude bad trials.
6. **Scott_visual_ITCs_individual2.m** uses 32-channel indices. Parieto-occipital clusters in this montage: **25 PO7, 26 PO3, 27 O1** and **62 PO8, 63 PO4, 64 O2**; 65–67 = VEOG, HEOGl, HEOGr.
7. Jasmine's `missingTrialsfromEEG` miscounts (039, 040, 053, 068, 101 have all 240 trials in the raw BDFs).
