# Jasmine visual EEG: analysis plan (agreed Nov 6, 2026)

Input: preprocessed epochs in `Jasmine-data/derivatives/preproc_v1/<session>/<session>-epo.fif` (metadata attached) plus `<session>_trials.csv`. Exclusions: JKSA200_XC (group unknown); JKSA025_MX and JKSA093_XX (≥ 50% trials rejected).
Code: `Jasmine-data/python/analysis/`. Figures: `Jasmine-data/derivatives/figures/` (PDF + PNG).

## Aims (from Victoria's notes)
1. Compare ADHD vs neurotypical in a visual selective/switching attention task.
2. ERPs and phase coherence (ITC): expect weaker attentional modulation in ADHD.
3. Stimulants (and caffeine) may "correct" neural signatures of attention.

## Task reminders
- Fixation dot. **Targets** stream above the dot (3 syllables at 0 / 0.5 / 1.25 s). **Distractors** on the right (5 syllables, +0.2 s). **Interrupter** ("supra target") on the left, on 2/3 of trials: early (+1.0 s → 1.0/1.5/2.25 s) or late (+1.5 s → 1.5/2.0/2.75 s).
- Cue at −1 s says **focal** (ignore the interrupter) or **broad** (prepare to switch to it).
- Six conditions: FN, FS1, FS2, BN, BS1, BS2 (F/B = focal/broad; N/S1/S2 = none/early/late interrupter).
- No-interrupter trials matter: in broad, the participant is preparing to switch but never does → look at target N1s.
- Early interrupter overlaps target encoding (discard + switch + restart memory). Late comes after the targets finish (little energetic overlap). Prior ADHD (auditory) paper Fig 3: no-interrupter traces look alike; with an interrupter, N1 is larger in NT.

## Groups (day-1 sessions)
| group | n | sessions |
|---|---|---|
| NT | 19 | all NT (8 XC caffeinated, 11 XX) |
| ADHD off-stimulant | 8 | 030_XX, 068_XX, 073_XX, 096_XC, 102_XX, 103_XX, 105_XX, 110_XX |
| ADHD on-stimulant | 16 | 026, 027, 028, 029, 031, 032, 034, 035, 040, 074, 084, 091, 093, 098, 100, 112 (8 MC, 8 MX) |

Comparisons:
1. NT vs ADHD-off.
2. NT vs ADHD-on.
3. ADHD-off vs ADHD-on (day 1, between-person).
4. Overall: NT vs all ADHD day 1 (24).
5. **Within-person on vs off**: the 12 two-visit ADHD participants (027, 028, 029, 030, 031, 032, 034, 084, 091, 096, 103, 105). Note that medication is confounded with visit order.

ADHD-off has n = 8 → low power; report effect sizes and use permutation statistics.

**Main comparison (decided Nov 6): NT vs ADHD (all).** ADHD-off vs ADHD-on (day 1) and within-person on vs off are supplementary. All comparisons are still computed and saved.

**ERP interrupter P3, focal vs broad (correct trials, pooled PO/O cluster, 250–450 ms after the 1st interrupter syllable)**, as reported in the auditory paper:
- Early interrupter: broad > focal in NT (+4.7 µV, dz 1.29, p < .001) and ADHD (all) (+3.2 µV, dz 1.04, p < .001); group × condition p = .15 (g = −0.45, NT larger).
- Late interrupter: broad > focal in NT (+2.3 µV, dz 0.80, p = .003) and ADHD (all) (+2.7 µV, dz 0.76, p = .002); group × condition p = .71.
- The late-interrupter broad P3 peaks around 450–500 ms, slightly after the fixed window; the time-course tests (broad ≠ focal, 1.85–2.12 s in NT) cover it.

## Decisions
- **Trials**: both versions. **Correct trials only** is the main analysis (matches Jasmine's individualITC.m); **all clean trials** is a robustness check. No minimum-trial cutoff for now (JKSA053_XX has 6 correct broad/no-interrupter trials).
- **Caffeine**: covariate in the group statistics; reported per group.
- **On vs off**: day-1 between-person plus within-person (12 pairs).
- **ERP scale**: microvolts only. The cue-N1-normalized version was dropped (Nov 6): dividing by a small cue N1 made it unstable.

## 1. Behavior (RESULTS, all trials)
- % correct (all 3 syllables in order) per attend × interrupter cell; also per-syllable accuracy and RT where useful.
- Figure: per condition per group, with individual points.
- Stats: group × attend × interrupter (permutation-based / mixed model) with a caffeine covariate.
- Day-1 → day-2 change in accuracy (12 two-visit participants), as asked in the original notes.

## 2. ERPs (erp_analysis.py, revised Nov 6 after the first real-data run)
- Cluster (changed Nov 6): **one pooled PO/O cluster** (PO7/PO3/O1 + PO8/PO4/O2) for every primary measure and figure. Only the N1 effects were lateralized (larger on the right, contralateral to the left-field interrupter); P3, late negativity, prep and cue were not. Pooling left the group results essentially unchanged and halves the number of tests. Left/right values stay in `erp_subject_measures.csv`.
- **N1 lateralization** (secondary, own family, with group and within-person tests): right − left cluster of the collapsed interrupter-syllable N1 delta; negative = larger modulation contralateral to the interrupter.
- Measures (broad − focal = delta; each component its own FDR family):
  - **N1**: one syllable-N1 latency (212 ms) from the response pooled over every target and interrupter syllable, both conditions, all participants; 50-ms window, minus the 50 ms before each syllable. Cue N1 at its own latency (145 ms). The first version (per-onset latency search) picked inconsistent peaks (105 ms; 250 ms at the window edge).
  - **Cue response**: broad vs focal cue are different pictures → a cue-processing difference, not stream attention.
  - **Preparatory shift**: mean 0–1.0 s, interrupter conditions pooled, pre-cue baseline.
  - **P3**: 250–450 ms after the first interrupter syllable, re 100 ms before it.
  - **Late negativity**: 1.0–1.4 s after the first interrupter syllable, pre-cue baseline.
- Figures (main = NT vs ADHD (all); 3-group, on/off and all-trials versions in `figures/erp/supplementary`): grand averages per group (mean ± SEM, 20 Hz low-pass for display, within-group broad ≠ focal time-course bars, P3 window with broad vs focal stars); `erp_P3_focal_vs_broad`; difference waves with cluster tests; box plots for N1 and for preparation/P3/late negativity; within-person; scalp maps at N1, P3 and late negativity.
- First results (correct trials; the group lines below are from the left/right version, pooled results are in the last block):
  - Every delta differs from 0 across participants: P3 +2.4 to +4.0 µV (dz 0.7–1.1); late negativity −2.2 to −2.8 µV; interrupter N1 −0.8 to −1.9 µV; preparatory shift about −0.4 µV; target N1 delta positive (+0.7 to +0.9 µV, smaller target N1 when preparing to switch).
  - Group differences are weak: NT > ADHD-on right P3 early (p = .023, FDR .09); ADHD-on vs off right cue delta (p = .013, FDR .03); within-person on vs off right cue delta (p = .034, FDR .07). No cluster-test group differences.
  - One ADHD-off participant is an outlier on late negativity (about +7 µV).
- **Collapsed syllable ERPs** (added Nov 6, for power): syllable-locked average (−0.10 to 0.45 s) over target syllables pooled across all 3 interrupter conditions until the interrupter starts (7 onset × condition averages), and over the 6 interrupter syllables (early + late). N1 at 212 ms; own FDR family "N1_collapsed". Figures `erp_syllables_collapsed`, `erp_deltaN1_collapsed_*`.
  - Target syllables: smaller N1 on broad trials, strongest in NT (+1.3 to +1.5 µV, dz 1.1); weaker in ADHD. NT vs ADHD (all), left cluster: p = .023 (FDR .09); NT vs ADHD-on left: p = .032.
  - Interrupter syllables: larger N1 on broad trials in every group (−1.1 to −2.3 µV); no group differences.
- **Pooled-cluster results** (current version; uncorrected p, FDR where it differs):
  - Target-syllable N1 (collapsed): NT +1.41 µV vs ADHD-on +0.49 (p = .037) and vs all ADHD +0.63 (p = .042, FDR .08).
  - Cue response: ADHD-on larger than NT (p = .046) and than ADHD-off (p = .045).
  - Late negativity, late interrupter: smaller in ADHD-off than NT (p = .047, FDR .10).
  - N1 lateralization: −0.59 µV across everyone (p = .001); ADHD-off most lateralized (−1.16), ADHD-on vs off p = .050.
  - Within-person on vs off: nothing below p = .05 (interrupter N1 and cue p ≈ .07).

## 3. ITC
- Morlet ITC (3 cycles at the lowest frequency, as in `newtimef [3 0.5]`), 1–20 Hz, averaged across frequency → ITC time course per cluster and condition; time-frequency maps too.
- **Trial-count bias (decided Nov 6)**: pairwise phase consistency (PPC, unbiased by trial count) instead of equalizing trial counts, which would have forced N = 6. All trials used; classic ITC kept as a supplementary figure. 2–20 Hz, 3 → 15 cycles. Script `itc_analysis.py`.
- Measures: ITC peaks at onsets (+N1 latency); **ΔITC** (broad − focal) at the interrupter onsets; ERP vs ITC side by side.
- Implemented as: PPC averaged over 2–20 Hz and the pooled PO/O cluster, 100-ms window at one syllable peak latency (203 ms; cue 156 ms) found blind to condition; same collapsed-syllable, lateralization and cue measures, families and comparisons as the ERPs. Main figures NT vs ADHD (all), others supplementary.
- **First results (correct trials, Nov 6):**
  - Interrupter syllables: phase locking much higher on broad than focal in both groups (NT +0.035 PPC, ADHD (all) +0.032; dz ≈ 1.5, p < .001). On focal trials there is almost no locking to the interrupter. The effect is right-lateralized (contralateral; p ≤ .007 in both groups).
  - Target syllables: lower locking on broad trials, same direction as the ERP target-syllable N1. Significant in ADHD (all) (dz −0.90, p < .001), not in NT (dz −0.37, p = .13).
  - Time-frequency: interrupter effect = evoked theta burst (~6–8 Hz, ~200 ms) plus sustained 2–4 Hz locking to the stream; target reduction is sustained at ~5–7 and ~10–11 Hz.
  - NT vs ADHD (all): no group differences on any measure (all p > .4). Only supplementary effect: ADHD-on vs ADHD-off on target syllables (p = .036, FDR .07).

## 4. Statistics
- Primary: one summary value per subject (ΔN1, ΔP3, late negativity, preparatory shift, ΔITC, accuracy) → permutation tests between groups, with a caffeine covariate (permutation of residuals / linear model).
- Exploratory: cluster-based permutation over time.

## Order of work
1. Behavior.
2. ERPs.
3. ITC.
4. Statistics and paper figures.
