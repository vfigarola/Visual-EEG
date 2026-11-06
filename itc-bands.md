# ITC by frequency band: theta (4–8 Hz) and slow (2–4 Hz) (Nov 6, 2026)

Supplement to the ITC section of `analysis-plan.md`. The 2–20 Hz averaged analysis (`itc_analysis.py`, `derivatives/analysis/itc/`, `figures/itc/`) is unchanged. Script: `python/analysis/itc_bands_analysis.py`. Outputs: `derivatives/analysis/itc_bands/{theta,slow}/` and `figures/itc_bands/{theta,slow}/` (+ `supplementary/`).

## Method
- Same PPC export, pooled PO/O cluster, participants, families, tests and figure layout as the 2–20 Hz analysis; PPC is averaged over theta = 4–8 Hz or slow = 2–4 Hz (4 Hz is in both). Main = NT vs ADHD (all), correct trials.
- Syllable-locked measures use each band's own blind peak latency: theta 203 ms (same as the broadband), slow 266 ms (cue 219 ms).
- New "stream following" measures (new family ITC_stream), broad − focal: prep (0–1.0 s after the cue), stream_early (1.0–2.65 s, early-interrupter trials), stream_late (1.5–3.15 s, late-interrupter trials).
- Scalp maps use band-specific 64-channel PPC and need a short MNE export (`<session>_bands.npz`, run on the Mac).

## Results (correct trials, broad − focal PPC, NT / ADHD (all))
- Interrupter syllables: theta +0.062 / +0.052 (dz 1.15 / 1.35, p ≤ .0006); slow +0.091 / +0.065 (dz 1.66 / 1.45, p ≤ .0004). Both bands show it; the slow-band effect is larger.
- Interrupter stream over time: slow early +0.078 / +0.054 and late +0.069 / +0.055 (dz 1.3–1.6, p ≤ .0004); theta early +0.035 / +0.025 (dz 0.8–0.9), late +0.046 / +0.041 (dz 1.0–1.5). Time-course cluster tests: broad > focal during almost the entire interrupter stream in both groups, both bands (slow: from the first interrupter syllable to ~3.3 s).
- Target syllables: theta lower on broad (NT −0.015 p = .09; ADHD −0.019 p = .007); slow: no consistent effect (tsyl NT p = .03, ADHD p = .24).
- Preparation (0–1.0 s): lower on broad in both bands (theta p ≤ .016; slow NT p = .006, ADHD p = .09); the cue pictures differ, so this is not stream attention.
- Cue: slow-band cue delta is large (dz 0.9–1.0, p ≤ .0004); theta cue is n.s.
- Lateralization of interrupter effect (right − left): slow NT dz 1.15 p < .001, ADHD dz 0.57 p = .009; theta NT p = .04, ADHD p = .07.
- **NT vs ADHD (all): nothing below p = .05 in either band** (smallest: slow int_early p = .060, FDR .18; slow isyl p = .09; NT larger than ADHD). No between-group cluster p < .1.
