# Ten-figure reproduction

Run from this directory: `python regenerate_all_figures.py`. The driver runs seven
source scripts and copies their generated outputs into `figures/` with manuscript
figure numbers. `code/` holds all source files, including standalone tests and
slower legacy validation not needed for standard figure generation.

Validation completed on 2026-10-20: all 10 figure outputs produced.
All 13 existing package unit tests passed.

Important: `fig8_unified_bimodal.png` illustrates trajectories of a rotationally
symmetric bimodal landscape with a degenerate critical latitude; it is not a
direct verification of the generic Morse theorem. `classification_rate_check.py`
checks an anisotropic maximum's predicted rates, not the global almost-everywhere
classification.
