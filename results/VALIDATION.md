# Validation recorded on 2026-10-20

- `PYTHONPATH=src python -m pytest -q`: **13 passed**.
- `MPLBACKEND=Agg python scripts/regenerate_core.py --output figures/generated`: **six images produced**.
- `python scripts/classification_rate_check.py`: completed and saved output; asymptotic rates agree with manuscript predictions in the tabulated tests.
- No assertion that numerical unit tests prove the global Morse theorem.
-