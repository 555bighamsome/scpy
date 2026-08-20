# Reproducing the “Bad Luck or a Changing World?” analysis

The main script is [`bad_luck_study.py`](bad_luck_study.py). It reads the CSV files in the repository's `g2ds8-osfstorage-archive/` directory, constructs behavioural change events, fits the fixed and adaptive Rescorla–Wagner models, and writes outputs to `results/bad_luck_study/`.

The repository currently contains 419 CSV files, but the reported analysis was run on the organizer-provided 400-file workshop subset. [`workshop_subset_files.txt`](workshop_subset_files.txt) records that exact set so the published numbers are reproducible instead of silently changing when extra files are added.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
```

## Run the analysis

```bash
PYTHONPATH=analysis python analysis/bad_luck_study.py
```

The first run creates preprocessing caches in `.analysis_work/cache/`. Remove those cache files, or call the preprocessing functions with `rebuild=True`, after changing filtering or event-construction logic.

## Run integrity tests

```bash
PYTHONPATH=analysis pytest -q analysis/test_change_mechanism_study.py
```

The tests check sequence padding, the random-choice baseline, finite temporally valid RW likelihoods, parameter constraints, and that GRU features contain only information available before the current choice.

## Files

- `bad_luck_study.py`: probability-change behavioural analysis, model comparison, bootstrap intervals, plotting, and report generation.
- `change_mechanism_study.py`: shared session loader, RW/adaptive-RW likelihoods, participant folds, GRU utilities, and evaluation helpers.
- `test_change_mechanism_study.py`: integrity tests for preprocessing and model inputs.

See [`../BAD_LUCK_STUDY.md`](../BAD_LUCK_STUDY.md) for the full English research document and a line-by-line explanation of data selection and analysis.
