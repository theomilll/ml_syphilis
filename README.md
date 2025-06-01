# Predicting Congenital Syphilis From PMCP Records

This project aims to predict congenital syphilis using machine learning techniques based on PMCP (Prevention of Mother-to-Child Transmission Program) records.

## Project Structure

```
.
├── data/                    # raw .csv from Mendeley (attributes.csv, data_set.csv)
├── notebooks/               # exploratory & sanity-check notebooks
├── src/
│   ├── preprocessing.py     # cleaning + feature engineering pipeline
│   ├── model_selection.py   # search spaces, CV & training routines
│   ├── evaluation.py        # metrics, plots, interpretability
│   └── utils.py
├── models/                  # persisted artefacts (.joblib or .pkl)
├── reports/
│   ├── figures/             # auto-generated charts
│   └── tables/
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```

## Setup

1.  Clone the repository.
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

(Instructions on how to run the pipeline will be added here as development progresses.)

## Reproducibility

Random seeds are fixed to `42` across the project to ensure reproducibility of results. See `src/utils.py` and individual scripts for details.
