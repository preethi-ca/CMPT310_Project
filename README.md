# CMPT310_Project

This repository contains the code and datasets for a CMPT 310 restaurant location success predictor. The project uses restaurant, demographic, geographic, transit, and competitor information from Metro Vancouver restaurants to support two prediction tasks:

- Regression: predict the expected Yelp rating out of 5.0.
- Classification: predict whether a restaurant is successful or not successful.

The project includes four main model scripts:

- `src/linear-regression.py`: Ridge regression for `target_rating`.
- `src/decision-tree.py`: Decision Tree regression for `target_rating`.
- `src/knn_classification.py`: K-Nearest Neighbours classification for `target_is_successful`.
- `src/xgboost_classification.py`: XGBoost classification for `target_is_successful`.

## Repository Structure

```text
CMPT310_Project/
  data/                 CSV datasets, GTFS file, and cached data files
  outputs/              Generated model plots and confusion matrices
  src/                  Python model, preprocessing, and data helper scripts
  Website_CMPT310/      Web interface source code
  README.md             Setup and run instructions
  requirements.txt      Python dependencies for the project scripts
  run_pipeline.py       One-command runner for the main model pipeline
  pyproject.toml        Poetry project config
  poetry.lock           Poetry lock file
```

## Main Data Files

- `data/location-information.csv`: dataset used by the regression scripts.
- `data/location-information-with-competitors.csv`: current dataset used when running the classification scripts for the final project pipeline.
- `data/yelp-and-demo-info.csv`: earlier combined Yelp/demographic dataset kept for reference.
- `data/transit_stops.csv`: transit stop data used by the feature-building helper scripts.

The target columns should stay out of the model input features:

- `target_rating` is the regression target.
- `target_is_successful` is the classification target.

## Web Interface Languages

The web interface in `Website_CMPT310/` is built with:

- TypeScript and TSX for the React components.
- CSS/Tailwind tooling for styling.
- JavaScript/Node.js for package scripts and build tooling.
- Vinext/Vite for the frontend build.
- Leaflet with OpenStreetMap tiles for the browser map.

## Setup

Use Python 3.13 or newer. The cleanup checks below were validated with Python 3.14.3.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run The Main Pipeline

The simplest way to run the main model pipeline is:

```bash
python run_pipeline.py
```

This runs the four main project models in order:

1. Ridge regression
2. Decision Tree regression
3. KNN classification
4. XGBoost classification

The pipeline passes `data/location-information-with-competitors.csv` into the classification scripts so the KNN and XGBoost models use the current competitor-enriched dataset.

If XGBoost is taking too long during a quick local check, you can run the other three models first:

```bash
python run_pipeline.py --skip-xgboost
```

To also regenerate the existing visualization PNG files in `outputs/`:

```bash
python run_pipeline.py --include-visualizations
```

To preview the commands without running them:

```bash
python run_pipeline.py --dry-run
```

## Run Individual Scripts

You can also run each script directly from the repository root:

```bash
python src/linear-regression.py
python src/decision-tree.py
python src/knn_classification.py data/location-information-with-competitors.csv
python src/xgboost_classification.py data/location-information-with-competitors.csv
python src/regression-visualization.py
python src/knn-visualization.py
```

## Other Helper Scripts

- `src/scrape_yelp_data.py`: gathers restaurant data from Yelp.
- `src/generate-demographic-data.py`: gathers demographic data.
- `src/build_transit_stops.py`: extracts transit stop data from GTFS files.
- `src/competitors.py`: calculates competitor count and nearest transit distance features.
- `src/preprocess.py` and `src/project_helper.py`: shared preprocessing and feature-engineering helpers.

These helper scripts support the dataset-building process. The four main model scripts listed above are the reproducible model pipeline for the final code submission.

## Code Submission Checklist

For the Canvas ZIP submission, include:

- Source code: `src/`, `run_pipeline.py`, and the config files.
- Dependencies/config: `requirements.txt`, `pyproject.toml`, and `poetry.lock`.
- Data files required to reproduce the models, especially `data/location-information.csv` and `data/location-information-with-competitors.csv`.
- Existing output plots if required by the report: files in `outputs/`.
- This `README.md`.

Do not include local-only folders such as `.git`, `.venv`, `__pycache__`, or `.pytest_cache`.

PowerShell example for a model-code submission ZIP:

```powershell
$items = @(
  "README.md",
  "requirements.txt",
  "pyproject.toml",
  "poetry.lock",
  "run_pipeline.py",
  "src",
  "data",
  "outputs"
)
Compress-Archive -Path $items -DestinationPath CMPT310_Project_code_submission.zip -Force
```

If the group decides the website source should also be included in the same Canvas ZIP, add `Website_CMPT310` to the `$items` list before running `Compress-Archive`.
