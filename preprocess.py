import pandas as pd

from project_helper import fit_preprocess, transform

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "full_information.csv"

feature_plan = {
    "store_name": "drop",
    "city": "one-hot",
    "latitude": "standard",
    "longitude": "standard",
    "median_income": "standard",
    "pop_density_sqkm": "standard",
    "competitor_count_500m": "standard",
    "nearest_transit_distance_m": "standard",
    "pct_age_20_39": "standard",
    "neighbourhood_name": "one-hot",
}

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["neighbourhood_name"] = df["neighbourhood_name"].fillna("Unknown")

# -----------------------------
# Targets
# y_regression -> target ratings
# y_classification -> success labels
# -----------------------------
y_regression = df["target_rating"].to_numpy()
y_classification = df["target_is_successful"].to_numpy()

# -----------------------------
# Fit preprocessing
# -----------------------------
params = fit_preprocess(df, feature_plan)

# -----------------------------
# Feature matrix
# X -> processed feature matrix
# -----------------------------
X = transform(df, feature_plan, params)