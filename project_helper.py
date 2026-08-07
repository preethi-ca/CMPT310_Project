import numpy as np
import pandas as pd


NUMERIC_IMPUTE_COLUMNS = [
    "median_income",
    "pop_density_sqkm",
    "competitor_count_500m",
    "nearest_transit_distance_m",
]


def engineer_features(df, numeric_impute_values=None):
    engineered = df.copy()

    for col in NUMERIC_IMPUTE_COLUMNS:
        engineered[col] = pd.to_numeric(engineered[col], errors="coerce")

    if numeric_impute_values is None:
        numeric_impute_values = engineered[NUMERIC_IMPUTE_COLUMNS].mean()

    engineered[NUMERIC_IMPUTE_COLUMNS] = engineered[NUMERIC_IMPUTE_COLUMNS].fillna(
        numeric_impute_values
    )

    engineered["log_median_income"] = np.log1p(engineered["median_income"])
    engineered["log_pop_density_sqkm"] = np.log1p(engineered["pop_density_sqkm"])
    engineered["log_competitor_count_500m"] = np.log1p(engineered["competitor_count_500m"])
    engineered["log_nearest_transit_distance_m"] = np.log1p(engineered["nearest_transit_distance_m"])
    engineered["income_density_ratio"] = engineered["median_income"] / (engineered["pop_density_sqkm"] + 1)
    engineered["competition_transit_ratio"] = engineered["competitor_count_500m"] / (engineered["nearest_transit_distance_m"] + 1)

    return engineered

# -----------------------------
# Preprocessing utilities
# -----------------------------
def fit_preprocess(train_df, feature_plan):
    params = {"standard": {}, "onehot": {}}
    for col, how in feature_plan.items():
        if how == "standard":
            mu = train_df[col].mean()
            sigma = train_df[col].std(ddof=0)
            if sigma == 0:
                sigma = 1.0
            params["standard"][col] = (mu, sigma)
        elif how == "one-hot":
            params["onehot"][col] = sorted(train_df[col].unique())
    return params


def transform(df, feature_plan, params):
    X_parts = []
    for col, how in feature_plan.items():
        if how == "drop":
            continue

        if how == "standard":
            mu, sigma = params["standard"][col]
            x = (df[col] - mu) / sigma
            X_parts.append(x.to_numpy().reshape(-1, 1))

        elif how == "one-hot":
            cats = params["onehot"][col]
            onehot = np.zeros((len(df), len(cats)))
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            for i, v in enumerate(df[col]):
                if v in cat_to_idx:
                    onehot[i, cat_to_idx[v]] = 1.0
            X_parts.append(onehot)

    return np.hstack(X_parts)  # (n, d)


class FeaturePreprocessor:
    def __init__(self, feature_plan):
        self.feature_plan = feature_plan
        self.params = None
        self.numeric_impute_values = None

    def fit(self, df, y=None):
        self.numeric_impute_values = df[NUMERIC_IMPUTE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).mean()
        engineered = engineer_features(df, self.numeric_impute_values)
        self.params = fit_preprocess(engineered, self.feature_plan)
        return self

    def transform(self, df):
        if self.params is None:
            raise ValueError("FeaturePreprocessor must be fit before transform.")
        engineered = engineer_features(df, self.numeric_impute_values)
        return transform(engineered, self.feature_plan, self.params)

    def fit_transform(self, df, y=None):
        return self.fit(df, y).transform(df)