import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from project_helper import fit_preprocess, transform

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
    "log_median_income": "standard",
    "log_pop_density_sqkm": "standard",
    "log_competitor_count_500m": "standard",
    "log_nearest_transit_distance_m": "standard",
    "income_density_ratio": "standard",
    "competition_transit_ratio": "standard",
    "neighbourhood_name": "one-hot",
}


def main():
    df = pd.read_csv(DATA_PATH)
    df["neighbourhood_name"] = df["neighbourhood_name"].fillna("Unknown")

    # testing different feature engineering techniques for the numeric features, including log transforms and ratios
    for col in [
        "median_income",
        "pop_density_sqkm",
        "competitor_count_500m",
        "nearest_transit_distance_m",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["log_median_income"] = np.log1p(df["median_income"])
    df["log_pop_density_sqkm"] = np.log1p(df["pop_density_sqkm"])
    df["log_competitor_count_500m"] = np.log1p(df["competitor_count_500m"])
    df["log_nearest_transit_distance_m"] = np.log1p(df["nearest_transit_distance_m"])
    df["income_density_ratio"] = df["median_income"] / (df["pop_density_sqkm"] + 1)
    df["competition_transit_ratio"] = df["competitor_count_500m"] / (df["nearest_transit_distance_m"] + 1)

    params = fit_preprocess(df, feature_plan)
    X = transform(df, feature_plan, params)
    y = df["target_is_successful"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=19))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Cross-validation accuracies:", cv_scores)
    print("Mean cross-validation accuracy:", cv_scores.mean())
    print("Std cross-validation accuracy:", cv_scores.std())


if __name__ == "__main__":
    main()
