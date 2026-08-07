import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = "location-information.csv"
TARGET_COLUMN = "log_review_count"

MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 5

numeric_features = [
    "price_level",
    "latitude",
    "longitude",
]

categorical_features = [
    "city",
    "primary_category",
]


# Code from A2
def kfold_indices(n, k=10, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    # predict log(1 + review_count).
    df[TARGET_COLUMN] = np.log1p(df["review_count"])

    return df


def make_model():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "tree",
                DecisionTreeRegressor(
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def evaluate_decision_tree(df):
    folds = kfold_indices(len(df), k=10, seed=42)

    rmses = []
    maes = []
    r2_scores = []

    feature_names = numeric_features + categorical_features
    X = df[feature_names]
    y = df[TARGET_COLUMN]

    for fold_index in range(10):
        test_idx = folds[fold_index]
        train_folds = []

        for other_fold_index in range(10):
            if other_fold_index != fold_index:
                train_folds.append(folds[other_fold_index])

        train_idx = np.hstack(train_folds)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # preprocess and train
        model = make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmses.append(rmse)
        maes.append(mae)
        r2_scores.append(r2)

    mean_rmse = np.mean(rmses)
    mean_mae = np.mean(maes)
    mean_r2 = np.mean(r2_scores)

    return mean_rmse, mean_mae, mean_r2


def main():
    df = prepare_data()
    mean_rmse, mean_mae, mean_r2 = evaluate_decision_tree(df)

    print("\nDecision Tree Regression for Review Count Prediction\n")

    print("Decision tree settings:")
    print(f"max_depth = {MAX_DEPTH}")
    print(f"min_samples_leaf = {MIN_SAMPLES_LEAF}\n")

    print("10-fold cross-validation performance:")
    print(f"RMSE = {mean_rmse:.4f}")
    print(f"MAE = {mean_mae:.4f}")
    print(f"R^2 = {mean_r2:.4f}")


if __name__ == "__main__":
    main()