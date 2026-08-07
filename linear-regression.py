import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


DATA_PATH = "location-information.csv"
TARGET_COLUMN = "target_rating"
LAMBDA_VALUE = 10.0

categorical_features = [
    "city",
    "primary_category",
    "price_level",
]

numeric_features = [
    "median_income",
    "latitude",
    "longitude",
]


# Code from A2
def kfold_indices(n, k=10, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)


# Code from A2
def rmse(y_true, y_pred):
    total_squared_error = 0.0
    size = len(y_true)

    for i in range(size):
        difference = y_true[i] - y_pred[i]
        total_squared_error = total_squared_error + (difference * difference)

    return np.sqrt(total_squared_error / size)


# Code from A2
def mae(y_true, y_pred):
    total_absolute_error = 0.0
    size = len(y_true)

    for i in range(size):
        difference = y_true[i] - y_pred[i]
        total_absolute_error = total_absolute_error + abs(difference)

    return total_absolute_error / size


def make_model():
    numeric_pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy = "most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown = "ignore",
                    sparse_output = False,
                ),
            ),
        ]
    )

    # convert categorical columns and scale numeric columns before regression
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    #  linear regression with regularization
    model = Pipeline(
        steps = [
            ("preprocess", preprocessor),
            ("ridge", Ridge(alpha=LAMBDA_VALUE)),
        ]
    )

    return model


def evaluate_model(df):
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
        y_train = y.iloc[train_idx].to_numpy(dtype=float)
        y_test = y.iloc[test_idx].to_numpy(dtype=float)

        # train on 9 folds, then predict Yelp ratings for the held-out fold
        model = make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmses.append(rmse(y_test, y_pred))
        maes.append(mae(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    mean_rmse = np.mean(rmses)
    mean_mae = np.mean(maes)
    mean_r2 = np.mean(r2_scores)

    return mean_rmse, mean_mae, mean_r2


def main():
    df = pd.read_csv(DATA_PATH)
    mean_rmse, mean_mae, mean_r2 = evaluate_model(df)

    print("\nLinear Regression for Yelp Rating Prediction\n")
    print(f"Data: {DATA_PATH}")
    print(f"Rows used: {len(df)}")
    print(f"Ridge lambda: {LAMBDA_VALUE}")

    print("10-fold cross-validation performance:")
    print(f"RMSE = {mean_rmse:.2f}")
    print(f"MAE = {mean_mae:.2f}")
    print(f"R^2 = {mean_r2:.2f}")


if __name__ == "__main__":
    main()