import numpy as np
import pandas as pd

from project_helper import fit_preprocess, transform


DATA_PATH = "full_information.csv"
TARGET_COLUMN = "target_rating"
LAMBDA_VALUES = [0.0, 0.01, 1.0, 10.0]

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


# Code from A2
def add_intercept(X_row_major):
    return np.hstack([np.ones((X_row_major.shape[0], 1)), X_row_major])


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


def r2_score(y_true, y_pred):
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_residual / ss_total)


def fit_ridge_regression(X_train, y_train, lambda_value):
    y_train = y_train.reshape(-1, 1)

    if lambda_value == 0:
        result = np.linalg.lstsq(X_train, y_train, rcond=None)
        weights = result[0]

        return weights

    regularizer = np.eye(X_train.shape[1])
    regularizer[0, 0] = 0.0

    X_transpose = X_train.T

    left_side = np.matmul(X_transpose, X_train)
    left_side = left_side + (lambda_value * regularizer)

    right_side = np.matmul(X_transpose, y_train)

    return np.linalg.solve(left_side, right_side)


def main():
    df = pd.read_csv(DATA_PATH)
    # handle missing data
    df["neighbourhood_name"] = df["neighbourhood_name"].fillna("Unknown")
    folds = kfold_indices(len(df), k=10, seed=42)

    print("\nRegression for Yelp Rating Prediction\n")
    print("lambda |  RMSE  |  MAE   |   R^2")
    print("----------------------------------")

    for lambda_value in LAMBDA_VALUES:
        rmses = []
        maes = []
        r2_scores = []

        for i in range(10):
            test_idx = folds[i]
            train_folds = []

            for j in range(10):
                if j != i:
                    train_folds.append(folds[j])

            train_idx = np.hstack(train_folds)

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
            y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

            # use the preprocessing functions
            params = fit_preprocess(train_df, feature_plan)
            X_train = add_intercept(transform(train_df, feature_plan, params))
            X_test = add_intercept(transform(test_df, feature_plan, params))

            weights = fit_ridge_regression(X_train, y_train, lambda_value)
            predictions = np.matmul(X_test, weights)
            y_pred = predictions.reshape(-1)

            rmses.append(rmse(y_test, y_pred))
            maes.append(mae(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))

        # print the output
        mean_rmse = np.mean(rmses)
        mean_mae = np.mean(maes)
        mean_r2 = np.mean(r2_scores)

        print(f"{lambda_value:<6} | {mean_rmse:.4f} | {mean_mae:.4f} | {mean_r2:.4f}")


if __name__ == "__main__":
    main()