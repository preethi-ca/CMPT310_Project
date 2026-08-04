import numpy as np
import pandas as pd

from project_helper import fit_preprocess, transform
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = "full_information.csv"
TARGET_COLUMN = "target_rating"

MAX_DEPTH_VALUES = [2, 3, 4, 5, None]
MIN_SAMPLES_LEAF_VALUES = [5, 10, 20]

# need to update which features will end up being used
feature_plan = {
    "store_name": "drop",
    "city": "one-hot",
    "latitude": "drop",
    "longitude": "drop",
    "median_income": "drop",
    "pop_density_sqkm": "drop",
    "competitor_count_500m": "drop",
    "nearest_transit_distance_m": "drop",
    "pct_age_20_39": "drop",
    "neighbourhood_name": "drop",
}


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


def evaluate_decision_tree(df, max_depth, min_samples_leaf):

    folds = kfold_indices(len(df), k=10, seed=42)
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

        # fit preprocessing on the training fold only
        params = fit_preprocess(train_df, feature_plan)
        X_train = transform(train_df, feature_plan, params)
        X_test = transform(test_df, feature_plan, params)

        # train the decision tree regression model
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # predict Yelp ratings for the test fold
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
    df["neighbourhood_name"] = df["neighbourhood_name"].fillna("Unknown")

    print("\nDecision Tree Regression for Yelp Rating Prediction\n")
    print("depth | min leaf |  RMSE  |  MAE   |   R^2")
    print("--------------------------------------------")

    best_rmse = None
    best_depth = None
    best_min_samples_leaf = None
    best_mae = None
    best_r2 = None

    for max_depth in MAX_DEPTH_VALUES:
        for min_samples_leaf in MIN_SAMPLES_LEAF_VALUES:
            mean_rmse, mean_mae, mean_r2 = evaluate_decision_tree(
                df,
                max_depth,
                min_samples_leaf,
            )

            if max_depth is None:
                depth_label = "None"
            else:
                depth_label = str(max_depth)

            print(
                f"{depth_label:<5} | "
                f"{min_samples_leaf:<8} | "
                f"{mean_rmse:.4f} | "
                f"{mean_mae:.4f} | "
                f"{mean_r2:.4f}"
            )

            if best_rmse is None:
                best_rmse = mean_rmse
                best_depth = max_depth
                best_min_samples_leaf = min_samples_leaf
                best_mae = mean_mae
                best_r2 = mean_r2
            else:
                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_depth = max_depth
                    best_min_samples_leaf = min_samples_leaf
                    best_mae = mean_mae
                    best_r2 = mean_r2

    print("\nBest decision tree setting:")
    print(f"max_depth = {best_depth}")
    print(f"min_samples_leaf = {best_min_samples_leaf}")
    print(f"RMSE = {best_rmse:.4f}")
    print(f"MAE = {best_mae:.4f}")
    print(f"R^2 = {best_r2:.4f}")


if __name__ == "__main__":
    main()