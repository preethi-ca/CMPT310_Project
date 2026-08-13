import argparse
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
import xgboost as xgb

from knn_classification import feature_plan, load_and_engineer
from project_helper import FeaturePreprocessor


DATA_PATH = "data/location-information-with-competitors.csv"
RANDOM_STATE = 42


def split_data(df):
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target_is_successful"],
        random_state=RANDOM_STATE,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        stratify=train_val_df["target_is_successful"],
        random_state=RANDOM_STATE,
    )
    return train_df, val_df, test_df


def make_matrix(train_df, val_df, test_df):
    preprocessor = FeaturePreprocessor(feature_plan)
    x_train = preprocessor.fit_transform(train_df)
    x_val = preprocessor.transform(val_df)
    x_test = preprocessor.transform(test_df)
    y_train = train_df["target_is_successful"].to_numpy()
    y_val = val_df["target_is_successful"].to_numpy()
    y_test = test_df["target_is_successful"].to_numpy()
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model(scale_pos_weight):
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        n_estimators=1000,
        scale_pos_weight=scale_pos_weight,
    )


def grid_search_xgb(x_train, y_train):
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = build_model(scale_pos_weight)

    param_grid = {
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 3, 5, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0.0, 0.5, 1.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [1.0, 3.0, 5.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        scoring="f1",
        cv=cv,
        n_iter=25,
        n_jobs=1,
        verbose=0,
        random_state=RANDOM_STATE,
    )
    search.fit(x_train, y_train)
    return search, scale_pos_weight


def best_iteration_from_booster(booster):
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None:
        return best_iteration + 1

    best_score = getattr(booster, "best_score", None)
    if best_score is None:
        return None

    try:
        evaluation = booster.evals_result()["validation_0"]["logloss"]
    except (AttributeError, KeyError):
        return None

    if not evaluation:
        return None

    return int(np.argmin(evaluation)) + 1


def fit_native_booster(x_train, y_train, x_val=None, y_val=None, params=None, num_boost_round=1000):
    train_matrix = xgb.DMatrix(x_train, label=y_train)
    evals = []
    if x_val is not None and y_val is not None:
        evals.append((xgb.DMatrix(x_val, label=y_val), "validation"))

    booster = xgb.train(
        params=params,
        dtrain=train_matrix,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=50 if evals else None,
        verbose_eval=False,
    )
    return booster


def sweep_thresholds(y_true, probabilities, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = 0.5
    best_f1 = -1.0
    fallback_threshold = 0.5
    fallback_f1 = -1.0
    records = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        current_f1 = f1_score(y_true, predictions, zero_division=0)
        current_precision = precision_score(y_true, predictions, zero_division=0)
        current_recall = recall_score(y_true, predictions, zero_division=0)
        current_accuracy = accuracy_score(y_true, predictions)
        records.append((threshold, current_f1, current_precision, current_recall, current_accuracy))

        if current_f1 > fallback_f1:
            fallback_f1 = current_f1
            fallback_threshold = threshold

        if len(np.unique(predictions)) < 2:
            continue

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    if best_f1 < 0:
        best_threshold = fallback_threshold
        best_f1 = fallback_f1

    return best_threshold, best_f1, records


def fit_calibrator(y_true, probabilities):
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probabilities, y_true)
    return calibrator


def evaluate_split(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    print(f"\n{name} evaluation at threshold={threshold:.2f}")
    print("Accuracy:", accuracy_score(y_true, predictions))
    print("Precision:", precision_score(y_true, predictions, zero_division=0))
    print("Recall:", recall_score(y_true, predictions, zero_division=0))
    print("F1:", f1_score(y_true, predictions, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, predictions))
    print("Classification report:\n", classification_report(y_true, predictions, zero_division=0))


def parse_args():
    parser = argparse.ArgumentParser(description="Run XGBoost classification on a CSV dataset.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_and_engineer(args.csv_path)
    train_df, val_df, test_df = split_data(df)
    x_train, x_val, x_test, y_train, y_val, y_test = make_matrix(train_df, val_df, test_df)

    search, scale_pos_weight = grid_search_xgb(x_train, y_train)
    print("Best CV params:", search.best_params_)
    print("Best CV F1:", search.best_score_)
    print("scale_pos_weight:", scale_pos_weight)

    best_params = search.best_params_.copy()

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": RANDOM_STATE,
        "eta": best_params["learning_rate"],
        "max_depth": best_params["max_depth"],
        "min_child_weight": best_params["min_child_weight"],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "gamma": best_params["gamma"],
        "alpha": best_params["reg_alpha"],
        "lambda": best_params["reg_lambda"],
        "scale_pos_weight": scale_pos_weight,
        "nthread": 1,
    }

    base_booster = fit_native_booster(x_train, y_train, x_val, y_val, base_params)

    val_probabilities = base_booster.predict(xgb.DMatrix(x_val))
    calibrator = fit_calibrator(y_val, val_probabilities)
    calibrated_val_probabilities = calibrator.transform(val_probabilities)

    best_threshold, best_val_f1, records = sweep_thresholds(y_val, calibrated_val_probabilities)
    print("Chosen threshold:", best_threshold)
    print("Validation F1 at chosen threshold:", best_val_f1)

    final_x_test = x_test
    final_y_test = y_test
    test_probabilities = base_booster.predict(xgb.DMatrix(final_x_test))
    calibrated_test_probabilities = calibrator.transform(test_probabilities)
    evaluate_split("Test", final_y_test, calibrated_test_probabilities, best_threshold)

    print("\nTop validation thresholds by F1:")
    for threshold, current_f1, precision, recall, accuracy in sorted(records, key=lambda row: row[1], reverse=True)[:5]:
        print(
            f"t={threshold:.2f} f1={current_f1:.4f} precision={precision:.4f} "
            f"recall={recall:.4f} acc={accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
