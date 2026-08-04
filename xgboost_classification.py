import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from knn_classification import feature_plan, load_and_engineer
from project_helper import fit_preprocess, transform


DATA_PATH = "yelp-and-demo-info.csv"
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
    params = fit_preprocess(train_df, feature_plan)
    x_train = transform(train_df, feature_plan, params)
    x_val = transform(val_df, feature_plan, params)
    x_test = transform(test_df, feature_plan, params)
    y_train = train_df["target_is_successful"].to_numpy()
    y_val = val_df["target_is_successful"].to_numpy()
    y_test = test_df["target_is_successful"].to_numpy()
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model(scale_pos_weight):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )


def grid_search_xgb(x_train, y_train):
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = build_model(scale_pos_weight)

    param_grid = {
        "n_estimators": [100, 250],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search, scale_pos_weight


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


def evaluate_split(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    print(f"\n{name} evaluation at threshold={threshold:.2f}")
    print("Accuracy:", accuracy_score(y_true, predictions))
    print("Precision:", precision_score(y_true, predictions, zero_division=0))
    print("Recall:", recall_score(y_true, predictions, zero_division=0))
    print("F1:", f1_score(y_true, predictions, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, predictions))
    print("Classification report:\n", classification_report(y_true, predictions, zero_division=0))


def main():
    df = load_and_engineer(DATA_PATH)
    train_df, val_df, test_df = split_data(df)
    x_train, x_val, x_test, y_train, y_val, y_test = make_matrix(train_df, val_df, test_df)

    search, scale_pos_weight = grid_search_xgb(x_train, y_train)
    print("Best CV params:", search.best_params_)
    print("Best CV F1:", search.best_score_)
    print("scale_pos_weight:", scale_pos_weight)

    best_params = search.best_params_.copy()

    base_model = build_model(scale_pos_weight)
    base_model.set_params(**best_params)
    base_model.fit(x_train, y_train)

    val_probabilities = base_model.predict_proba(x_val)[:, 1]
    best_threshold, best_val_f1, records = sweep_thresholds(y_val, val_probabilities)
    print("Chosen threshold:", best_threshold)
    print("Validation F1 at chosen threshold:", best_val_f1)

    combined_df = pd.concat([train_df, val_df], axis=0)
    combined_params = fit_preprocess(combined_df, feature_plan)
    combined_x = transform(combined_df, feature_plan, combined_params)
    combined_y = combined_df["target_is_successful"].to_numpy()
    final_x_test = transform(test_df, feature_plan, combined_params)
    final_y_test = test_df["target_is_successful"].to_numpy()

    final_scale_pos_weight = float((combined_y == 0).sum() / max((combined_y == 1).sum(), 1))
    final_model = build_model(final_scale_pos_weight)
    final_model.set_params(**best_params)
    final_model.fit(combined_x, combined_y)

    test_probabilities = final_model.predict_proba(final_x_test)[:, 1]
    evaluate_split("Test", final_y_test, test_probabilities, best_threshold)

    print("\nTop validation thresholds by F1:")
    for threshold, current_f1, precision, recall, accuracy in sorted(records, key=lambda row: row[1], reverse=True)[:5]:
        print(
            f"t={threshold:.2f} f1={current_f1:.4f} precision={precision:.4f} "
            f"recall={recall:.4f} acc={accuracy:.4f}"
        )


if __name__ == "__main__":
    main()