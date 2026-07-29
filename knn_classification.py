import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    "log_median_income": "standard",
    "log_pop_density_sqkm": "standard",
    "log_competitor_count_500m": "standard",
    "log_nearest_transit_distance_m": "standard",
    "income_density_ratio": "standard",
    "competition_transit_ratio": "standard",
    "neighbourhood_name": "drop",
}


def load_and_engineer(path=DATA_PATH):
    df = pd.read_csv(path)
    df["neighbourhood_name"] = df["neighbourhood_name"].fillna("Unknown")

    # coerce numeric and impute medians
    for col in [
        "median_income",
        "pop_density_sqkm",
        "competitor_count_500m",
        "nearest_transit_distance_m",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # engineered features
    df["log_median_income"] = np.log1p(df["median_income"])
    df["log_pop_density_sqkm"] = np.log1p(df["pop_density_sqkm"])
    df["log_competitor_count_500m"] = np.log1p(df["competitor_count_500m"])
    df["log_nearest_transit_distance_m"] = np.log1p(df["nearest_transit_distance_m"])
    df["income_density_ratio"] = df["median_income"] / (df["pop_density_sqkm"] + 1)
    df["competition_transit_ratio"] = df["competitor_count_500m"] / (df["nearest_transit_distance_m"] + 1)

    return df

def build_feature_matrix(df, feature_plan):
    params = fit_preprocess(df, feature_plan)
    X = transform(df, feature_plan, params)
    return X


# -----------------------------
# loosely from q2
# -----------------------------
def knn_predict(X_train, y_train, X_test, k, p=2, weights=None):
    # supports p=1 (Manhattan) or p=2 (Euclidean); weights=None or 'distance'
    preds = []
    for x in X_test:
        if p == 1:
            dists = np.sum(np.abs(X_train - x), axis=1)
        else:
            dists = np.sqrt(np.sum((X_train - x) ** 2, axis=1))

        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_train[nn_idx].astype(int)

        if weights == 'distance':
            # weight by inverse distance (add small eps)
            eps = 1e-8
            nn_dists = dists[nn_idx]
            w = 1.0 / (nn_dists + eps)
            # sum weights per class
            classes = np.unique(nn_labels)
            score = {}
            for c in classes:
                score[c] = w[nn_labels == c].sum()
            # pick class with highest weighted score
            preds.append(max(score.items(), key=lambda t: t[1])[0])
        else:
            vote = np.bincount(nn_labels)
            preds.append(np.argmax(vote))
    return np.array(preds)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def f1_score_simple(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def stratified_kfold_indices(y, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    folds = [[] for _ in range(n_splits)]
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, n_splits)
        for i in range(n_splits):
            folds[i].extend(parts[i].tolist())
    
    return [np.array(f) for f in folds]


def tune_knn(X, y, k_values, cv_folds=5, p_values=[2], weights_options=[None, 'distance']):
    folds = stratified_kfold_indices(y, n_splits=cv_folds, seed=42)
    best = None
    results = []
    for k in k_values:
        for p in p_values:
            for w in weights_options:
                f1s = []
                for i in range(cv_folds):
                    test_idx = folds[i]
                    train_idx = np.hstack([folds[j] for j in range(cv_folds) if j != i])
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    y_pred = knn_predict(X_train, y_train, X_test, k=k, p=p, weights=w)
                    f1s.append(f1_score_simple(y_test, y_pred))
                mean_f1 = np.mean(f1s)
                results.append(((k, p, w), mean_f1))
                if best is None or mean_f1 > best[1]:
                    best = ((k, p, w), mean_f1)
    return best, results



def evaluate_knn_cv(X, y, k_values, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print("\nK | Mean Acc | Mean F1")
    print("---------------------")
    results = []
    for k in k_values:
        model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        results.append((k, acc_scores.mean(), f1_scores.mean()))
        print(f"{k:2d} |   {acc_scores.mean():.4f}  |  {f1_scores.mean():.4f}")

    return results


def grid_search_knn(X, y):
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    param_grid = {
        "knn__n_neighbors": list(range(1, 31)),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=1, verbose=0)
    gs.fit(X, y)
    return gs


def final_evaluation(X_train, X_test, y_train, y_test, best_params):
    model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(**best_params))])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nFinal evaluation on holdout set:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def main():
    df = load_and_engineer()
    X = build_feature_matrix(df, feature_plan)
    y = df["target_is_successful"].to_numpy()

    # quick CV sweep for k
    k_values = list(range(1, 31))
    evaluate_knn_cv(X, y, k_values, cv_folds=5)

    # grid search for best hyperparameters
    print("\nRunning GridSearchCV to optimize KNN hyperparameters...")
    gs = grid_search_knn(X, y)
    print("Best params:", gs.best_params_)
    print("Best CV score:", gs.best_score_)

    # final holdout evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_knn_params = {k.replace("knn__", ""): v for k, v in gs.best_params_.items()}
    final_evaluation(X_train, X_test, y_train, y_test, best_knn_params)


if __name__ == "__main__":
    main()
