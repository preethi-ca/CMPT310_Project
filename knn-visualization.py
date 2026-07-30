import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from knn_classification import (
    build_feature_matrix,
    evaluate_knn_cv,
    feature_plan,
    grid_search_knn,
    load_and_engineer,
)


def main():
    # -----------------------------
    # Load and preprocess dataset
    # -----------------------------
    df = load_and_engineer()

    X = build_feature_matrix(df, feature_plan)
    y = df["target_is_successful"].to_numpy()

    # -----------------------------
    # Accuracy and F1 vs k
    # -----------------------------
    k_values = list(range(1, 31))
    results = evaluate_knn_cv(X, y, k_values, cv_folds=5)

    accuracy_values = [result[1] for result in results]
    f1_values = [result[2] for result in results]

    # -----------------------------
    # Find best KNN parameters
    # -----------------------------
    grid_search = grid_search_knn(X, y)

    best_params = {
        key.replace("knn__", ""): value
        for key, value in grid_search.best_params_.items()
    }

    best_k = best_params["n_neighbors"]

    print("\nBest parameters:", best_params)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    # -----------------------------
    # KNN performance plot
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        accuracy_values,
        marker="o",
        linewidth=2,
        label="Accuracy",
    )

    plt.plot(
        k_values,
        f1_values,
        marker="o",
        linewidth=2,
        label="F1 Score",
    )

    best_accuracy = accuracy_values[best_k - 1]

    plt.scatter(
        best_k,
        best_accuracy,
        s=100,
        zorder=5,
        label=f"Selected k = {best_k}",
    )

    plt.annotate(
        f"k = {best_k}",
        (best_k, best_accuracy),
        xytext=(best_k + 1, best_accuracy + 0.005),
    )

    plt.xlabel("Number of Neighbours (k)")
    plt.ylabel("Mean Cross-Validation Score")
    plt.title("KNN Performance vs k")
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "knn_performance_vs_k.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -----------------------------
    # Train final KNN model
    # -----------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(**best_params)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Successful", "Successful"],
    )

    display.plot(cmap="Blues")
    plt.title("KNN Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "knn_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()