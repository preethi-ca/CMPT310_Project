import numpy as np

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