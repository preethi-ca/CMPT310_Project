import pandas as pd

restaurants = pd.read_csv("milestone1.csv")
demographics = pd.read_csv("city_demographics.csv")


merged = restaurants.merge(
    demographics,
    on="city",
    how="left"
)


missing = merged[merged["median_income"].isna()]

if len(missing) > 0:
    print("error")
    print(missing["city"].unique())


merged.to_csv(
    "milestone1_modified.csv",
    index=False
)