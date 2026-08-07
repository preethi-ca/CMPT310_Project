import pandas as pd

# read the CSV
businesses_df = pd.read_csv("550-info.csv")

# redefine success using a 4.2 rating threshold
businesses_df["target_is_successful"] = (businesses_df["target_rating"] > 4.1).astype(int)

# save the updated CSV
businesses_df.to_csv("test-4.2.csv", index=False)

# validate
print(businesses_df["target_is_successful"].value_counts())