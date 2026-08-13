import pandas as pd

# load the dataset
df = pd.read_csv("data/price_level_update.csv")

successful = (df["target_is_successful"] == 1).sum()
unsuccessful = (df["target_is_successful"] == 0).sum()

print(f"Successful restaurants: {successful}")
print(f"Unsuccessful restaurants: {unsuccessful}")
