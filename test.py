import pandas as pd

businesses_df = pd.read_csv("test550.csv")

successful = (businesses_df["target_is_successful"] == 1).sum()
unsuccessful = (businesses_df["target_is_successful"] == 0).sum()

print(f"Successful restaurants: {successful}")
print(f"Unsuccessful restaurants: {unsuccessful}")