import pandas as pd

# load the dataset
df = pd.read_csv("location-information.csv")

df = df.drop(columns=['zip_code'])
# print(df)

df.to_csv("location-information.csv", index=False)