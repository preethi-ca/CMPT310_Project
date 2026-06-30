import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree as bt

# load data from csv (latitude and longitude)
df = pd.read_csv("milestone1.csv").dropna(subset=["latitude", "longitude"])

# ball tree needs radian so convert to radian
coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy())

# magic radius calculacation
tree = bt(coords_rad, metric="haversine")

EARTH_RADIUS = 6371000 # in meters
r = 500 / EARTH_RADIUS # 500 meters

# Count neighbors within 500 meters
neighbors = tree.query_radius(coords_rad, r=r, count_only=True)

df["competitor_count_500m"] = neighbors - 1 # subtract 1 to exclude the store itself

df.to_csv("milestone1_with_competitors.csv", index=False)
