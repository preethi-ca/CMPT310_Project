import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree as bt


EARTH_RADIUS = 6371000  # in meters

# load data from csv (latitude and longitude)
comp_df = pd.read_csv("milestone1.csv").dropna(subset=["latitude", "longitude"])
if comp_df.empty:
	raise ValueError(
		f"milestone1.csv does not contain any valid latitude/longitude rows."
	)

transit_df = pd.read_csv("transit_stops.csv").dropna(subset=["latitude", "longitude"])
if transit_df.empty:
	raise ValueError(
		f"transit_stops.csv does not contain any valid latitude/longitude rows."
	)

# ball tree needs radian so convert to radian
coords_rad = np.radians(comp_df[["latitude", "longitude"]].to_numpy())
transit_coords_rad = np.radians(transit_df[["latitude", "longitude"]].to_numpy())

# magic radius calculacation
tree = bt(coords_rad, metric="haversine")
transit_tree = bt(transit_coords_rad, metric="haversine")
r = 500 / EARTH_RADIUS # 500 meters

# Count neighbors within 500 meters
neighbors = tree.query_radius(coords_rad, r=r, count_only=True)

comp_df["competitor_count_500m"] = neighbors - 1 # subtract 1 to exclude the store itself

# Nearest transit stop distance in meters
transit_distances_rad, _ = transit_tree.query(coords_rad, k=1)
comp_df["nearest_transit_distance_m"] = transit_distances_rad[:, 0] * EARTH_RADIUS

comp_df.to_csv("milestone1_with_competitors.csv", index=False)

print("Competitor count and nearest transit stop distance calculated and saved to milestone1_with_competitors.csv")
