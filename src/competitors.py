import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree as bt

DATA_PATH = "data/location-information.csv"
OUTPUT_PATH = "data/location-information-with-competitors.csv"
TRANSIT_PATH = "data/transit_stops.csv"
EARTH_RADIUS = 6371000  # in meters

# load data from csv (latitude and longitude)
comp_df = pd.read_csv(DATA_PATH).dropna(subset=["latitude", "longitude"])
if comp_df.empty:
	raise ValueError(
		f"Data from {DATA_PATH} does not contain any valid latitude/longitude rows."
	)

transit_df = pd.read_csv(TRANSIT_PATH).dropna(subset=["latitude", "longitude"])
if transit_df.empty:
	raise ValueError(
		f"{TRANSIT_PATH} does not contain any valid latitude/longitude rows."
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

comp_df.to_csv(OUTPUT_PATH, index=False)

print(f"Competitor count and nearest transit stop distance calculated and saved to {OUTPUT_PATH}")
