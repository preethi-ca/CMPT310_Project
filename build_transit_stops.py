import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

DEFAULT_GTFS_PATH = "google_transit.zip"


def read_stops_from_path(source_path: Path) -> pd.DataFrame:
	if source_path.is_dir():
		stops_path = source_path / "stops.txt"
		if not stops_path.exists():
			raise FileNotFoundError(f"{stops_path} does not exist.")
		return pd.read_csv(stops_path)

	if source_path.suffix.lower() != ".zip":
		raise ValueError("GTFS source must be a directory or a .zip archive.")

	with ZipFile(source_path) as archive:
		if "stops.txt" not in archive.namelist():
			raise FileNotFoundError("stops.txt was not found in the GTFS archive.")
		with archive.open("stops.txt") as stops_file:
			return pd.read_csv(stops_file)


def load_stops_from_gtfs(source_path: str) -> pd.DataFrame:
	stops_df = read_stops_from_path(Path(source_path))

	latitude_column = "stop_lat" if "stop_lat" in stops_df.columns else "latitude"
	longitude_column = "stop_lon" if "stop_lon" in stops_df.columns else "longitude"

	if latitude_column not in stops_df.columns or longitude_column not in stops_df.columns:
		raise ValueError("stops.txt must contain stop_lat/stop_lon or latitude/longitude columns.")

	transit_df = stops_df[[latitude_column, longitude_column]].rename(
		columns={latitude_column: "latitude", longitude_column: "longitude"}
	)
	transit_df = transit_df.dropna(subset=["latitude", "longitude"])
	return transit_df.drop_duplicates().reset_index(drop=True)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Build transit_stops.csv from a local GTFS stops.txt source."
	)
	parser.add_argument(
		"source",
		nargs="?",
		default=DEFAULT_GTFS_PATH,
		help="GTFS directory or .zip archive (default: google_transit.zip)",
	)
	parser.add_argument(
		"-o",
		"--output",
		default="transit_stops.csv",
		help="Output CSV path (default: transit_stops.csv)",
	)
	args = parser.parse_args()

	output_path = Path(args.output)
	transit_df = load_stops_from_gtfs(args.source)

	if transit_df.empty:
		raise ValueError(
			"No valid transit stop coordinates were found in the GTFS source."
		)

	transit_df.to_csv(output_path, index=False)
	print(f"Wrote {len(transit_df)} transit stops to {output_path}")


if __name__ == "__main__":
	main()