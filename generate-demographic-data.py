# Adds CensusMapper demographic information for each store

"""
calculates: median_income, pop_density_sqkm, pct_age_20_39
(median income is for individuals, not household)
(pop_density_sqkm uses nearby DAs to estimate combined population density)
(pct_age_20_39 represents the percentage of the population between 20 and 39 years old)
"""

"""
Data is from CensusMapper API, using Statistics Canada 2021 Census data
Dataset: CA21, region: Vancouver CMA (includes main Metro Vancouver areas),
         code 59933, geography level: Dissemination Area (DA)
"""

from pathlib import Path
import os
import geopandas
import pandas
import pycancensus


INPUT_FILE = Path("updated-columns.csv")
# input file needs the latitude/longitude columns
OUTPUT_FILE = Path("full_information.csv")
ENV_FILE = Path(".env")

# info for the api
CENSUS_DATASET = "CA21"
VANCOUVER_CMA_REGION = {"CMA": "59933"}
CENSUS_LEVEL = "DA"
RADIUS_METRES = 1000
PROJECTED_COORDINATE_SYSTEM = "EPSG:3347"


#  CensusMapper IDs needed
CENSUS_VECTORS = {
    "population_2021": "v_CA21_1",
    "pop_density_sqkm": "v_CA21_6",
    "median_income": "v_CA21_560",
    "age_20_24": "v_CA21_89",
    "age_25_29": "v_CA21_107",
    "age_30_34": "v_CA21_125",
    "age_35_39": "v_CA21_143",
}


OUTPUT_COLUMNS_TO_ADD = [
    "median_income",
    "pop_density_sqkm",
    "pct_age_20_39",
]


def load_environment_variables_from_dotenv():
    # gets CANCENSUS_API_KEY from .env 
    if not ENV_FILE.exists():
        return

    with ENV_FILE.open("r", encoding="utf-8") as env_file:
        for line in env_file:
            cleaned_line = line.strip()

            if cleaned_line == "":
                continue

            if cleaned_line.startswith("#"):
                continue

            if "=" not in cleaned_line:
                continue

            variable_name, variable_value = cleaned_line.split("=", 1)
            variable_name = variable_name.strip()
            variable_value = variable_value.strip().strip('"').strip("'")

            os.environ[variable_name] = variable_value


def set_censusmapper_api_key():
    # sets CensusMapper API key for pycancensus
    load_environment_variables_from_dotenv()

    api_key = os.environ.get("CANCENSUS_API_KEY")

    if api_key is not None and api_key.strip() != "":
        pycancensus.set_api_key(api_key.strip())
        return

    existing_api_key = pycancensus.get_api_key()

    if existing_api_key is not None:
        return

    raise ValueError(
        "missing API key"
    )

# finds the actual column name returned by pycancensus for a vector
def find_returned_column_for_vector(census_data, vector_name):
    if vector_name in census_data.columns:
        return vector_name

    for column in census_data.columns:
        column_text = str(column)

        if column_text.startswith(vector_name + ":"):
            return column

        if column_text.startswith(vector_name + " "):
            return column

        if column_text.endswith("." + vector_name):
            return column

    raise ValueError(
        f"missing column {vector_name}. "
        f"returned columns were {list(census_data.columns)}"
    )

# renames CensusMapper vector columns 
def rename_census_vector_columns(census_data):
    columns_to_rename = {}

    for feature_name, vector_name in CENSUS_VECTORS.items():
        returned_column = find_returned_column_for_vector(census_data, vector_name)
        columns_to_rename[returned_column] = feature_name

    return census_data.rename(columns=columns_to_rename)

#gets the census info by calling API
def download_census_data():
    set_censusmapper_api_key()

    census_data = pycancensus.get_census(
        dataset=CENSUS_DATASET,
        regions=VANCOUVER_CMA_REGION,
        vectors=list(CENSUS_VECTORS.values()),
        level=CENSUS_LEVEL,
        geo_format="geopandas",
        resolution="simplified",
        labels="short",
        use_cache=True,
        quiet=False,
    )

    census_data = rename_census_vector_columns(census_data)

    # EPSG:3347 uses metres
    census_data = census_data.to_crs(PROJECTED_COORDINATE_SYSTEM)
    census_data["centroid_geometry"] = census_data.geometry.centroid

    return census_data

# reads the input CSV and turns coordinates into map points
def load_input_rows():
    input_rows = pandas.read_csv(INPUT_FILE).copy()

    if "latitude" not in input_rows.columns:
        raise ValueError(f"{INPUT_FILE} latitude missing")

    if "longitude" not in input_rows.columns:
        raise ValueError(f"{INPUT_FILE} latitude missnig")

    existing_output_columns = []

    for column in OUTPUT_COLUMNS_TO_ADD:
        if column in input_rows.columns:
            existing_output_columns.append(column)

    if len(existing_output_columns) > 0:
        raise ValueError(
            f"{INPUT_FILE} already contains the columns"
        )

    input_rows["source_row_id"] = range(len(input_rows))
    input_rows_with_coordinates = input_rows.dropna(subset=["latitude", "longitude"])

    input_points = geopandas.GeoDataFrame(
        input_rows_with_coordinates,
        geometry=geopandas.points_from_xy(
            input_rows_with_coordinates["longitude"],
            input_rows_with_coordinates["latitude"],
        ),
        crs="EPSG:4326",
    )
    input_points = input_points.to_crs(PROJECTED_COORDINATE_SYSTEM)

    return input_rows, input_points


def sum_without_missing_values(values):
    values_without_missing = values.dropna()

    if values_without_missing.empty:
        return pandas.NA

    return values_without_missing.sum()


def average_without_missing_values(values):
    values_without_missing = values.dropna()

    if values_without_missing.empty:
        return pandas.NA

    return values_without_missing.mean()


def calculate_combined_population_density(nearby_areas):
    density_rows = nearby_areas[["population_2021", "pop_density_sqkm"]].dropna()
    density_rows = density_rows[density_rows["pop_density_sqkm"] > 0]

    if density_rows.empty:
        return pandas.NA

    total_population = density_rows["population_2021"].sum()
    estimated_area_sqkm = (
        density_rows["population_2021"] / density_rows["pop_density_sqkm"]
    ).sum()

    if estimated_area_sqkm == 0:
        return pandas.NA

    return total_population / estimated_area_sqkm


def calculate_percentage(numerator, denominator):
    if pandas.isna(numerator):
        return pandas.NA

    if pandas.isna(denominator):
        return pandas.NA

    if denominator == 0:
        return pandas.NA

    return (numerator / denominator) * 100


# calculates demographic features for one store coordinate
def calculate_features_for_one_point(point_geometry, census_data):
    distances_from_point = census_data["centroid_geometry"].distance(point_geometry)
    nearby_areas = census_data[distances_from_point <= RADIUS_METRES]

    if nearby_areas.empty:
        return {
            "median_income": pandas.NA,
            "pop_density_sqkm": pandas.NA,
            "pct_age_20_39": pandas.NA,
        }

    # population is used for age percentages
    population_2021 = sum_without_missing_values(nearby_areas["population_2021"])

    age_20_to_39 = sum_without_missing_values(
        nearby_areas["age_20_24"]
        + nearby_areas["age_25_29"]
        + nearby_areas["age_30_34"]
        + nearby_areas["age_35_39"]
    )

    return {
        "median_income": average_without_missing_values(
            nearby_areas["median_income"]
        ),
        "pop_density_sqkm": calculate_combined_population_density(nearby_areas),
        "pct_age_20_39": calculate_percentage(age_20_to_39, population_2021),
    }

# calculates demographic features for every row with coordinates
def calculate_features_for_all_points(input_points, census_data):
    feature_rows = []

    for _, input_point in input_points.iterrows():
        feature_values = calculate_features_for_one_point(
            input_point.geometry,
            census_data,
        )
        feature_values["source_row_id"] = input_point["source_row_id"]
        feature_rows.append(feature_values)

    features = pandas.DataFrame(feature_rows)

    for column in OUTPUT_COLUMNS_TO_ADD:
        if column in features.columns:
            features[column] = pandas.to_numeric(
                features[column],
                errors="coerce",
            ).round(2)

    return features


def main():
    input_rows, input_points = load_input_rows()
    census_data = download_census_data()
    census_features = calculate_features_for_all_points(input_points, census_data)

    output = input_rows.merge(census_features, on="source_row_id", how="left")
    output = output.drop(columns=["source_row_id"])
    output.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()