import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box
from sklearn.ensemble import RandomForestRegressor

INPUT_FILE_PATH = "/home/gmcirco/Documents/Projects/data/crime_2019-2023_v2.csv"
INPUT_FEATURES_PATH = "/home/gmcirco/Documents/Projects/data/features_malmo.csv"
INPUT_REGION_PATH = (
    "/home/gmcirco/Documents/Projects/data/malmo_shapefiles/DeSo_Malmö.shp"
)

GRID_CELL_SIZE_METERS = 300
DEFAULT_CRS = "4326"
PROJECTED_CRS = "3006"


def _raw_points_from_csv(file_path, fields_to_lower=True):
    "Load csv input from a file path"
    df = pd.read_csv(file_path)
    if fields_to_lower:
        df.columns = [x.lower() for x in df.columns]
    return df


def _contains_long_lat(listcols) -> bool:
    "Check if long-lat colums are present in input csv"
    lower_cols = [item.lower() for item in listcols]
    is_latitude_present = "latitude" in lower_cols
    is_longitude_present = "longitude" in lower_cols

    return is_latitude_present and is_longitude_present


def _count_point_in_polygon(points_gdf, polygon_gdf):
    joined = gpd.sjoin(points_gdf, polygon_gdf, predicate="within")
    counts = joined.groupby("index_right").size()

    return counts


def _nearest_point_distance(points_gdf, polygon_gdf):
    centroids = polygon_gdf.geometry.centroid
    distances = centroids.apply(lambda g: points_gdf.distance(g).min())

    return distances


def _get_grid_centroid_xy(grid_polygon):
    return grid_polygon.centroid.get_coordinates()


def input_points_to_spatial(input_points):
    """
    Converts a pandas DataFrame (input_points) with latitude and longitude
    values to a GeoDataFrame (spatial points object).
    """

    try:
        if not _contains_long_lat(input_points.columns.tolist()):
            print("Necessary columns 'latitude' and 'longitude' not found.")
            return None

        input_gdf = gpd.GeoDataFrame(
            input_points,
            geometry=gpd.points_from_xy(input_points.longitude, input_points.latitude),
            crs=DEFAULT_CRS,
        )
        return input_gdf

    except Exception as e:
        print(f"Error converting points to GeoDataFrame: {e}")
        return None


def visualize(points_gdf, grid_gdf, region_gdf):
    """Quick visualization of crime points, grid, and region boundary."""
    _, ax = plt.subplots(figsize=(8, 8))
    region_gdf.plot(ax=ax, color="none", edgecolor="black", linewidth=1)
    grid_gdf.plot(ax=ax, color="none", edgecolor="lightgray")
    points_gdf.plot(ax=ax, color="red", markersize=5, alpha=0.5)
    plt.title("Crime Points and Grid")
    plt.show()


def visualize_grid_by_feature(gdf, feature):
    """
    Plots a GeoDataFrame grid colored by a single numeric feature.

    Parameters:
    - gdf: GeoDataFrame with a geometry column.
    - feature: str, the column name to color by.
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(
        column=feature, ax=ax, cmap="viridis", legend=True, edgecolor="k", linewidth=0.2
    )
    ax.set_axis_off()
    ax.set_title(f"Grid colored by {feature}", fontsize=14)
    plt.show()


def visualize_predictions(gdf, predictions, title="", cmap="viridis"):
    """
    Plots a GeoDataFrame grid colored by predictions.

    Parameters:
    - gdf: GeoDataFrame with a geometry column.
    - predictions: array-like of predicted values, same length as gdf.
    - title: plot title.
    - cmap: colormap for the values.
    """
    gdf = gdf.copy()
    gdf["pred"] = predictions

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(column="pred", ax=ax, cmap=cmap, legend=True, edgecolor="k", linewidth=0.2)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14)
    plt.show()


def visualize_predictions_osm(
    gdf, predictions, title="", cmap="viridis", use_osm=True, zoom=12
):
    """
    Plots a GeoDataFrame grid colored by predictions, optionally overlaid on an OpenStreetMap basemap.

    Parameters:
    - gdf: GeoDataFrame with a geometry column.
    - predictions: array-like of predicted values, same length as gdf.
    - title: str, plot title.
    - cmap: colormap for the values.
    - use_osm: bool, whether to include an OpenStreetMap basemap.
    - zoom: int, zoom level for the OSM tiles.
    """
    gdf = gdf.copy()
    gdf["pred"] = predictions

    # Ensure the GeoDataFrame is in the correct CRS for OSM (EPSG:3857)
    if gdf.crs is None or gdf.crs.to_string() != "EPSG:3857":
        gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(
        column="pred",
        ax=ax,
        cmap=cmap,
        legend=True,
        edgecolor="k",
        linewidth=0.2,
        alpha=0.7,
    )

    if use_osm:
        ctx.add_basemap(
            ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom
        )

    ax.set_axis_off()
    ax.set_title(title or "Predicted Crime Intensity", fontsize=14)
    plt.show()


def create_grid(polygon_gdf, cell_size):
    "Create a regular grid over a polygon shapefile, add x-y coordinates as output"

    minx, miny, maxx, maxy = polygon_gdf.total_bounds
    x_coords = list(range(int(minx), int(maxx) + cell_size, cell_size))
    y_coords = list(range(int(miny), int(maxy) + cell_size, cell_size))

    grid_cells = []
    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            # Create a box (grid cell) for the current coordinates
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))

    # 3. Create a GeoDataFrame from the grid cells
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=polygon_gdf.crs)

    # 4. Intersect the grid with the polygon to clip
    # The 'clip' function is an easier alternative to 'overlay' for simple clipping
    clipped_grid = gpd.clip(grid_gdf, polygon_gdf)

    # add xy-coords
    coords_xy = _get_grid_centroid_xy(clipped_grid)
    clipped_grid["x"] = coords_xy["x"]
    clipped_grid["y"] = coords_xy["y"]

    return clipped_grid


# for testing = load
# input should be either just points as a csv, or points csv + shapefile
df = _raw_points_from_csv(INPUT_FILE_PATH)
features = _raw_points_from_csv(INPUT_FEATURES_PATH)
region = gpd.read_file(INPUT_REGION_PATH)


def main(
    crime_points=df,
    features_points=features,
    study_region=region,
    do_projection=True,
    projected_crs=PROJECTED_CRS,
    export_raw=True,
):

    # first, take crime points and study region
    # convert to spatial, project, and define a study grid
    points_spatial = input_points_to_spatial(crime_points)
    features_spatial = input_points_to_spatial(features_points)

    if do_projection:
        points_spatial = points_spatial.to_crs(projected_crs)
        features_spatial = features_spatial.to_crs(projected_crs)
        study_region = study_region.to_crs(projected_crs)

    # define grid & clip points to grid
    region_grid = create_grid(study_region, GRID_CELL_SIZE_METERS)
    clipped_points = gpd.clip(points_spatial, study_region)
    clipped_features = gpd.clip(features_spatial, study_region)

    # --------- Add Features ---------#

    # then functionality to perform grid counts of outcome variable
    # and add spatial risk factors
    unique_times = clipped_points["yearvar"].unique()  # TODO: Set as an argparse param

    for time in unique_times:
        polygon_counts = _count_point_in_polygon(
            clipped_points[clipped_points["yearvar"] == time], region_grid
        )
        region_grid[f"crimes_{time}"] = (
            region_grid.index.map(polygon_counts).fillna(0).astype(int)
        )

    # distance from polygon to nearest risk factor
    unique_types = clipped_features["type"].unique()  # TODO: Set as an argparse param
    for type in unique_types:
        feature_dist = _nearest_point_distance(
            clipped_features[clipped_features["type"] == type], region_grid
        )
        region_grid[type] = feature_dist

    # !! TEST ONLY !!
    # try a basic random forest model
    # use 2020 - 2021 crimes as predictions
    # fit on 2022 crimes
    # use 2023 crimes as hold-out evaluation for model eval
    # !! TEST ONLY !!

    pred_features = [
        "crimes_2020",
        "crimes_2021",
        "gas_station",
        "bar",
        "liquor_store",
        "x",
        "y",
    ]
    target = "crimes_2022"
    eval_target = "crimes_2023"

    X = region_grid[pred_features]
    y = region_grid[target]

    rf = RandomForestRegressor(n_estimators=500, criterion="poisson", random_state=42)
    rf.fit(X, y)

    # Predict
    y_pred = rf.predict(X)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))

    visualize_predictions_osm(region_grid, y_pred)  # export pred map to osm

    # insert code to compute metrics like PAI, PEI, RRI, etc...

    # optional export as csv
    if export_raw:
        region_grid.to_csv("output/raw_input_features.csv")


if __name__ == "__main__":
    main()
