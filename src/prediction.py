import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import box
import numpy as np

DEFAULT_PROJECTED_CRS = "3857"  # note, not ideal for accuracy


class GridPred:
    def __init__(
        self,
        input_crime_data,
        crime_time_variable,
        input_features_data=None,
        features_names_variable=None,
        input_study_region=None,
        input_crs=None,
    ):

        # input crs
        self.input_crs = input_crs

        # load crime input data points
        # should contain long-lat and a date variable
        self.crime_points = self._raw_points_from_tabular(input_crime_data)

        # load optional predictor features
        self.features_points = None
        if input_features_data is not None:
            self.features_points = self._raw_points_from_tabular(input_features_data)

        # load optional study region
        self.study_region = None
        if input_study_region is not None:
            if isinstance(input_study_region, gpd.GeoDataFrame):
                self.study_region = input_study_region
            else:
                self.study_region = gpd.read_file(input_study_region)

        # variable names for time & features
        self.timevar = crime_time_variable
        self.features_var = features_names_variable

        # init model placeholders
        self.region_grid = None
        self.X = None
        self.y = None
        self.eval = None

    def _raw_points_from_tabular(self, data, fields_to_lower=True):
        """
        Accepts either:
        - a file path to a CSV
        - a pandas DataFrame

        Returns a pandas DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, str):
            df = pd.read_csv(data)
        else:
            raise TypeError("Input must be a pandas DataFrame or a path to a CSV file.")

        if fields_to_lower:
            df.columns = [c.lower() for c in df.columns]

        return df

    def _contains_long_lat(self, listcols) -> bool:
        "Check if long-lat colums are present in input csv"
        lower_cols = [item.lower() for item in listcols]
        is_latitude_present = "latitude" in lower_cols
        is_longitude_present = "longitude" in lower_cols

        return is_latitude_present and is_longitude_present

    def _count_point_in_polygon(self, points_gdf, polygon_gdf):
        joined = gpd.sjoin(points_gdf, polygon_gdf, predicate="within")
        counts = joined.groupby("index_right").size()

        return counts

    def _nearest_point_distance(self, points_gdf, polygon_gdf):
        # Extract coordinates into numpy arrays for vectorization
        g_centroids = polygon_gdf.geometry.centroid
        src_coords = np.column_stack((g_centroids.x, g_centroids.y))
        tgt_coords = np.column_stack((points_gdf.geometry.x, points_gdf.geometry.y))

        # Build a KDTree for fast nearest-neighbor lookup
        tree = cKDTree(tgt_coords)

        # Query the tree: k=1 returns the nearest neighbor
        # distances is a numpy array of float distances
        distances, _ = tree.query(src_coords, k=1)

        return pd.Series(distances, index=polygon_gdf.index)

    def _get_grid_centroid_xy(self, grid_polygon):
        return grid_polygon.centroid.get_coordinates()

    def _validate_df(self, df, required=False):
        """Validate a pandas DataFrame input."""
        if df is None:
            if required:
                raise ValueError("Required input is missing.")
            return None
        if not hasattr(df, "empty"):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty and required:
            raise ValueError("Required input DataFrame is empty.")
        return df

    def load_and_process_inputs(
        self,
        points_file,
        features_file=None,
        region_file=None,
        do_projection=False,
        projected_crs=None,
    ):

        try:
            points = self._validate_df(points_file, required=True)
            points_spatial = self.input_points_to_spatial(points)

            # features (CSV)
            features = (
                self.input_points_to_spatial(features_file)
                if features_file is not None
                else None
            )

            # region (shapefile)
            region = None
            if region_file is not None:
                if isinstance(region_file, gpd.GeoDataFrame):
                    region = region_file
                else:
                    region = gpd.read_file(region_file)

        except Exception as e:
            raise RuntimeError(f"Error loading input files: {e}")

        # Handle projection
        if do_projection:
            if projected_crs is None:
                if self.study_region is not None and self.study_region.crs is not None:
                    projected_crs = self.study_region.crs
                    print(
                        f"No projected CRS provided — adopting region CRS {projected_crs}"
                    )
                else:
                    print(
                        f"No projected CRS provided — defaulting to {DEFAULT_PROJECTED_CRS}"
                    )
                    projected_crs = DEFAULT_PROJECTED_CRS

        return points_spatial, features, region

    def input_points_to_spatial(self, input_points):
        """
        Converts a pandas DataFrame (input_points) with latitude and longitude
        values to a GeoDataFrame (spatial points object).
        """

        try:
            if not self._contains_long_lat(input_points.columns.tolist()):
                print("Necessary columns 'latitude' and 'longitude' not found.")
                return None

            # Check for NA values in spatial fields, drop
            input_points = input_points.dropna(subset=["latitude", "longitude"]).copy()

            input_gdf = gpd.GeoDataFrame(
                input_points,
                geometry=gpd.points_from_xy(
                    input_points.longitude, input_points.latitude
                ),
                crs=self.input_crs,
            )
            return input_gdf

        except Exception as e:
            print(f"Error converting points to GeoDataFrame: {e}")
            return None

    def create_grid(self, polygon_gdf, cell_size):
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
        coords_xy = self._get_grid_centroid_xy(clipped_grid)
        clipped_grid["x"] = coords_xy["x"]
        clipped_grid["y"] = coords_xy["y"]

        return clipped_grid

    def prepare_data(
        self, grid_cell_size, do_projection=True, projected_crs=None, export_raw=False
    ):

        # Handle projection logic cleanly
        if not do_projection:
            projected_crs = None  # explicitly disable projection
        elif projected_crs is None:
            print(f"No projected CRS provided — defaulting to {DEFAULT_PROJECTED_CRS}")
            projected_crs = DEFAULT_PROJECTED_CRS

        # do pre-processing on provided features
        points_spatial, features_spatial, study_region = self.load_and_process_inputs(
            self.crime_points,
            self.features_points,
            self.study_region,
            do_projection,
            projected_crs,
        )

        # Reproject points if projection is enabled and a projected_crs is set
        if projected_crs is not None:
            if points_spatial.crs != projected_crs:
                points_spatial = points_spatial.to_crs(projected_crs)

            if features_spatial is not None and features_spatial.crs != projected_crs:
                features_spatial = features_spatial.to_crs(projected_crs)

            # Ensure region is also in the projected CRS (if it wasn't already loaded in it)
            if study_region is not None and study_region.crs != projected_crs:
                study_region = study_region.to_crs(projected_crs)

        # If we didn't get a spatial boundary object, use convex hull of all points
        if study_region is None:
            hull = points_spatial.geometry.union_all().convex_hull
            study_region = gpd.GeoDataFrame(geometry=[hull], crs=points_spatial.crs)

        # define the region grid, clip input points to boundry
        self.region_grid = self.create_grid(study_region, grid_cell_size)
        clipped_points = gpd.clip(points_spatial, study_region)

        # optional, if risk features are present, also clip
        if self.features_points is not None and not self.features_points.empty:
            clipped_features = gpd.clip(features_spatial, study_region)

        # --------- Add Features ---------#

        # then functionality to perform grid counts of outcome variable events
        # and add spatial risk factors
        unique_times = sorted(clipped_points[self.timevar].unique().tolist())

        for time in unique_times:
            polygon_counts = self._count_point_in_polygon(
                clipped_points[clipped_points[self.timevar] == time], self.region_grid
            )
            self.region_grid[f"events_{time}"] = (
                self.region_grid.index.map(polygon_counts).fillna(0).astype(int)
            )

        # check that we have at least three time periods
        # train, test, validation
        num_time_periods = len(unique_times)
        if num_time_periods < 3:
            raise ValueError(
                f"Insufficient time periods ({num_time_periods}) for train/test/eval split."
            )

        time_fit = unique_times[0 : num_time_periods - 2]
        time_test = unique_times[num_time_periods - 2]
        time_eval = unique_times[num_time_periods - 1]

        # if named features variables are provided, identify distance to grid cells
        unique_types = []
        if (
            self.features_var
            and self.features_points is not None
            and not self.features_points.empty
        ):
            unique_types = clipped_features[self.features_var].unique().tolist()
            for type in unique_types:
                feature_dist = self._nearest_point_distance(
                    clipped_features[clipped_features[self.features_var] == type],
                    self.region_grid,
                )
                self.region_grid[type] = feature_dist

        # Assuming unique_types contains the risk factors (e.g., 'gas_station', 'bar')
        crime_pred_features = [f"events_{t}" for t in time_fit]
        pred_features = crime_pred_features + unique_types + ["x", "y"]

        target = f"events_{time_test}"
        eval_target = f"events_{time_eval}"

        self.X = self.region_grid[pred_features]
        self.y = self.region_grid[target]
        self.eval = self.region_grid[eval_target]
