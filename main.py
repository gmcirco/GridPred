import pandas as pd
import argparse

from src.prediction import GridPred
from src.plotting import visualize_predictions
from sklearn.ensemble import RandomForestRegressor
from argparse import Namespace 
from pathlib import Path

def get_parser_args() -> Namespace:
    """
    Sets up and parses all command-line arguments for the crime prediction model.

    Returns:
        Namespace: An object containing all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the a GridPred crime prediction model."
    )

    # 1. Required positional: crime CSV path
    parser.add_argument(
        "input_file_path", type=str, help="Path to the input crime data CSV file."
    )

    # 2. Optional file inputs
    parser.add_argument(
        "--input_features_path",
        type=str,
        help="Path to the input predictive features CSV file.",
        default=None,
    )
    parser.add_argument(
        "--input_region_path",
        type=str,
        help="Path to the study region shapefile.",
        default=None,
    )

    # 3. Required column names from the crime CSV
    parser.add_argument(
        "--crime_time_variable",
        type=str,
        required=True,
        help="Column representing the temporal crime variable in the input crime CSV.",
    )

    parser.add_argument(
        "--features_names_variable",
        type=str,
        help="Column representing feature names in the predictive features CSV file.",
    )

    parser.add_argument(
        "--input_crs",
        type=str,
        default="EPSG:4326",
        help="EPSG code for the input point CSV data (e.g., 'EPSG:3508').",
    )

    parser.add_argument(
        "--projected_crs",
        type=str,
        default=None,
        help="Projected CRS to use if --do_projection is set (e.g., 'EPSG:3857'). ",
    )

    parser.add_argument(
        "--do_projection",
        action="store_true",
        help="If set, project coordinates into a projected CRS.",
    )

    parser.add_argument(
        "--grid_size",
        type=int,
        default=300,
        help="Size of the grid cell units (e.g., 300 meters).",
    )

    return parser.parse_args()

def export_results(gridpred_model, preds, filename="preds.csv"):

    # set up data
    export_df = gridpred_model.X.copy()
    export_df["y_eval"] = gridpred_model.eval
    export_df["y_pred"] = preds
    export_df = export_df.reset_index().rename(columns={"index": "grid_cell_id"})

    # ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # export to CSV
    export_df.to_csv(output_dir / filename, index=False)

def main():

    # parse input args
    args = get_parser_args()

    # ---------------------------------- #
    # Do Gridpred data prep stuff here
    # ---------------------------------- #

    gridpred = GridPred(
        input_crime_data=args.input_file_path,
        input_features_data=args.input_features_path,
        input_study_region=args.input_region_path,
        crime_time_variable=args.crime_time_variable,
        features_names_variable=args.features_names_variable,
        input_crs=args.input_crs,
    )

    gridpred.prepare_data(
        grid_cell_size=args.grid_size,
        do_projection=args.do_projection,
        projected_crs=args.projected_crs,
    )

    # very basic demo model workflow
    # can replace with xgboost or whatever model
    X = gridpred.X
    y = gridpred.y

    rf = RandomForestRegressor(n_estimators=1000, criterion="poisson", random_state=42)
    rf.fit(X, y)

    # Predict
    y_pred = rf.predict(X)

    # export to output folder
    export_results(gridpred, y_pred)

    # print feature importances
    # TODO: in future, can be logged and plotted
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))

    # plotting
    region_grid = gridpred.region_grid
    visualize_predictions(region_grid, y_pred)



if __name__ == "__main__":
    main()
