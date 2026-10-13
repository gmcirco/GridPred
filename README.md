# GridPred — Spatial Grid-Based Crime Prediction

GridPred is a framework for turning point-level crime data into grid-based predictive features.  
It handles:

- Converting latitude/longitude to spatial points  
- Generating grid cells over a study region
- Aggregating crime counts over time
- Calculating nearest-distance-to-feature variables
- Training a basic crime forecasting model (Random Forest Poisson regressor)

Outputs include predictions and evaluation-period observed crime values for each grid cell.

---

### Quickstart (Makefile Usage)

You can quickly install all dependencies by invoking `install` from the makefile:

```bash
make install
```

| Command            | Description                                                     |
|-------------------|-----------------------------------------------------------------|
| `make run`        | Run full workflow with crime CSV, region shapefile, and predictor locations |
| `make run-noregion` | Run with convex hull region (no shapefile provided)              |
| `make run-nopreds`  | Run without spatial predictor features                           |
| `make clean`        | Remove cache + logs                                             |
| `make install`      | Install dependencies via `uv`                                   |

Demo data from Hartford, CT are provided in input for testing purposes.

### OpenStreetMap basemap

The prediction plot does not request OpenStreetMap tiles by default. To include
an OSM basemap in a Makefile run, supply a User-Agent that identifies your
application:

```bash
make run OSM_USER_AGENT='YourApp/1.0 (+https://example.org)'
```

The same option works with `run-noregion` and `run-nopreds`. When calling
`main.py` directly, add `--use-osm --osm-user-agent 'YourApp/1.0 (+https://example.org)'`
to your existing command. When using the plotting function from Python:

```python
from gridpred.plotting import visualize_predictions

visualize_predictions(
    grid, predictions,
    use_osm=True,
    osm_user_agent="YourApp/1.0 (+https://example.org)",
)
```

Use your own application name and contact URL, following the
[OpenStreetMap tile usage policy](https://operations.osmfoundation.org/policies/tiles/).

## Future Enhancements

- [ ] Add additional ML models (XGBoost, LightGBM, etc.)
- [x] Spatial visualization of model outputs
- [x] Automated scoring and metrics output (PAI, PEI, RRI)

