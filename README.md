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

## Future Enhancements

- [ ] Add additional ML models (XGBoost, LightGBM, etc.)
- [ ] Add SHAP explainers
- [x] Spatial visualization of model outputs
- [x] Automated scoring and metrics output (PAI, PEI, RRI)

