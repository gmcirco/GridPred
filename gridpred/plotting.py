import contextily as ctx
import matplotlib.pyplot as plt


def visualize(points_gdf, grid_gdf, region_gdf):
    """Quick visualization of incident points, grid, and region boundary."""
    _, ax = plt.subplots(figsize=(8, 8))
    region_gdf.plot(ax=ax, color="none", edgecolor="black", linewidth=1)
    grid_gdf.plot(ax=ax, color="none", edgecolor="lightgray")
    points_gdf.plot(ax=ax, color="red", markersize=5, alpha=0.5)
    plt.title("Incident Points and Grid")
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


def visualize_predictions(
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

    _, ax = plt.subplots(1, 1, figsize=(10, 10))
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
    ax.set_title(title or "Predicted Incident Intensity", fontsize=14)
    plt.show()
