import contextily as ctx
import matplotlib.pyplot as plt


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


def visualize_predictions(
    gdf, predictions, title="", cmap="viridis", use_osm=False, zoom=12,
    osm_user_agent=None,
):
    """
    Plots a GeoDataFrame grid colored by predictions, optionally overlaid on an OpenStreetMap basemap.

    Parameters:
    - gdf: GeoDataFrame with a geometry column.
    - predictions: array-like of predicted values, same length as gdf.
    - title: str, plot title.
    - cmap: colormap for the values.
    - use_osm: bool, whether to include an OpenStreetMap basemap (default: False).
    - zoom: int, zoom level for the OSM tiles.
    - osm_user_agent: str, application identifier required when use_osm is True.
    """
    if use_osm and (not osm_user_agent or not osm_user_agent.strip()):
        raise ValueError("osm_user_agent is required when use_osm=True")

    gdf = gdf.copy()
    gdf["pred"] = predictions

    # OSM tiles use EPSG:3857; preserve the input CRS when no basemap is requested.
    if use_osm and (gdf.crs is None or gdf.crs.to_string() != "EPSG:3857"):
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
            ax,
            crs=gdf.crs,
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom=zoom,
            headers={"User-Agent": osm_user_agent},
        )

    ax.set_axis_off()
    ax.set_title(title or "Predicted Crime Intensity", fontsize=14)
    plt.show()
