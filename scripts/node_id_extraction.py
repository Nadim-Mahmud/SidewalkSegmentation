import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
import rasterio
from tqdm import tqdm

# Directory containing GeoTIFF files
input_dir = "../data/manhattan/images"
output_dir = "../data/manhattan/node/ids"
os.makedirs(output_dir, exist_ok=True)

SIDEWALK_FILTER = '["highway"~"footway|path|pedestrian|steps"]["foot"!="no"]'

# Process each GeoTIFF file in the directory
tif_files = list(Path(input_dir).glob("*.tif"))
for tif_file in tqdm(tif_files, desc="Extracting node ids"):
    print(f"Extracting node ids for {tif_file.name}")
    with rasterio.open(tif_file) as src:
        crs = src.crs
        bounds = src.bounds

        # Create bounding box geometry
        bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox_geom]}, crs=crs)
        bbox_gdf = bbox_gdf.to_crs(epsg=4326)  # Convert to WGS84 for OSMnx

        # Get the geometry in WGS84
        geom_wgs = bbox_gdf.iloc[0].geometry

        # Download and clip the street network using polygon
        G = ox.graph_from_polygon(
            geom_wgs,
            simplify=False,
            retain_all=True,           # keep disconnected sidewalk fragments
            custom_filter=SIDEWALK_FILTER
        )

        # Extract node IDs within the GeoTIFF extent
        node_ids = list(G.nodes)

        # Save node IDs to a text file, one per line
        output_file = os.path.join(output_dir, f"nodes_{tif_file.stem}.txt")
        with open(output_file, "w") as f:
            for node_id in node_ids:
                f.write(f"{node_id}\n")

        print(f"Node IDs for {tif_file.name} saved to {output_file}")