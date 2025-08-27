import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
import rasterio

# Directory containing GeoTIFF files
input_dir = "/home/hanew/your_project_folder/omniacc/data/tifs/manhattan/images"
output_dir = "/home/hanew/your_project_folder/omniacc/data/tifs/manhattan/intersection_nodes/ids"
os.makedirs(output_dir, exist_ok=True)

# --- Choose ONE of these filters depending on what you need ---

# (A) Sidewalks/paths (your original)
SIDEWALK_FILTER = '["highway"~"footway|path|pedestrian|steps"]["foot"!="no"]'

# (B) Motor-vehicle roads only (typical “road intersections”)
ROAD_FILTER = (
    '["highway"]["highway"!~"footway|path|pedestrian|steps|cycleway|'
    'construction|proposed|raceway|bridleway|corridor|escape|bus_guideway|'
    'busway|service|track"]'
)

# (C) If you want both drivable + residential/local streets but not sidewalks:
# ROAD_FILTER = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street"]'

# Which filter to use:
CUSTOM_FILTER = ROAD_FILTER  # or SIDEWALK_FILTER

# Minimum number of distinct streets meeting to count as an intersection
MIN_STREETS = 2  # use 3 to require 3-way+ junctions

for tif_file in Path(input_dir).glob("*.tif"):
    print(f"Extracting node ids for {tif_file.name}")
    with rasterio.open(tif_file) as src:
        crs = src.crs
        bounds = src.bounds

        # Build bbox polygon in the raster CRS, then project to WGS84 for OSMnx
        bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox_geom]}, crs=crs).to_crs(epsg=4326)
        geom_wgs = bbox_gdf.iloc[0].geometry

        # 1) Download the network with your chosen filter
        # - simplify=True is usually best for street-count logic
        G = ox.graph_from_polygon(
            geom_wgs,
            simplify=True,
            retain_all=True,
            custom_filter=CUSTOM_FILTER,
        )

        # 2) (Optional but recommended) consolidate very close nodes so one junction
        # isn’t split into multiple near-duplicate nodes. Adjust tolerance as needed (meters).
        # Only do this if your AOI is small; can be expensive on big graphs.
        # from osmnx.simplification import consolidate_intersections
        # G = ox.consolidate_intersections(G, tolerance=10, rebuild_graph=True)

        # 3) Compute # of streets per node and pick intersections
        streets_per_node = ox.stats.count_streets_per_node(G)
        intersection_node_ids = [n for n, k in streets_per_node.items() if k >= MIN_STREETS]

        # If instead you want nodes like traffic lights/crossings from tags,
        # you could use:
        # poi = ox.features_from_polygon(
        #     geom_wgs,
        #     tags={"highway": ["traffic_signals", "stop", "crossing"]}
        # )
        # intersection_node_ids = list(poi.index)  # these are OSM node IDs

        # 4) Save node IDs
        output_file = os.path.join(output_dir, f"nodes_{tif_file.stem}.txt")
        with open(output_file, "w") as f:
            for node_id in intersection_node_ids:
                f.write(f"{node_id}\n")

        print(f"Found {len(intersection_node_ids)} intersections. Saved to {output_file}")
