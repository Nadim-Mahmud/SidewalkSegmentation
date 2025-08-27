import os
import math
import json
import pickle
import numpy as np
import networkx as nx
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import LineString, Point, MultiLineString, Polygon
from shapely.ops import unary_union, transform as shapely_transform
import pyproj
import overpy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from GeotiffMapper_SW import GeotiffMapper

'''
Script for creating B/W masks of the OSM graph data 

'''


def create_road_mask(geotiff_path, output_mask_path,
                     spacing_meters=10.0, line_width=1):
    """
    Creates a binary mask GeoTIFF (roads=255, background=0) for a given GeoTIFF,
    preserving all original metadata (CRS, transform, tags, etc).
    """
    # 1. Load original & metadata
    with rasterio.open(geotiff_path) as src:
        meta = src.meta.copy()  # grabs driver, dtype, count, crs, transform, etc
        width, height = src.width, src.height

        # Instantiate mapper & build graph
        mapper = GeotiffMapper(geotiff_path)
        mapper.get_geotiff_bounds()
        ways = mapper.get_osm_roads()
        G = mapper.create_road_graph(ways, spacing_meters=spacing_meters)

    # 2. Create blank mask with PIL
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    # 3. Draw each edge
    with rasterio.open(geotiff_path) as src:
        for u, v, data in G.edges(data=True):
            # map sidewalk→1, crossing→2, else→0 (or another class)
            ft = data.get('footway', '').lower()
            if ft == 'sidewalk':
                fill = 'green'
            elif ft == 'crossing':
                fill = 'red'
            else:
                # skip or assign another label
                continue

            try:
                row_u, col_u = src.index(u[0], u[1])
                row_v, col_v = src.index(v[0], v[1])
            except Exception:
                continue

            draw.line([(col_u, row_u), (col_v, row_v)],
                      fill=fill, width=line_width)

    # 4. Convert to NumPy and write GeoTIFF
    mask_arr = np.array(mask_img, dtype=np.uint8)

    # Update metadata: single band, uint8 mask
    meta.update({
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 1,
        'compress': 'lzw'            # optional: compression
    })

    with rasterio.open(output_mask_path, 'w', **meta) as dst:
        dst.write(mask_arr, 1)
        # Optionally copy over any additional tags:
        dst.update_tags(**src.tags())

    print(f"Saved mask GeoTIFF (with metadata) to {output_mask_path}")


def process_directory(input_dir, output_dir, spacing_meters=10.0, line_width=1):
    """
    Processes all TIFF files in the input directory to create road masks.
    
    Args:
        input_dir (str): Directory containing input geotiff files.
        output_dir (str): Directory to save the output JPEG mask files.
        spacing_meters (float): Spacing for segmentization.
        line_width (int): Thickness of road lines in the mask.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each .tif file
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, base_name + "_mask.tif")
            
            print(f"Processing {input_path} ...")
            try:
                create_road_mask(input_path, output_path, spacing_meters, line_width)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate BW rasterized road network masks from geotiff images."
    )
    parser.add_argument(
        "--input_dir", default='/home/hanew/your_project_folder/omniacc/data/tifs', help="Path to the input directory containing TIFF files."
    )
    parser.add_argument(
        "--output_dir", default = '/home/hanew/your_project_folder/omniacc/data/gt_masks', help="Path to the output directory to save JPG masks."
    )
    parser.add_argument(
        "--spacing", type=float, default=10.0, help="Spacing in meters for segmentizing road segments."
    )
    parser.add_argument(
        "--line_width", type=int, default=5, help="Line width (in pixels) for drawing roads on the mask."
    )
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.spacing, args.line_width)
