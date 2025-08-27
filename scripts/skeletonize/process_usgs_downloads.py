import os
import sys
import subprocess
import glob
from turtle import st
from tqdm.auto import trange, tqdm
from dotenv import load_dotenv
import json
from scripts.GeotiffMapper_SW import GeotiffMapper
import rasterio
import numpy as np
from shapely import STRtree
import shapely
from tqdm.auto import tqdm
from shapely.geometry import shape, mapping, LineString, Point, box
from shapely.ops import unary_union
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform, transform_bounds, reproject, Resampling, transform_geom
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt
from skimage.color import hsv2rgb
from PIL import Image

load_dotenv()
PROJECT_ROOT = '/home/hanew/omniacc/'

DATA_ROOT = '/home/hanew/omniacc/data'
def tqdm_print(*args): 
    tqdm.write(' '.join([str(arg) for arg in args]))
print = tqdm_print

RAW_ZIPS_FOLDER = os.path.join(DATA_ROOT, 'raw')
PROCESSED_IMAGES_FOLDER = '/home/hanew/your_project_folder/omniacc/data/tifs/manhattan/images'
PROCESSED_GEOJSON_FOLDER = '/home/hanew/your_project_folder/omniacc/data/tifs/manhattan/geojsons'
ROAD_MASKS_FOLDER = '/home/hanew/your_project_folder/omniacc/data/tifs/manhattan/rasters' 

#TODO: This script needs some organization, it has configs/functions that we do not need for this project.
RASTERIZE_CONFIGS = {
    "roads_meters_valid": {
        "pixel_size_meters": 1.0,
        "distance_limit_meters": 200.0,
        "road_labels": 'valid',
        "multiclass": False,
        "base_tile_size": 1000,
        "progress_enabled": True,
        "scale_tag": '1m',
        "output_folder": os.path.join(DATA_ROOT, 'processed', 'masks', 'roads_meters'),
    },
    "roads_meters_all": {
        "pixel_size_meters": 1.0,
        "distance_limit_meters": 200.0,
        "road_labels": 'all',
        "multiclass": False,
        "base_tile_size": 1000,
        "progress_enabled": True,
        "scale_tag": '1m',
        "output_folder": os.path.join(DATA_ROOT, 'processed', 'masks', 'roads_meters_all'),
    },
    "roads_50cm_valid": {
        "pixel_size_meters": 0.5,
        "distance_limit_meters": 100.0,
        "road_labels": 'valid',
        "multiclass": False,
        "base_tile_size": 1000,
        "progress_enabled": True,
        "scale_tag": '50cm',
        "output_folder": os.path.join(DATA_ROOT, 'processed', 'masks', 'roads_50cm_valid'),
    }
}

DEFAULT_RASTERIZE_CONFIG = RASTERIZE_CONFIGS["roads_meters_all"]

# ROAD_MASKS_FOLDER = paths.ROAD_MASKS_FOLDER + "_meters_all"

# Rasterize all road types
os.environ.setdefault('ROAD_LABELS', 'all')

# Configuration for rasterization - set pixel size here or via environment
# Each field controls a specific aspect of the rasterization process:
# - pixel_size_meters: Target pixel resolution in meters (e.g., 1.0 for 1m, 0.5 for 50cm)
# - distance_limit_meters: Maximum distance for UV vector computation in meters, also used for tile overlap
# - road_labels: Which road types to include - 'all' (all OSM highway types) or 'valid' (major road types only)
# - multiclass: Whether to use multiclass labeling (True) or binary (False) for road types
# - base_tile_size: Interior size of processing tiles in pixels (before adding overlap margins)
# - progress_enabled: Whether to show progress bars and status messages during processing
# - scale_tag: Tag to use in output filenames (e.g., "50cm", "1m", "high-res")
RASTERIZE_CONFIG = {
    "default": {
        "pixel_size_meters": float(os.environ.get('PIXEL_SIZE_METERS', '1.0')),
        "distance_limit_meters": float(os.environ.get('DISTANCE_LIMIT_METERS', '200.0')),
        "road_labels": os.environ.get('ROAD_LABELS', 'all'),  # 'valid' or 'all'
        "multiclass": os.environ.get('MULTICLASS', 'True').lower() == 'true',
        "base_tile_size": int(os.environ.get('BASE_TILE_SIZE', '1000')),
        "progress_enabled": os.environ.get('PROGRESS_ENABLED', 'True').lower() == 'true',
        "scale_tag": os.environ.get('SCALE_TAG', '1m'),  # Simple tag for filenames
    }
}

ROAD_TAGS = {
    'background': 0,
    'motorway': 1,
    'trunk': 2,
    'primary': 3,
    'secondary': 4,
    'tertiary': 5,
    'unclassified': 6,
    'residential': 7,
    'motorway_link': 8,
    'trunk_link': 9,
    'primary_link': 10,
    'secondary_link': 11,
    'tertiary_link': 12,
    'living_street': 13,
    'service': 14,
    'pedestrian': 15,
    'track': 16,
    'bus_guideway': 17,
    'escape': 18,
    'raceway': 19,
    'road': 20,
    'busway': 21,
    'footway': 22,
    'bridleway': 23,
    'steps': 24,
    'corridor': 25,
    'path': 26,
    'via_ferrata': 27,
    'sidewalk': 28,
    'crossing': 29,
    'traffic_island': 30,
    'cycleway': 31,
    'construction': 32,
    'proposed': 33,
    # Additional tags (less common or newer)
    'elevator': 34,         # For vertical transportation
    'driveway': 35,         # Private driveways
    'alley': 36,            # Narrow urban streets
    'drive-through': 37,    # Roads for drive-through services
    'services': 38,         # Service areas on highways
    'rest_area': 39,        # Rest areas along major roads
}

VALID_ROAD_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "unclassified", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link"
}
class TiledRasterizer:
    def __init__(self, geojson_file, geotiff_file, output_roads_uv, output_inters_uv, 
                 config=None, disable_progress=False):
        """
        Initialize the TiledRasterizer.
        
        Args:
            geojson_file (str): Path to input GeoJSON file
            geotiff_file (str): Path to reference GeoTIFF file
            output_roads_uv (str): Path for roads UV output
            output_inters_uv (str): Path for intersections UV output
            config (dict, optional): Configuration dictionary
            disable_progress (bool): If True, disable progress output
        """
        self.geojson_file = geojson_file
        self.geotiff_file = geotiff_file
        self.output_roads_uv = output_roads_uv
        self.output_inters_uv = output_inters_uv
        self.disable_progress = disable_progress
        
        # Use provided config or get default
        if config is None:
            config = DEFAULT_RASTERIZE_CONFIG
        self.config = config
        
        # Extract configuration parameters
        self.distance_limit_m = config['distance_limit_meters']
        self.multiclass = config['multiclass']
        self.progress_enabled = config['progress_enabled'] and not disable_progress

        # Determine road types based on configuration
        if config['road_labels'] == 'valid':
            self.road_types = VALID_ROAD_TYPES
        elif config['road_labels'] == 'all':
            self.road_types = set(ROAD_TAGS.keys())
        else:
            self.road_types = VALID_ROAD_TYPES
            
        # Calculate distance limit in pixels based on GeoTIFF resolution
        with rasterio.open(geotiff_file) as src:
            pixel_size = src.res[0] * src.crs.linear_units_factor[1]
            self.distance_limit_p = int(np.ceil(self.distance_limit_m / pixel_size))
            # Get image bounds for filtering features
            self.image_bounds = src.bounds
            self.image_crs = src.crs

        # Total tile size including overlap on each side
        self.tile_full_size = config['base_tile_size'] + 2 * self.distance_limit_p

        # Load vector data
        with open(self.geojson_file) as f:
            self.geojson_data = json.load(f)

        # Build lists of vector features (roads) and line geometries (for intersections)
        self.road_features = []
        self.line_geometries = []
        self.line_labels = []
        
        # Create a bounding box for the image bounds to filter features
        image_bbox = box(*self.image_bounds)
        
        # Process features to find valid road geometries
        for feature in self.geojson_data.get("features", []):
            geom = shape(feature["geometry"])
            road_type = feature["properties"].get("highway", None)
            if isinstance(geom, LineString) and road_type in self.road_types:
                # Reproject geometry to image CRS for bounds checking
                geom_reproj = shape(transform_geom("EPSG:4326", self.image_crs.to_string(), mapping(geom)))
                
                # Check if the feature intersects the image bounds
                if geom_reproj.intersects(image_bbox):
                    self.road_features.append((geom, 255))
                    self.line_geometries.append(geom)
                    self.line_labels.append(ROAD_TAGS[road_type])

        # Check if we have any valid road features to rasterize
        if not self.road_features:
            if self.progress_enabled:
                tqdm_print(f"No valid road features found in {self.geojson_file}, skipping rasterization")
            self.skip_processing = True
            return
        else:
            self.skip_processing = False
        
        # Precompute intersections (unique endpoints and intersections)
        self.intersections = self.find_unique_intersections(self.line_geometries)

    def find_unique_intersections(self, line_geometries, eps=1e-5):
        """Find unique intersection points between line geometries."""
        # Build a spatial index on the input geometries
        tree = STRtree(line_geometries)
        unique_points = {}

        # Helper function to round coordinates for uniqueness
        def round_point(pt):
            return (round(pt.x / eps), round(pt.y / eps))
        
        # Check endpoints first
        for geom in line_geometries:
            for coord in (geom.coords[0], geom.coords[-1]):
                pt = Point(coord)
                unique_points[round_point(pt)] = pt

        # For each geometry, query its nearby geometries using the spatial index.
        desc = "Finding intersections" if self.progress_enabled else None
        disable = not self.progress_enabled
        
        for geom in tqdm(line_geometries, desc=desc, unit="geom", disable=disable):
            # Query the tree for nearby geometries (including itself)
            candidates = tree.query(geom)
            for candidate in candidates:
                if not isinstance(candidate, shapely.geometry.base.BaseGeometry):
                    candidate = line_geometries[candidate]
                # Avoid self-checks and duplicate pair evaluations.
                if candidate is geom:
                    continue
                if geom.intersects(candidate):
                    inter = geom.intersection(candidate)
                    if inter.is_empty:
                        continue
                    if inter.geom_type == "Point":
                        unique_points[round_point(inter)] = inter
                    elif hasattr(inter, "geoms"):
                        for pt in inter.geoms:
                            if pt.geom_type == "Point":
                                unique_points[round_point(pt)] = pt

        return list(unique_points.values())

    def compute_uv(self, mask):
        """
        Compute U/V using the Euclidean distance transform.
        Returns an array of shape (H, W, 3) with U, V, and distance (last channel).
        """
        distances, indices = distance_transform_edt(mask == 0, return_indices=True)
        rows, cols = np.indices(mask.shape)
        nearest_y, nearest_x = indices
        dx = nearest_x - cols
        dy = nearest_y - rows
        uv = np.stack([dx, dy, distances], axis=-1)
        uv[distances > self.distance_limit_p] = self.distance_limit_p
        return uv

    def create_jpg_thumbnail(self, tiff_path, target_width=800):
        """
        Create a JPG thumbnail of a TIFF file.
        
        Args:
            tiff_path (str): Path to the input TIFF file
            target_width (int): Target width for the thumbnail in pixels
        """
        if not os.path.exists(tiff_path):
            if self.progress_enabled:
                tqdm_print(f"TIFF file not found: {tiff_path}")
            return
            
        try:
            with rasterio.open(tiff_path) as src:
                # Read the data - handle different band counts appropriately
                if src.count == 1:
                    # Single band - read as grayscale
                    data = src.read(1)
                    # Normalize to 0-255 range
                    data_min, data_max = np.percentile(data[data != src.nodata if src.nodata is not None else data], [2, 98])
                    data_norm = np.clip((data - data_min) / (data_max - data_min) * 255, 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(data_norm, mode='L')
                elif src.count >= 3:
                    # Multi-band - assume RGB for first 3 bands
                    r = src.read(1)
                    g = src.read(2) 
                    b = src.read(3)
                    
                    # Stack and normalize
                    rgb_data = np.stack([r, g, b], axis=-1)
                    # Handle nodata values
                    if src.nodata is not None:
                        mask = (r != src.nodata) & (g != src.nodata) & (b != src.nodata)
                        rgb_data = rgb_data * mask[..., np.newaxis]
                    
                    # Normalize each band to 0-255 range
                    rgb_norm = np.zeros_like(rgb_data, dtype=np.uint8)
                    for i in range(3):
                        band_data = rgb_data[..., i]
                        valid_data = band_data[band_data != (src.nodata if src.nodata is not None else 0)]
                        if len(valid_data) > 0:
                            data_min, data_max = np.percentile(valid_data, [2, 98])
                            if data_max > data_min:
                                rgb_norm[..., i] = np.clip((band_data - data_min) / (data_max - data_min) * 255, 0, 255)
                    
                    pil_image = Image.fromarray(rgb_norm, mode='RGB')
                else:
                    # Two bands - treat as grayscale using first band
                    data = src.read(1)
                    data_min, data_max = np.percentile(data[data != src.nodata if src.nodata is not None else data], [2, 98])
                    data_norm = np.clip((data - data_min) / (data_max - data_min) * 255, 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(data_norm, mode='L')
                
                # Calculate thumbnail size maintaining aspect ratio
                original_width, original_height = pil_image.size
                if original_width > target_width:
                    scale_factor = target_width / original_width
                    thumbnail_height = int(original_height * scale_factor)
                    pil_image = pil_image.resize((target_width, thumbnail_height), Image.Resampling.LANCZOS)
                
                # Generate thumbnail filename
                thumbnail_path = os.path.splitext(tiff_path)[0] + "_thumbnail.jpg"
                
                # Save as JPG with high quality
                pil_image.save(thumbnail_path, "JPEG", quality=85, optimize=True)
                
                if self.progress_enabled:
                    tqdm_print(f"Created thumbnail: {thumbnail_path}")
                    
        except Exception as e:
            if self.progress_enabled:
                tqdm_print(f"Error creating thumbnail for {tiff_path}: {str(e)}")

    def visualize_uv(self, uv):
        """
        Compute an RGB visualization from a UV (and distance) array.
        Returns an 8-bit RGB image (as a numpy array).
        """
        # Compute hue from the angle of the vector
        hue = (np.arctan2(uv[..., 1], uv[..., 0]) + np.pi) / (2 * np.pi)
        # Compute saturation from the magnitude (normalized by distance_limit)
        sat = np.sqrt(uv[..., 0]**2 + uv[..., 1]**2)
        sat = np.clip(sat / self.distance_limit_p, 0, 1)
        # For value, use a binary mask where nonzero saturation becomes 1.0
        val = (sat > 0).astype(np.float32)
        # Invert saturation for visualization purposes
        sat = 1 - sat
        hsv = np.stack([hue, sat, val], axis=-1)
        rgb = (hsv2rgb(hsv) * 255).astype(np.uint8)
        return rgb

    def process(self):
        """Process the rasterization with tiled approach."""
        # Check if we should skip processing due to empty GeoJSON
        if self.skip_processing:
            return
            
        # Open the reference GeoTIFF to get metadata and overall dimensions
        with rasterio.open(self.geotiff_file) as src:
            self.meta = src.meta.copy()
            self.width = src.width
            self.height = src.height
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds

        # Prepare output file names and profiles
        uv_profile = {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": 2,
            "dtype": "int16",
            "crs": self.crs,
            "transform": self.transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        roads_uv_file = self.output_roads_uv
        inters_uv_file = self.output_inters_uv

        mask_profile = {
            "driver": "PNG",
            "height": self.height,
            "width": self.width,
            "count": 1,
            "dtype": "uint8",
            "crs": self.crs,
            "transform": self.transform,
        }
        roads_mask_file = os.path.splitext(roads_uv_file)[0].replace("-roads-uv", "-roads") + ".png"
        inters_mask_file = os.path.splitext(roads_uv_file)[0].replace("-roads-uv", "-intersections") + ".png"

        vis_profile = {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": 3,
            "dtype": "uint8",
            "crs": self.crs,
            "transform": self.transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        
        # NOTE: Was using PNG for visualization, but changed to GTiff because of size limitations. Now trying JPG....
        roads_vis_file = os.path.splitext(roads_uv_file)[0].replace("-roads-uv", "-roads-uv-vis") + ".jpg"
        inters_vis_file = os.path.splitext(inters_uv_file)[0].replace("-intersections-uv", "-intersections-uv-vis") + ".jpg"

        # Track output files for cleanup in case of errors.
        output_files = [roads_uv_file, inters_uv_file, roads_mask_file, inters_mask_file,  roads_vis_file, inters_vis_file]

        # Open all output datasets.
        roads_dst = None
        inters_dst = None
        roads_mask_dst = None
        roads_vis_dst = None
        inters_vis_dst = None

        try:
            roads_dst = rasterio.open(roads_uv_file, "w", **uv_profile)
            inters_dst = rasterio.open(inters_uv_file, "w", **uv_profile)
            roads_mask_dst = rasterio.open(roads_mask_file, "w", **mask_profile)
            inters_mask_dst = rasterio.open(inters_mask_file, "w", **mask_profile)
            roads_vis_dst = rasterio.open(roads_vis_file, "w", **vis_profile)
            inters_vis_dst = rasterio.open(inters_vis_file, "w", **vis_profile)

            # Compute tile grid over the full raster (non-overlapping interior tiles)
            tile_indices = []
            base_tile_size = self.config['base_tile_size']
            for top in range(0, self.height, base_tile_size):
                for left in range(0, self.width, base_tile_size):
                    bottom = min(top + base_tile_size, self.height)
                    right = min(left + base_tile_size, self.width)
                    tile_indices.append((top, left, bottom, right))

            # Reproject intersection points to the same CRS as the GeoTIFF
            intersections_reproj = [
                shape(transform_geom("EPSG:4326", self.crs.to_string(), mapping(pt)))
                for pt in self.intersections
            ]
            
            road_geom_reproj = [
                shape(transform_geom("EPSG:4326", self.crs.to_string(), mapping(geom)))
                for geom in self.line_geometries
            ]

            desc = "Processing tiles" if self.progress_enabled else None
            disable = not self.progress_enabled
            pbar = tqdm(total=len(tile_indices), desc=desc, disable=disable)
            
            # Process each tile
            for (top, left, bottom, right) in tile_indices:
                # Determine extended window coordinates (with margin)
                ext_top = max(0, top - self.distance_limit_p)
                ext_left = max(0, left - self.distance_limit_p)
                ext_bottom = min(self.height, bottom + self.distance_limit_p)
                ext_right = min(self.width, right + self.distance_limit_p)

                ext_window = Window(ext_left, ext_top, ext_right - ext_left, ext_bottom - ext_top)
                ext_transform = rasterio.windows.transform(ext_window, self.transform)
                ext_shape = (int(ext_window.height), int(ext_window.width))

                # Geographic bounds of the extended window for filtering vector features
                minx, miny, maxx, maxy = rasterio.windows.bounds(ext_window, self.transform)
                window_bbox = box(minx, miny, maxx, maxy)

                # Filter and reproject road features for this tile
                roads_features_tile = []
                for i, geom_reproj in enumerate(road_geom_reproj):
                     if geom_reproj.intersects(window_bbox):  
                        if self.multiclass:
                            # Get the road type from the original geometry
                            label = self.line_labels[i]
                        else:
                            label = 255
                        roads_features_tile.append((geom_reproj, label))
                        
                # Rasterize roads mask for the extended window
                roads_mask_full = rasterize(
                    roads_features_tile,
                    out_shape=ext_shape,
                    transform=ext_transform,
                    fill=0,
                    dtype="uint8"
                )
                
                # For intersections, filter points in the extended window
                inter_points = [pt for pt in intersections_reproj if pt.intersects(window_bbox)]
                inter_shapes = [(mapping(pt), 255) for pt in inter_points]
                intersections_mask_full = rasterize(
                    inter_shapes,
                    out_shape=ext_shape,
                    transform=ext_transform,
                    fill=0,
                    dtype="uint8"
                )

                # Compute UV arrays for roads and intersections on the extended window
                roads_uv_full = self.compute_uv(roads_mask_full)
                inters_uv_full = self.compute_uv(intersections_mask_full)

                # Determine indices for the interior (non-overlapped) region within the extended window
                y_off = top - ext_top
                x_off = left - ext_left
                y_end = y_off + (bottom - top)
                x_end = x_off + (right - left)

                roads_uv_crop = roads_uv_full[y_off:y_end, x_off:x_end, :]
                inters_uv_crop = inters_uv_full[y_off:y_end, x_off:x_end, :]

                # Write UV geotiff data (only U and V channels)
                out_window = Window(left, top, right - left, bottom - top)
                roads_dst.write(roads_uv_crop[:, :, 0].astype(np.int16), 1, window=out_window)
                roads_dst.write(roads_uv_crop[:, :, 1].astype(np.int16), 2, window=out_window)
                inters_dst.write(inters_uv_crop[:, :, 0].astype(np.int16), 1, window=out_window)
                inters_dst.write(inters_uv_crop[:, :, 1].astype(np.int16), 2, window=out_window)

                # Write roads mask (crop the interior from the extended window)
                roads_mask_tile = roads_mask_full[y_off:y_end, x_off:x_end]
                roads_mask_dst.write(roads_mask_tile.astype(np.uint8), 1, window=out_window)
                
                # Write intersections mask (crop the interior from the extended window)
                inters_mask_tile = intersections_mask_full[y_off:y_end, x_off:x_end]
                inters_mask_dst.write(inters_mask_tile.astype(np.uint8), 1, window=out_window)

                # Compute visualization for roads and intersections from the UV arrays
                roads_vis_tile = self.visualize_uv(roads_uv_full)[y_off:y_end, x_off:x_end, :]
                inters_vis_tile = self.visualize_uv(inters_uv_full)[y_off:y_end, x_off:x_end, :]

                # Write visualization tiles (3 bands) for roads
                roads_vis_dst.write(roads_vis_tile[:, :, 0], 1, window=out_window)
                roads_vis_dst.write(roads_vis_tile[:, :, 1], 2, window=out_window)
                roads_vis_dst.write(roads_vis_tile[:, :, 2], 3, window=out_window)

                # Write visualization tiles for intersections
                inters_vis_dst.write(inters_vis_tile[:, :, 0], 1, window=out_window)
                inters_vis_dst.write(inters_vis_tile[:, :, 1], 2, window=out_window)
                inters_vis_dst.write(inters_vis_tile[:, :, 2], 3, window=out_window)

                pbar.update(1)
            pbar.close()

            # Close all output datasets
            roads_dst.close()
            inters_dst.close()
            roads_mask_dst.close()
            inters_mask_dst.close()
            roads_vis_dst.close()
            inters_vis_dst.close()

            if self.progress_enabled:
                print("Finished processing all tiles and saved output GeoTIFFs and images.")
                print(f" Roads : {roads_vis_file}")
                print(f" Xings : {inters_vis_file}")

            # Create JPG thumbnails of the output TIFF files
            self.create_jpg_thumbnail(roads_vis_file, target_width=800)
            self.create_jpg_thumbnail(inters_vis_file, target_width=800)
            self.create_jpg_thumbnail(self.geotiff_file, target_width=800)

            print("Created thumbnails for output files.")

        except Exception as err:
            # Attempt to close any open datasets
            for ds in [roads_dst, inters_dst, roads_mask_dst, roads_vis_dst, inters_vis_dst]:
                if ds is not None:
                    try:
                        ds.close()
                    except Exception:
                        pass
            # Remove any partially written output files
            for f in output_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception as cleanup_err:
                    print(f"Error cleaning up file {f}: {cleanup_err}")
            raise err


STAGE = 'fetch' # choices=["all", "extract", "fetch", "rasterize"]

PYTHON_EXE = sys.executable  # Get the current Python executable path

def run_usgs_downloader_check():
    print("[INFO] Please run the 'usgs_geotiff_downloader.py' script manually to download zip files.")


def run_extract_tifs():
    print("[INFO] Running 'extract_tifs.py'...")
    subprocess.run([
        PYTHON_EXE, "scripts/extract_tifs.py",
        "--zip_folder", RAW_ZIPS_FOLDER,
        "--tif_folder", PROCESSED_IMAGES_FOLDER
    ], check=True)


def run_fetch_osm_roads(images_folder: str, geojson_folder: str) -> None:
    """
    For each .tif in images_folder (recursively), fetch OSM sidewalks/footways
    using GeotiffMapper.get_osm_roads() and save a GeoJSON next to it in
    geojson_folder with the same base name.

    Args:
        images_folder: Root directory containing GeoTIFFs.
        geojson_folder: Output directory for GeoJSON files.
    """
    os.makedirs(geojson_folder, exist_ok=True)

    tif_paths = sorted(glob.glob(os.path.join(images_folder, "**", "*.tif"), recursive=True))
    if not tif_paths:
        print(f"[WARN] No .tif files found under {images_folder}")
        return

    for tif_path in tif_paths:
        base = os.path.splitext(os.path.basename(tif_path))[0]
        geojson_path = os.path.join(geojson_folder, f"{base}.geojson")

        if not os.path.exists(geojson_path):
            print(f"[INFO] Fetching roads for {tif_path}...")
            try:
                mapper = GeotiffMapper(tif_path)
                ways = mapper.get_osm_roads()  # expected to return list of overpy.Way

                features = []
                seen_way_ids = set()

                for w in ways:
                    # Dedupe by way id just in case
                    if getattr(w, "id", None) in seen_way_ids:
                        continue
                    seen_way_ids.add(getattr(w, "id", None))

                    # Build LineString coords in GeoJSON order: [lon, lat]
                    coords = []
                    for n in getattr(w, "nodes", []):
                        # n.lon / n.lat are strings in some overpy versions -> cast to float
                        if n.lon is not None and n.lat is not None:
                            coords.append([float(n.lon), float(n.lat)])

                    # Skip degenerate geometries
                    if len(coords) < 2:
                        continue

                    # Tags -> properties (ensure JSON-serializable)
                    props = {"osm_id": getattr(w, "id", None)}
                    tags = getattr(w, "tags", {}) or {}
                    props.update({k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
                                  for k, v in tags.items()})

                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": props
                    })

                fc = {
                    "type": "FeatureCollection",
                    "name": base,
                    "features": features
                }

                with open(geojson_path, "w", encoding="utf-8") as f:
                    json.dump(fc, f, ensure_ascii=False)

                print(f"[INFO] Saved GeoJSON to {geojson_path} ({len(features)} features).")

            except Exception as e:
                print(f"[ERROR] Failed to fetch/save for {tif_path}: {e}")

        else:
            print(f"[INFO] GeoJSON already exists for {tif_path}, skipping...")

from rasterio.warp import transform, calculate_default_transform, reproject, Resampling

def get_utm_crs(lat, lon):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return CRS.from_epsg(32600 + zone)
    else:
        return CRS.from_epsg(32700 + zone)
    
def warp_to_pixel_size(input_filename, output_filename, meters_per_pixel):
    """
    Reprojects the input GeoTIFF into a UTM coordinate system determined by the
    geographic center (centroid) of the source image. The output image will have
    a pixel size of exactly meters_per_pixel.
    
    For debugging, this version prints the source bounds and the new bounds
    (both in the source CRS and transformed to geographic coordinates).
    """
    with rasterio.open(input_filename) as src:
        # Debug: check source data values
        src_band1 = src.read(1)
        
        # Print original bounds in source CRS
        
        # Transform source bounds to geographic (EPSG:4326)
        src_bounds_latlon = transform_bounds(src.crs, CRS.from_epsg(4326),
                                               src.bounds.left, src.bounds.bottom,
                                               src.bounds.right, src.bounds.top)
        
        # Compute the center of the source image in source CRS coordinates.
        bounds = src.bounds
        center_x = (bounds.left + bounds.right) / 2.0
        center_y = (bounds.top + bounds.bottom) / 2.0

        # Transform the center to geographic coordinates (EPSG:4326).
        lon_arr, lat_arr = transform(src.crs, CRS.from_epsg(4326), [center_x], [center_y])
        lon = lon_arr[0]
        lat = lat_arr[0]
        
        # Determine the target UTM CRS.
        target_crs = get_utm_crs(lat, lon)

        # Use calculate_default_transform to compute the new transform, width, and height.
        dst_transform, new_width, new_height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=meters_per_pixel)

        # Print new bounds in target CRS (in meters)
        new_bounds = (
            dst_transform.c,
            dst_transform.f + new_height * dst_transform.e,  # bottom
            dst_transform.c + new_width * dst_transform.a,     # right
            dst_transform.f                              # top
        )
        
        # Also, transform the new bounds to geographic (lat/lon) for debugging.
        new_bounds_latlon = transform_bounds(target_crs, CRS.from_epsg(4326),
                                               new_bounds[0], new_bounds[1],
                                               new_bounds[2], new_bounds[3])

        # Prepare new metadata.
        new_meta = src.meta.copy()
        new_meta.update({
            "crs": target_crs,
            "transform": dst_transform,
            "width": new_width,
            "height": new_height,
            "driver": "GTiff"
        })

        # Use src.nodata if available.
        src_nodata = src.nodata
        if src_nodata is not None:
            print("Source nodata:", src_nodata)
        
        # Reproject each band.
        with rasterio.open(output_filename, "w", **new_meta) as dst:
            for i in range(1, src.count + 1):
                dest = np.empty((new_height, new_width), dtype=src.dtypes[0])
                reproject(
                    source=rasterio.band(src, i),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src_nodata,
                    dst_nodata=src_nodata
                )
                dst.write(dest, i)

def rasterize_geojson_to_geotiff(geojson_file, geotiff_file, output_folder=None, config=None, disable_progress=False):
    """
    High-level function to rasterize a GeoJSON file to align with a GeoTIFF.
    
    Args:
        geojson_file (str): Path to the input GeoJSON file
        geotiff_file (str): Path to the reference GeoTIFF file
        output_folder (str, optional): Path to the output folder. If None, uses config's output_folder.
        config (dict, optional): Configuration dictionary. If None, uses default config.
        disable_progress (bool): If True, disable progress output
        
    Returns:
        dict: Dictionary with paths to generated output files
    """
    # Use provided config or get default
    if config is None:
        print("[ERROR]: No config provided.")
    
    # Compute output file paths using the geojson filename as stem
    stem = os.path.join(output_folder, os.path.splitext(os.path.basename(geojson_file))[0])
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Define output file paths using simple scale tag
    pixel_size = config['pixel_size_meters']
    scale_tag = config['scale_tag']
    
    geotiff_scaled_path = f"{stem}-{scale_tag}.tif"
    output_roads_uv = f"{stem}-roads-uv.tif"
    output_inters_uv = f"{stem}-intersections-uv.tif"
    
    # Warp GeoTIFF to specified pixel size if not already done
    if not os.path.isfile(geotiff_scaled_path):
        if not disable_progress and config['progress_enabled']:
            tqdm_print(f"Warping GeoTIFF to {scale_tag} pixel size...")
        warp_to_pixel_size(geotiff_file, geotiff_scaled_path, pixel_size)
    
    # Initialize and run the tiled rasterizer
    rasterizer = TiledRasterizer(
        geojson_file, 
        geotiff_scaled_path,
        output_roads_uv, 
        output_inters_uv,
        config=config,
        disable_progress=disable_progress
    )
    rasterizer.process()
    
    # Return paths to generated files
    return {
        'scaled_geotiff': geotiff_scaled_path,
        'roads_uv': output_roads_uv,
        'intersections_uv': output_inters_uv,
        'roads_mask': os.path.splitext(output_roads_uv)[0].replace("-roads-uv", "-roads") + ".png",
        'intersections_mask': os.path.splitext(output_roads_uv)[0].replace("-roads-uv", "-intersections") + ".png",
        'roads_vis': os.path.splitext(output_roads_uv)[0].replace("-roads-uv", "-roads-uv-vis") + ".jpg",
        'intersections_vis': os.path.splitext(output_inters_uv)[0].replace("-intersections-uv", "-intersections-uv-vis") + ".jpg"
    }

def run_rasterize(images_folder, geojson_folder, masks_folder, config_name=None):
    """
    Run rasterization using the configuration-based approach.
    
    Args:
        images_folder (str): Path to folder containing TIFF files
        geojson_folder (str): Path to folder containing GeoJSON files  
        masks_folder (str): Path to output folder for rasterized masks
        config_name (str, optional): Name of configuration to use. If None, uses 'default'.
    """

    # print("[INFO] Running rasterization for each TIFF and corresponding GeoJSON file...")

    # Get the configuration by name
    config = DEFAULT_RASTERIZE_CONFIG
    
    pixel_size = config['pixel_size_meters']
    scale_tag = config['scale_tag']
    print(f"[INFO] Using pixel size: {pixel_size}m (scale tag: {scale_tag})")
    
    tiff_files = glob.glob(os.path.join(images_folder, "*.tif"))
    total_files = len(tiff_files)
    
    for index, tif_path in enumerate(tqdm(tiff_files, desc='Rasterizing TIFFs'), start=1):
        print(f"[INFO] Processing file {index}/{total_files}: {tif_path}")
        tif_filename = os.path.basename(tif_path)
        geojson_path = os.path.join(geojson_folder, tif_filename.replace(".tif", ".geojson"))
        
        if not os.path.exists(geojson_path):
            print(f"[WARNING] GeoJSON not found for {tif_path}, skipping rasterization.")
            continue
            
        # Check if key output files already exist to skip processing
        stem = os.path.join(masks_folder, os.path.splitext(tif_filename)[0])
        expected_outputs = [
            f"{stem}-{scale_tag}.tif",  # Scaled GeoTIFF
            f"{stem}-roads-uv.tif",     # Roads UV file
            f"{stem}-intersections-uv.tif",  # Intersections UV file
            f"{stem}-roads.png",        # Roads mask
        ]
        
        # Check if all key output files exist
        all_outputs_exist = all(os.path.exists(output_path) for output_path in expected_outputs)
        
        if all_outputs_exist:
            print(f"[INFO] All output files already exist for {tif_filename}, skipping...")
            continue
            
        print(f"[INFO] Rasterizing {geojson_path} with {tif_path}...")
        
        try:
            # Use the high-level configuration-based function
            output_files = rasterize_geojson_to_geotiff(
                geojson_path,
                tif_path,
                masks_folder,
                config=config,
                disable_progress=False
            )
            
            if config.get('progress_enabled', True):
                print(f"[INFO] Successfully rasterized {tif_filename}")
                print(f"       Generated {len([f for f in output_files.values() if os.path.exists(f)])} output files")
                
        except Exception as e:
            print(f"[ERROR] Failed to rasterize {tif_filename}: {str(e)}")
            continue
                

def main():

    if not os.path.exists(PROCESSED_IMAGES_FOLDER):
        print(f"[ERROR] Images folder '{PROCESSED_IMAGES_FOLDER}' does not exist. Ensure the correct path is set.")
        return

    if STAGE in ["all", "fetch"]:
        # Step 3: Fetch OSM roads
        os.makedirs(PROCESSED_GEOJSON_FOLDER, exist_ok=True)
        run_fetch_osm_roads(PROCESSED_IMAGES_FOLDER, PROCESSED_GEOJSON_FOLDER)

    
    if STAGE in ["all", "rasterize"]:
        # Step 4: Rasterize GeoJSON files using configuration-based approach
        os.makedirs(ROAD_MASKS_FOLDER, exist_ok=True)
        run_rasterize(PROCESSED_IMAGES_FOLDER, PROCESSED_GEOJSON_FOLDER, ROAD_MASKS_FOLDER, config_name="default")
    

if __name__ == "__main__":
    main()
