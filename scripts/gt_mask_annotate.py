"""
Batch annotate (roads, sidewalks, crosswalks) from ONE GeoJSON per TIFF.

- For each .tif in INPUT_TIFS_DIR, loads a matching GeoJSON named <basename>.geojson
  from INPUT_GEOJSONS_DIR.
- Classifies features by OSM-like tags.
- Buffers lines/points in meters via EPSG:3857 (per class).
- Clips to raster extent, rasterizes by class with first-hit-by-priority.
- Saves a colorized 3-band GeoTIFF mask in OUTPUT_DIR using the original tif filename.

Dependencies: rasterio, geopandas, shapely, numpy, Pillow (Pillow not used here now)
"""

import os
import glob
import rasterio
from rasterio import features
import numpy as np
import geopandas as gpd
from shapely.geometry import box

# -----------------------------------------------------------------------------
# I/O Directories (edit these)
# -----------------------------------------------------------------------------
INPUT_TIFS_DIR      = "/home/hanew/your_project_folder/omniacc/data/tifs/manhattan/images"
INPUT_GEOJSONS_DIR  = "/home/hanew/your_project_folder/omniacc/data/tifs/manhattan/geojsons"
OUTPUT_DIR          = "/home/hanew/your_project_folder/omniacc/data/tifs/manhattan/masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Class IDs and rendering
# -----------------------------------------------------------------------------
CLASS_BG         = 0
CLASS_SIDEWALK   = 1
CLASS_CROSSWALK  = 2

# Draw order: earlier wins when compositing
CLASS_PRIORITY = [CLASS_CROSSWALK, CLASS_SIDEWALK]

# Per-class buffer (meters) for lines/points
CLASS_BUFFER_M = {
    CLASS_SIDEWALK: 1.0,
    CLASS_CROSSWALK: 1.0,
}

# Output color palette (RGB)
PALETTE_RGB = {
    CLASS_BG:        (0,   0,   0),   # Background - black
    CLASS_SIDEWALK:  (0,   0, 255),   # Sidewalk   - blue
    #CLASS_ROAD:      (0, 255,   0),   # Road       - green
    CLASS_CROSSWALK: (255, 0,   0),   # Crosswalk  - red
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_crs(gdf, default_epsg=4326):
    """Ensure the GeoDataFrame has a CRS; assume WGS84 if missing."""
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=default_epsg)
    return gdf

def buffer_in_meters(gdf, meters):
    """Buffer geometries by meters using EPSG:3857, then back to original CRS."""
    if meters <= 0:
        return gdf
    src_crs = gdf.crs
    gdf_3857 = gdf.to_crs(epsg=3857)
    gdf_3857["geometry"] = gdf_3857.geometry.buffer(meters)
    gdf = gdf_3857.to_crs(src_crs)
    return gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

def classify_feature(props):
    """
    Return class ID from a feature's properties dict using OSM-like tags.
    """
    p = {str(k).lower(): (str(v).lower() if v is not None else None) for k, v in props.items()}

    highway   = p.get("highway")
    footway   = p.get("footway")
    sidewalk  = p.get("sidewalk")
    crossing  = p.get("crossing")
    fclass    = p.get("fclass") or p.get("class")
    feature   = p.get("feature") or p.get("type")
    service   = p.get("service")
    foot      = p.get("foot")
    bicycle   = p.get("bicycle")

    # ---------- CROSSWALKS ----------
    if (
        footway == "crossing" or
        (crossing is not None and crossing not in ("no", "none")) or
        feature in ("crosswalk", "zebra")
    ):
        return CLASS_CROSSWALK

    # ---------- SIDEWALKS ----------
    if (
        footway == "sidewalk" or
        sidewalk in ("yes", "left", "right", "both") or
        (highway == "footway" and footway != "crossing") or
        (highway == "path" and (foot in ("designated", "yes")) and bicycle not in ("designated","yes"))
    ):
        return CLASS_SIDEWALK

    # ---------- ROADS ----------
    road_like = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "service", "unclassified", "living_street",
        "road", "motorway_link", "trunk_link", "primary_link",
        "secondary_link", "tertiary_link"
    }
    if (highway in road_like) or (fclass in road_like) or (feature in road_like):
        # Exclude parking lots/aisles explicitly
        if service in ("parking", "parking_aisle"):
            return CLASS_BG
        # If explicit sidewalk tag present, prefer sidewalk
        if sidewalk in ("yes", "left", "right", "both"):
            return CLASS_SIDEWALK
        return CLASS_BG

    return CLASS_BG

def classify_row(row):
    props = {} if row is None else (row.get("properties", None) or {})
    if not props:
        props = {k: row[k] for k in row.index if k != "geometry"}
    return classify_feature(props)

def rasterize_pair(tif_path, geojson_path, output_path):
    # Open raster
    with rasterio.open(tif_path) as src:
        transform = src.transform
        raster_crs = src.crs
        height, width = src.height, src.width
        raster_bounds_poly = box(*src.bounds)
        raster_profile = src.profile

    # Read and align GeoJSON
    gdf = gpd.read_file(geojson_path)
    gdf = ensure_crs(gdf, default_epsg=4326)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    # Clip & clean
    gdf = gdf[gdf.geometry.intersects(raster_bounds_poly)]
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    if gdf.empty:
        print(f"[WARN] No overlapping features: {os.path.basename(geojson_path)}")
        # still write an all-background mask to keep outputs consistent
        profile_rgb = raster_profile.copy()
        profile_rgb.update(driver="GTiff", dtype=rasterio.uint8, count=3, compress="lzw")
        H, W = height, width
        rgb = np.zeros((3, H, W), dtype=np.uint8)
        with rasterio.open(output_path, "w", **profile_rgb) as dst:
            dst.write(rgb[0], 1); dst.write(rgb[1], 2); dst.write(rgb[2], 3)
        return

    # Classify
    gdf["__cid__"] = gdf.apply(classify_row, axis=1)
    gdf = gdf[gdf["__cid__"].isin(set(CLASS_PRIORITY))]
    if gdf.empty:
        print(f"[WARN] No features matched class rules: {os.path.basename(geojson_path)}")
        # write background-only output
        profile_rgb = raster_profile.copy()
        profile_rgb.update(driver="GTiff", dtype=rasterio.uint8, count=3, compress="lzw")
        H, W = height, width
        rgb = np.zeros((3, H, W), dtype=np.uint8)
        with rasterio.open(output_path, "w", **profile_rgb) as dst:
            dst.write(rgb[0], 1); dst.write(rgb[1], 2); dst.write(rgb[2], 3)
        return

    # Work copy + geometry-type masks
    gdf_work = gdf.copy()
    is_line  = gdf_work.geometry.geom_type.isin(["LineString","MultiLineString"])
    is_point = gdf_work.geometry.geom_type.isin(["Point","MultiPoint"])

    # Buffer lines & points per class
    for cid in set(gdf_work["__cid__"]):
        sel = gdf_work["__cid__"] == cid
        buf_m = CLASS_BUFFER_M.get(cid, 0.0)

        sel_lines = sel & is_line
        if sel_lines.any() and buf_m > 0:
            gdf_work.loc[sel_lines, "geometry"] = buffer_in_meters(gdf_work.loc[sel_lines], buf_m).geometry

        sel_pts = sel & is_point
        if sel_pts.any():
            gdf_work.loc[sel_pts, "geometry"] = buffer_in_meters(gdf_work.loc[sel_pts], max(buf_m, 1.0)).geometry

    gdf_work = gdf_work[~gdf_work.geometry.is_empty & gdf_work.geometry.notna()]

    # Build multiclass mask with priority
    multiclass_mask = np.zeros((height, width), dtype=np.uint8)
    for cid in CLASS_PRIORITY:
        layer = gdf_work[gdf_work["__cid__"] == cid]
        if layer.empty:
            continue

        layer_mask = features.rasterize(
            ((geom, 1) for geom in layer.geometry),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        multiclass_mask = np.where((multiclass_mask == 0) & (layer_mask > 0), cid, multiclass_mask)

    # Colorize and write GeoTIFF
    H, W = multiclass_mask.shape
    rgb = np.zeros((3, H, W), dtype=np.uint8)
    for cid, (r, g, b) in PALETTE_RGB.items():
        m = (multiclass_mask == cid)
        if m.any():
            rgb[0][m] = r
            rgb[1][m] = g
            rgb[2][m] = b

    profile_rgb = raster_profile.copy()
    profile_rgb.update(driver="GTiff", dtype=rasterio.uint8, count=3, compress="lzw")

    with rasterio.open(output_path, "w", **profile_rgb) as dst:
        dst.write(rgb[0], 1)
        dst.write(rgb[1], 2)
        dst.write(rgb[2], 3)

# -----------------------------------------------------------------------------
# Batch runner
# -----------------------------------------------------------------------------
def main():
    tif_paths = sorted(glob.glob(os.path.join(INPUT_TIFS_DIR, "*.tif")))
    if not tif_paths:
        raise RuntimeError(f"No .tif files found in {INPUT_TIFS_DIR}")

    for tif_path in tif_paths:
        base = os.path.splitext(os.path.basename(tif_path))[0]
        geojson_path = os.path.join(INPUT_GEOJSONS_DIR, f"{base}.geojson")
        output_path  = os.path.join(OUTPUT_DIR, f"{base}.tif")  # save using original tif name

        if not os.path.exists(geojson_path):
            print(f"[SKIP] Missing GeoJSON for {base}: {geojson_path}")
            continue

        try:
            print(f"[PROCESS] {base}")
            rasterize_pair(tif_path, geojson_path, output_path)
            print(f"[OK] Wrote {output_path}")
        except Exception as e:
            print(f"[ERROR] {base}: {e}")

if __name__ == "__main__":
    main()
