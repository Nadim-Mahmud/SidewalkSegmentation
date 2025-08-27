""" 
Copyright @ 2025 Siddhant S. Karki
Computer Science & Software Engineering Dept.
Miami University OH
"""
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image

class TIFProcessor:
    """
        A utility class for processing GeoTIFF files and extracting image patches 
    based on geographic coordinates. This class provides functionality to 
    convert latitude and longitude to pixel coordinates, calculate pixel 
    bounds for a given window size, and extract image patches as PIL images.

    Methods:
        latlon_to_pixel(lat, lon):
            Converts geographic coordinates (latitude, longitude) to pixel 
            coordinates in the GeoTIFF.
        calc_pixel_bounds(center_row, center_col):
            Calculates the pixel bounds for a patch centered at the given 
            pixel coordinates.
        get_patch(lat, lon):
            Extracts an image patch centered at the given geographic 
            coordinates (latitude, longitude) and returns it as a PIL image.
    """

    def __init__(self, tif_path, window_size=(400, 400)):
        self.window_size = window_size
        self._init_tif(tif_path)

    def _init_tif(self, tif_path):
        self.src = rasterio.open(tif_path)
        self.crs = self.src.crs
        self.transform = self.src.transform
        self.transformer = Transformer.from_crs(
            "EPSG:4326", self.crs, always_xy=True
        )

        # Auto-detect number of bands
        bands = self.src.count
        if bands == 1:
            # Single-channel (grayscale)
            self.tif_data = self.src.read(1)
            self.pil_mode = "L"
        else:
            # Multi-channel (use up to first 3 bands as RGB)
            channels = min(3, bands)
            self.tif_data = np.dstack([
                self.src.read(i)
                for i in range(1, channels + 1)
            ])
            self.pil_mode = "RGB"

    def latlon_to_pixel(self, lat, lon):
        # Convert lat/lon to projection coordinates, then to pixel indices
        x_proj, y_proj = self.transformer.transform(lon, lat)
        inv = ~self.transform
        col, row = inv * (x_proj, y_proj)
        return int(col), int(row)

    def calc_pixel_bounds(self, center_col, center_row):
        half_w = self.window_size[1] // 2
        half_h = self.window_size[0] // 2
        return (
            (center_row - half_h, center_row + half_h),
            (center_col - half_w, center_col + half_w),
        )

    def get_patch(self, lat, lon, rgb_indices=(1, 2, 3)):
        # Compute pixel window
        col, row = self.latlon_to_pixel(lat, lon)
        half_h = self.window_size[0] // 2
        half_w = self.window_size[1] // 2
        r0, r1 = row - half_h, row + half_h
        c0, c1 = col - half_w, col + half_w

        # Clamp to image bounds
        r0 = max(0, r0); c0 = max(0, c0)
        r1 = min(self.src.height, r1); c1 = min(self.src.width, c1)
        if (r1 - r0) != self.window_size[0] or (c1 - c0) != self.window_size[1]:
            # You'll skip these upstream anyway
            pass

        # Read just the window (faster, and keeps dtype)
        if self.src.count == 1:
            arr = self.src.read(1, window=((r0, r1), (c0, c1)))
            bands = [arr]
        else:
            # Choose bands explicitly; rasterio is 1-based indexing
            bands = [self.src.read(b, window=((r0, r1), (c0, c1))) for b in rgb_indices]
            # If a band index is out of range, fall back to first N bands
            if any(b is None for b in bands):
                bands = [self.src.read(i, window=((r0, r1), (c0, c1))) for i in range(1, min(3, self.src.count)+1)]

        patch = np.dstack(bands) if len(bands) > 1 else bands[0]

        # Build mask: prefer read_masks; else nodata
        try:
            masks = [self.src.read_masks(b if self.src.count > 1 else 1,
                                        window=((r0, r1), (c0, c1))) for _ in bands]
            mask = np.minimum.reduce(masks)  # 0 = nodata, 255 = valid
            valid = mask > 0
        except Exception:
            nd = self.src.nodata
            if nd is not None:
                valid = ~np.any(patch == nd, axis=2) if patch.ndim == 3 else (patch != nd)
            else:
                valid = np.ones(patch.shape[:2], dtype=bool)

        # Percentile stretch to 0–255 per band over valid pixels
        def stretch_band(band, valid_mask):
            vals = band[valid_mask]
            if vals.size == 0:
                return np.zeros_like(band, dtype=np.uint8)
            lo, hi = np.percentile(vals, [2, 98])  # tweak if needed
            if hi <= lo:
                # fallback to simple scaling
                lo, hi = vals.min(), vals.max() if vals.size else (0, 1)
                if hi == lo:
                    return np.zeros_like(band, dtype=np.uint8)
            out = (band.astype(np.float32) - lo) * (255.0 / (hi - lo))
            out = np.clip(out, 0, 255).astype(np.uint8)
            out[~valid_mask] = 0
            return out

        if patch.ndim == 2:
            out = stretch_band(patch, valid)
            return Image.fromarray(out, mode="L")

        stretched = np.dstack([stretch_band(patch[..., i], valid) for i in range(patch.shape[2])])

        # Optional: add alpha so NoData is transparent
        alpha = (valid * 255).astype(np.uint8)
        rgba = np.dstack([stretched, alpha])
        return Image.fromarray(rgba, mode="RGBA")






