""" 
Copyright @ 2025 Siddhant S. Karki
Computer Science & Software Engineering Dept.
Miami University OH
"""

import math
import requests
from PIL import Image
from io import BytesIO


class TileExtractor:
    """
    A utility class for extracting map tiles from OpenStreetMap and combining them into a single image.
    This class provides functionality to fetch map tiles based on geographical coordinates (latitude and longitude)
    and zoom level, and to generate a cropped image centered around the specified location. It supports creating
    a 3x3 grid of tiles to ensure seamless cropping and provides a fallback mechanism for missing tiles.

    Attributes:
        zoom (int): The zoom level for the map tiles, ranging from 0 (world view) to 19 (maximum detail).

    Methods:
        get_tile_image(lat, lon, size=256):
            Fetches and combines map tiles to create a cropped image centered at the specified latitude and longitude.
    """
    def __init__(self, zoom: int = 16):
        if not (0 <= zoom <= 19):
            raise ValueError("Zoom must be between 0 and 19")
        self.zoom = zoom

    def _deg2num(self, lat, lon):
        lat_rad = math.radians(lat)
        n = 2.0 ** self.zoom
        xtile = (lon + 180.0) / 360.0 * n
        ytile = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
        return xtile, ytile

    def _fetch_tile(self, x, y):
        url = f"https://tile.openstreetmap.org/{self.zoom}/{x}/{y}.png"
        headers = {
            "User-Agent": "TileExtractor/1.0 (https://example.com)",
            "Accept": "image/png"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    def get_tile_image(self, lat, lon, size=256):
        x_float, y_float = self._deg2num(lat, lon)
        x_tile, y_tile = int(x_float), int(y_float)

        # Create 3x3 tile grid (768x768)
        combined = Image.new('RGB', (256 * 3, 256 * 3))

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                try:
                    tile = self._fetch_tile(x_tile + dx, y_tile + dy)
                except Exception:
                    tile = Image.new("RGB", (256, 256), (255, 255, 255))
                combined.paste(tile, ((dx + 1) * 256, (dy + 1) * 256))

        # Pixel offset of exact lat/lon
        x_offset = int((x_float - x_tile) * 256)
        y_offset = int((y_float - y_tile) * 256)

        # Crop center
        left = 256 + x_offset - size // 2
        upper = 256 + y_offset - size // 2
        return combined.crop((left, upper, left + size, upper + size))

