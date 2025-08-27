""" 
Copyright @ 2025 Siddhant S. Karki
Computer Science & Software Engineering Dept.
Miami University OH
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io
import math
import requests
from typing import Tuple, List, Dict



class OSMExtractor:
    """
        A class to extract OpenStreetMap (OSM) data for a given geographical location.
    This class provides methods to convert latitude and longitude to tile coordinates,
    retrieve OSM XML data for a specified location, and handle zoom levels for map tiles.
    Attributes:
        zoom (int): The zoom level for the map tiles, ranging from 0 (world view) to 19 (street view).
    Methods:
        _deg2num_f(lat_deg, lon_deg):
            Converts latitude and longitude to fractional tile coordinates (x, y) based on the zoom level.
        _num2deg(x, y):
            Converts tile coordinates (x, y) back to latitude and longitude.
        get_osmxml(lat, lon):
            Retrieves OSM XML data for the specified latitude and longitude using the Overpass API.
    """
    def __init__(self, zoom: int = 16):
        if not (0 <= zoom <= 19):
            raise ValueError("Zoom must be between 0 and 19")
        self.zoom = zoom

    def _deg2num_f(self, lat_deg, lon_deg):
        lat_rad = math.radians(lat_deg)
        n = 2 ** self.zoom
        x = (lon_deg + 180.0) / 360.0 * n
        y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        return x, y

    def _num2deg(self, x, y):
        n = 2 ** self.zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    def get_osmxml(self, lat, lon):
        x_f, y_f = self._deg2num_f(lat, lon)

        north, west = self._num2deg(x_f - 0.5, y_f - 0.5)
        south, east = self._num2deg(x_f + 0.5, y_f + 0.5)

        bbox = f"{south},{west},{north},{east}"

        query = f"""
        [out:xml][timeout:25];
          (
            node({bbox});
            way({bbox});
            relation({bbox});
          );
          out geom;
        """

        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=30)
        r.raise_for_status()
        return r.text
    

class OSMParser:
    """
    A parser for extracting nodes and ways from OpenStreetMap (OSM) XML data.
    This class processes OSM XML data to extract geographic nodes and ways. 
    Nodes represent specific points with latitude and longitude, while ways 
    represent paths or areas defined by a sequence of nodes.
    Attributes:
        root (Element): The root element of the parsed OSM XML data.
        nodes (Dict[str, Tuple[float, float]]): A dictionary mapping node IDs 
            to their geographic coordinates (longitude, latitude).
        ways (List[List[Tuple[float, float]]]): A list of ways, where each way 
            is represented as a list of geographic coordinates (longitude, latitude).
    """

    def __init__(self, osm_xml: str):
        self.root = ET.fromstring(osm_xml)
        self.nodes = self._extract_nodes()
        self.ways = self._extract_ways()

    def _extract_nodes(self) -> Dict[str, Tuple[float, float]]:
        nodes = {}
        for node in self.root.findall('node'):
            node_id = node.attrib['id']
            lat = float(node.attrib['lat'])
            lon = float(node.attrib['lon'])
            nodes[node_id] = (lon, lat)
        return nodes

    def _extract_ways(self) -> List[List[Tuple[float, float]]]:
        ways = []
        for way in self.root.findall('way'):
            coords = []
            for nd in way.findall('nd'):
                ref = nd.attrib['ref']
                if ref in self.nodes:
                    coords.append(self.nodes[ref])
            if len(coords) >= 2:
                ways.append(coords)
        return ways
    

class OSMRenderer:
    """
    A class for rendering OpenStreetMap (OSM) ways as an image.
    This class takes a list of ways (each represented as a list of coordinate tuples)
    and generates a rendered image of these ways using Matplotlib. The rendered image
    is returned as a PIL Image object.
    Attributes:
        ways (List[List[Tuple[float, float]]]): A list of ways, where each way is a list
            of (latitude, longitude) coordinate tuples.
        size (Tuple[int, int]): The size of the rendered image in pixels (width, height).
            Defaults to (512, 512).
    Methods:
        render() -> Image.Image:
            Renders the OSM ways as an image and returns it as a PIL Image object.
    """

    def __init__(self, ways: List[List[Tuple[float, float]]], size: Tuple[int, int] = (512, 512)):
        self.ways = ways
        self.size = size

    def render(self) -> Image.Image:
        fig = plt.Figure(figsize=(self.size[0]/100, self.size[1]/100), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        for coords in self.ways:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, linewidth=1, color='black')

        ax.axis('off')
        ax.set_aspect('equal')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        return Image.open(buf)