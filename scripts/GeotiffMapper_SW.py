import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
import overpy
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, MultiLineString, Polygon
from shapely.ops import unary_union
from rasterio.plot import show
import pyproj
import osmnx as ox
import geopandas as gpd
import pickle
import math
import json
import os
import numpy as np
import shapely
from shapely.ops import transform
from shapely.geometry import Point, MultiPoint, GeometryCollection

class GeotiffMapper:
    """
    This class extracts a road network
    from a given geotiff image input.
    
    Attributes:
        geotiff_path (string): The path to the geotiff image.
        image_name (string): The name of the image file without the path.
        default_bounds (rasterio.coords.BoundingBox): Bounds of the input image.
        geotiff_crs (rasterio.crs.CRS): The current CRS of the input image.
        wgs84_bounds (tuple): Transformed image bounds to WGS84 coordinates.
    """
    def __init__(self, geotiff_path):
        self.geotiff_path = geotiff_path
        self.image_name = os.path.basename(geotiff_path)
        self.default_bounds = None
        self.wgs84_bounds = None
        self.geotiff_crs = None

    def get_geotiff_bounds(self):
        """
        Reads the geotiff image and obtains the
        WGS84 bounds.

        Initializes self.wgs84_bounds and self.geotiff_crs.
        """
        with rasterio.open(self.geotiff_path) as src:
            self.default_bounds = src.bounds
            self.geotiff_crs = src.crs
            self.wgs84_bounds = transform_bounds(src.crs, 'EPSG:4326', *self.default_bounds)
    
    # 2. Retrieve OSM road data
    def get_osm_roads(self):
        """
        Collecting OSM road network data based on the 
        WGS84 bounds of the image.

        Returns:
            result.ways: A list of overpy.Way values extracted from the geotiff.
        """
        if self.wgs84_bounds is None:
            self.get_geotiff_bounds()  # Ensure bounds are initialized

        api = overpy.Overpass()

        min_lon, min_lat, max_lon, max_lat = self.wgs84_bounds  # Unpack bounds

        '''
        query = f"""
        [out:json][timeout:90];
        (
            nwr["highway"="footway"]["footway"="sidewalk"]({min_lat}, {min_lon}, {max_lat}, {max_lon});
            nwr["highway"="footway"]["footway"="crossing"]({min_lat}, {min_lon}, {max_lat}, {max_lon});
        );
        (._;>;);
        out body;
        """
        '''

        query = f"""[out:json][timeout:25];
        (
        way["footway"="sidewalk"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["footway"="crossing"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="footway"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="pedestrian"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="path"]["foot"!="no"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="steps"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["ramp"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["incline"]["highway"="footway"]({min_lat},{min_lon},{max_lat},{max_lon});

        /* Roads */
        way["highway"="residential"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="primary"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="secondary"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="tertiary"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="unclassified"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="service"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="track"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="trunk"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["highway"="motorway"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

        result = api.query(query)
        return result.ways

    @staticmethod
    def _meters_to_degrees(lat, meters): 
        # Constants 
        earth_radius = 6378137  # Earth's radius in meters (WGS84)
        
        # Degrees per meter for latitude 
        lat_deg_per_meter = 1 / (2 * math.pi * earth_radius) * 360 

        # Adjust for longitude
        lon_deg_per_meter = lat_deg_per_meter / math.cos(math.radians(lat))
        return meters * np.hypot(lat_deg_per_meter, lon_deg_per_meter)
    
    @staticmethod
    def calc_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points on the Earth
        using the Haversine formula.
        
        Args:
            lat1, lon1: Coordinates of the first point (in degrees).
            lat2, lon2: Coordinates of the second point (in degrees).
            
        Returns:
            Distance in meters.
        """
        R = 6371000  # Earth radius in meters
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
    
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
    
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))
    
    def create_road_graph(self, ways, spacing_meters=10.0):
        """
        Builds the road network graph as a networkx graph object.
        
        Args:
            ways (list): A list of overpy.Way values extracted from the geotiff.
        
        Returns:
            G: A networkx graph of the extracted road network.
        """
        G = nx.Graph()
        lines = []
        
        # Transformer to convert from WGS84 (EPSG:4326) to GeoTIFF CRS
        proj_from = pyproj.Transformer.from_crs('epsg:4326', self.geotiff_crs, always_xy=True)
        
        # Create image bounds polygon
        image_polygon = Polygon([
            (self.default_bounds.left, self.default_bounds.bottom),
            (self.default_bounds.right, self.default_bounds.bottom),
            (self.default_bounds.right, self.default_bounds.top),
            (self.default_bounds.left, self.default_bounds.top),
            (self.default_bounds.left, self.default_bounds.bottom)
        ])

    
        for way in ways:
            coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
            # Reproject coordinates from WGS84 to GeoTIFF CRS
            reprojected_coords = [proj_from.transform(lon, lat) for lon, lat in coords]
            
            line = LineString(reprojected_coords)
            spacing_degrees = self._meters_to_degrees(line.centroid.y, spacing_meters)
            #line = shapely.segmentize(line, spacing_degrees)
            
        
            # Intersect the LineString with the image polygon
            clipped_line = line.intersection(image_polygon)
            if clipped_line.is_empty:
                continue
    
            # Fetch road type, surface type, and number of lanes
            road_type    = way.tags.get('highway',  'unknown')
            surface_type = way.tags.get('surface',  'unknown')
            lanes        = way.tags.get('lanes',    'unknown')
            footway_type = way.tags.get('footway',  'unknown')  # NEW

        
            # Instead of using spacing_degrees and a non-existent shapely.segmentize,
            # we use our custom segmentize_linestring function.
            if isinstance(clipped_line, LineString):
                segmentized_segments = segmentize_linestring(clipped_line, spacing_meters)
            elif isinstance(clipped_line, MultiLineString):
                segmentized_segments = []
                for geom in clipped_line.geoms:
                    segmentized_segments.extend(segmentize_linestring(geom, spacing_meters))
            else:
                segmentized_segments = []

            for seg in segmentized_segments:
                coords = list(seg.coords)
                for i in range(len(coords) - 1):
                    u = coords[i]
                    v = coords[i + 1]
                    G.add_edge(u, v, road_type=road_type, surface_type=surface_type, lanes=lanes, footway=footway_type)
                    
        # Identify intersections
        if lines:
            intersections = unary_union(lines).intersection(unary_union(lines))
            if isinstance(intersections, Point):
                G.add_node((intersections.x, intersections.y))
            elif hasattr(intersections, 'geoms'):
                for point in intersections.geoms:
                    if isinstance(point, Point):
                        G.add_node((point.x, point.y))
        
        # Saving node pos argument
        node_positions = {node: (node[0], node[1]) for node in G.nodes()}
        nx.set_node_attributes(G, node_positions, 'pos')
        
        # Set the image name for each node
        node_image_names = {node: self.image_name for node in G.nodes()}
        nx.set_node_attributes(G, node_image_names, 'image_name')
    
        # Collect road type, surface type, and number of lanes for each node
        node_road_types = {}
        node_surface_types = {}
        node_lanes = {}
        
        for node in G.nodes():
            road_types = set()
            surface_types = set()
            lanes_values = set()
        
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                road_type = edge_data.get('road_type', 'unknown')
                surface_type = edge_data.get('surface_type', 'unknown')
                lanes = edge_data.get('lanes', 'unknown')
                
                road_types.add(road_type)
                surface_types.add(surface_type)
                lanes_values.add(lanes)
            
            # Get the first road type, surface type, and lanes value from the sets
            first_road_type = sorted(road_types)[0] if road_types else 'unknown'
            first_surface_type = sorted(surface_types)[0] if surface_types else 'unknown'
            first_lanes = sorted(lanes_values)[0] if lanes_values else 'unknown'
            
            node_road_types[node] = first_road_type
            node_surface_types[node] = first_surface_type
            node_lanes[node] = first_lanes
        
        # Set road type, surface type, and lanes as node attributes
        nx.set_node_attributes(G, node_road_types, 'road_type')
        nx.set_node_attributes(G, node_surface_types, 'surface_type')
        nx.set_node_attributes(G, node_lanes, 'lanes')
        
        G = self.label_nodes(G)
        return G
    

    def render_map(self, G, output_path="road_network.png"):
        """
        Displays the road network graph over the input geotiff image and ensures that
        parts of the graph outside the image bounds are still visible.
        
        Args:
            G (networkx.classes.graph.Graph): A networkx graph of the extracted road network.
            output_path (str): Path to save the output PNG file.
        """
        with rasterio.open(self.geotiff_path) as src:
            img = src.read(1)
            bounds = src.bounds
            image_extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
            
            pos = nx.get_node_attributes(G, 'pos')
            
            xs = [coord[0] for coord in pos.values()]
            ys = [coord[1] for coord in pos.values()]
            graph_left = min(xs) if xs else bounds.left
            graph_right = max(xs) if xs else bounds.right
            graph_bottom = min(ys) if ys else bounds.bottom
            graph_top = max(ys) if ys else bounds.top
            
            union_left = min(bounds.left, graph_left)
            union_right = max(bounds.right, graph_right)
            union_bottom = min(bounds.bottom, graph_bottom)
            union_top = max(bounds.top, graph_top)
            
            union_extent = (union_left, union_right, union_bottom, union_top)

            margin_x = 0.05 * (union_right - union_left)
            margin_y = 0.05 * (union_top - union_bottom)
            union_extent_with_margin = (
                union_left - margin_x,
                union_right + margin_x,
                union_bottom - margin_y,
                union_top + margin_y
            )
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img, cmap='gray', extent=image_extent)
            nx.draw(G, pos=pos, node_color='r', ax=ax, edge_color='b', with_labels=False, node_size=1, width=1)
            
            ax.set_xlim(union_extent_with_margin[0], union_extent_with_margin[1])
            ax.set_ylim(union_extent_with_margin[2], union_extent_with_margin[3])
            
            plt.axis('off')
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            plt.tight_layout()
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()

    def classify_node(self, graph, node):
        """
        Classifies a node as one of: straight, end, corner, intersection.
        
        Args:
             graph (networkx.classes.graph.Graph): A networkx graph.
             node (tuple): A node from the graph to classify.
        """
        neighbors = list(graph.neighbors(node))
        degree = len(neighbors)
    
        if degree == 1:
            return 'end'
    
        if degree == 2:
            node_pos = graph.nodes[node]['pos']
            pos1 = graph.nodes[neighbors[0]]['pos']
            pos2 = graph.nodes[neighbors[1]]['pos']
    
            line1 = LineString([node_pos, pos1])
            line2 = LineString([node_pos, pos2])
    
            angle = self.calculate_angle_between_lines(line1, line2)
            if 170 <= angle <= 190:
                return 'straight'
            else:
                return 'corner'
    
        if degree >= 3:
            return 'intersection'
    
    def calculate_angle_between_lines(self, line1, line2):
        """
        Calculates the angle between two lines connected to a node.
        
        Args:
             line1 (LineString): Edge connecting one neighbor.
             line2 (LineString): Edge connecting another neighbor.
             
        Returns:
            Angle between the two lines in degrees.
        """
        vector1 = (line1.coords[1][0] - line1.coords[0][0], line1.coords[1][1] - line1.coords[0][1])
        vector2 = (line2.coords[1][0] - line2.coords[0][0], line2.coords[1][1] - line2.coords[0][1])
    
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
            
        angle = math.degrees(math.acos(cos_angle))
        return angle
        
    def label_nodes(self, graph):
        """
        Assigns a label to each node: straight, end, corner, or intersection.
        
        Args:
             graph (networkx.classes.graph.Graph): A networkx graph.
             
        Returns:
            The graph with nodes labeled.
        """
        for node in graph.nodes():
            label = self.classify_node(graph, node)
            graph.nodes[node]['label'] = label
        return graph
             
    def graph_to_geojson(self, G, edge_filename, node_filename):
        """
        Converts the networkx graph into separate GeoJSON files for edges and nodes.
    
        Args:
            G (networkx.classes.graph.Graph): The networkx graph.
            edge_filename (str): Filename for the edges GeoJSON.
            node_filename (str): Filename for the nodes GeoJSON.
        """
        proj_to_wgs84 = pyproj.Transformer.from_crs(self.geotiff_crs, 'EPSG:4326', always_xy=True)
    
        edge_features = []
        node_features = []
    
        for u, v, data in G.edges(data=True):
            u_pos = G.nodes[u]['pos']
            v_pos = G.nodes[v]['pos']
    
            u_lonlat = proj_to_wgs84.transform(u_pos[0], u_pos[1])
            v_lonlat = proj_to_wgs84.transform(v_pos[0], v_pos[1])
    
            geometry = {
                "type": "LineString",
                "coordinates": [u_lonlat, v_lonlat]
            }
    
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": data
            }
    
            edge_features.append(feature)
    
        for n, node_data in G.nodes(data=True):
            pos = node_data['pos']
            lonlat = proj_to_wgs84.transform(pos[0], pos[1])
    
            geometry = {
                "type": "Point",
                "coordinates": lonlat
            }
    
            properties = {
                "label": node_data.get('label', ''),
                "image_name": node_data.get('image_name', ''),
                "road_types": node_data.get('road_types', '')
            }
    
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties
            }
    
            node_features.append(feature)
    
        edge_feature_collection = {
            "type": "FeatureCollection",
            "features": edge_features
        }
    
        node_feature_collection = {
            "type": "FeatureCollection",
            "features": node_features
        }
    
        with open(edge_filename, 'w') as f:
            json.dump(edge_feature_collection, f)
    
        with open(node_filename, 'w') as f:
            json.dump(node_feature_collection, f)

def segmentize_linestring(line, segment_length):
    """
    Breaks a LineString into smaller segments so that the distance
    between consecutive points is at most segment_length.
    
    Args:
        line (shapely.geometry.LineString): The input line.
        segment_length (float): Desired spacing (same units as line.length).
    
    Returns:
        List[LineString]: List of LineString segments.
    """
    if segment_length <= 0 or line.length <= segment_length:
        return [line]
    
    num_segments = int(math.ceil(line.length / segment_length))
    points = [line.interpolate(i * segment_length) for i in range(num_segments)]
    if points[-1].distance(Point(line.coords[-1])) > 1e-8:
        points.append(Point(line.coords[-1]))
    
    segments = [
        LineString([points[i].coords[0], points[i + 1].coords[0]])
        for i in range(len(points) - 1)
    ]
    return segments
