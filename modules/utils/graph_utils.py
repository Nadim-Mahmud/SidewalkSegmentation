""" 
Copyright @ 2025 Siddhant S. Karki
Computer Science & Software Engineering Dept.
Miami University OH
"""

import xml.etree.ElementTree as ET
import networkx as nx

class GraphExtractor:
    """
    Extracts a NetworkX graph from an OSM XML file.
    Parses nodes and highway-tagged ways to build a directed road network graph.
    """

    def __init__(self, osm_file_path: str):
        """
        Initialize the GraphExtractor with the OSM file path.

        Args:
            osm_file_path (str): Path to the .osm or .osm.xml file.
        """
        self.osm_file_path = osm_file_path
        self.graph = nx.DiGraph()
        self.node_map = {}
        self.namespace = {}
        self.root = None

    def load_osm(self):
        """Parse the OSM XML file and set the root element."""
        tree = ET.parse(self.osm_file_path)
        self.root = tree.getroot()

        if 'osm' in self.root.tag:
            self.namespace = {'osm': 'http://www.openstreetmap.org/osm/0.6'}

    def parse_nodes(self):
        """Extract node elements and add them as graph nodes."""
        for node in self.root.findall("node", self.namespace):
            node_id = node.attrib['id']
            lat = float(node.attrib['lat'])
            lon = float(node.attrib['lon'])
            self.node_map[node_id] = (lat, lon)
            self.graph.add_node(node_id, pos=(lat, lon))

    def parse_ways(self):
        """Extract highway-tagged way elements and add them as graph edges."""
        for way in self.root.findall("way", self.namespace):
            nds = [nd.attrib['ref'] for nd in way.findall("nd", self.namespace)]
            tags = {
                tag.attrib['k']: tag.attrib['v']
                for tag in way.findall("tag", self.namespace)
            }

            if 'highway' in tags:
                for u, v in zip(nds, nds[1:]):
                    if u in self.node_map and v in self.node_map:
                        self.graph.add_edge(u, v, highway=tags['highway'])

    def extract_graph(self) -> nx.DiGraph:
        """
        Execute the full extraction pipeline and return the graph.

        Returns:
            networkx.DiGraph: Extracted road network graph.
        """
        self.load_osm()
        self.parse_nodes()
        self.parse_ways()
        return self.graph