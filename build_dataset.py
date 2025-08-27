import os
import pickle
import requests
import argparse
from PIL import Image
import xml.etree.ElementTree as ET

from modules.utils.osm_utils import OSMExtractor
from modules.utils.tile_utils import TileExtractor
from modules.utils.satellite_utils import TIFProcessor
from modules.utils.graph_utils import GraphExtractor

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, time
from urllib.parse import urlencode

NET_SEM = threading.Semaphore(1)   # serialize network so threads don't hammer API
RATE_DELAY = 1.0                   # seconds between requests (tune to be polite)
MAX_TRIES = 6                      # retry with exponential backoff on 429/5xx
USER_AGENT = "MiamiCEC-NodeFetcher/1.0"


# ─── replace your get_lat_lon_from_node_id with this version ───────────────────
def get_lat_lon_from_node_id(node_id):
    """
    Polite, retrying fetch to avoid Overpass/OSM rate limits.
    """
    base = "https://api.openstreetmap.org/api/0.6/node/"
    url = f"{base}{node_id}"
    backoff = 1.0

    for attempt in range(1, MAX_TRIES + 1):
        with NET_SEM:  # only one in-flight API call at a time
            try:
                time.sleep(RATE_DELAY)  # pace requests
                response = requests.get(
                    url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=10,
                )

                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    node = root.find("node")
                    if node is not None:
                        lat = float(node.attrib["lat"])
                        lon = float(node.attrib["lon"])
                        return lat, lon
                    return None

                if response.status_code == 404:
                    return None

                if response.status_code in (429, 500, 502, 503, 504):
                    # honor Retry-After if present, then back off
                    ra = response.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        time.sleep(int(ra))
                    else:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60)  # cap at 60s
                    continue

                # other non-OKs: raise
                raise Exception(f"OSM API error: {response.status_code}: {response.text[:200]}")

            except requests.RequestException as e:
                if attempt == MAX_TRIES:
                    print(f"[ERROR] Request failed for node {node_id} after {attempt} tries: {e}")
                    return None
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    return None

def save_graph(parent_dir):
    osm_dir = os.path.join(parent_dir, "osm.xml")
    graph_extractor = GraphExtractor(osm_file_path=osm_dir)

    main_graph = graph_extractor.extract_graph()

    # keeping nodes with degree > 0
    connected = [node for node in main_graph.nodes if main_graph.degree[node] > 0]

    sub_graph = main_graph.subgraph(connected).copy()
    graph_dir = os.path.join(parent_dir, "graph.pkl")

    with open(graph_dir, "wb") as file:
        pickle.dump(sub_graph, file)


def process_single_node(node_id, tif_processor, tile_extractor, osm_extractor, output_dir):
    latlon = get_lat_lon_from_node_id(node_id)
    if latlon is None:
        return f"[WARN] Node {node_id} not found or invalid."

    lat, lon = latlon

    try:
        patch_img = tif_processor.get_patch(lat, lon)
        if patch_img.size != (224, 224):
            return f"[SKIP] Node {node_id} patch is {patch_img.size}, not 224x224."
        
        osm_xml = osm_extractor.get_osmxml(lat, lon)
        tile_img = tile_extractor.get_tile_image(lat, lon)
        pixel_coords = tif_processor.latlon_to_pixel(lat, lon)

        # Save tile data
        tile_img = tile_img.convert("RGB")

        # Save the image as RGB (e.g., as JPEG or PNG)
        node_dir = os.path.join(output_dir, str(node_id))
        os.makedirs(node_dir, exist_ok=True)

        # Save images
        tile_img.save(os.path.join(node_dir, "tile.png"))
        patch_img.save(os.path.join(node_dir, "satellite.png"))

        # Save OSM XML
        with open(os.path.join(node_dir, "osm.xml"), "w", encoding="utf-8") as f:
            f.write(osm_xml)

        save_graph(node_dir)

        # Save metadata
        with open(os.path.join(node_dir, "metadata.txt"), "w") as f:
            f.write(f"Lat: {lat}\nLon: {lon}\nPixel: {pixel_coords}\n")

        return f"[OK] Node {node_id} processed."

    except Exception as e:
        node_dir = os.path.join(output_dir, str(node_id))
        if os.path.exists(node_dir):
            try:
                os.rmdir(node_dir)
            except OSError:
                for root, dirs, files in os.walk(node_dir, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
            os.rmdir(node_dir)
        return f"[ERROR] Failed to process node {node_id}: {e}"


def process_node_ids_parallel(
    node_file_path: str,
    tif_path: str,
    output_dir: str = "output",
    zoom: int = 18,
    max_workers: int = 8
):
    # Step 1: Read and deduplicate node IDs
    with open(node_file_path, "r") as f:
        node_ids = list(set([line.strip() for line in f if line.strip().isdigit()]))

    tif_processor = TIFProcessor(tif_path=tif_path, window_size=(224, 224))
    osm_extractor = OSMExtractor(zoom=zoom)
    tile_extractor = TileExtractor(zoom=zoom)
    os.makedirs(output_dir, exist_ok=True)

    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_node, node_id, tif_processor, tile_extractor, osm_extractor, output_dir
            ): node_id for node_id in node_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing nodes"):
            result = future.result()
            print(result)


def process_directory(
    node_dir: str,
    tif_dir: str,
    output_dir: str,
    zoom: int,
    max_workers: int
):
    """
    Iterate over all .tif/.tiff files in tif_dir.
    For each tif file name F, look for nodes_{F}.txt inside node_dir.
    If found, process; if not, skip.
    Writes outputs under: output_dir/<basename_of_F_without_ext>/
    """
    if not os.path.isdir(node_dir):
        raise ValueError(f"node_dir does not exist or is not a directory: {node_dir}")
    if not os.path.isdir(tif_dir):
        raise ValueError(f"tif_dir does not exist or is not a directory: {tif_dir}")

    tif_files = [
        f for f in os.listdir(tif_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ]
    if not tif_files:
        print(f"[WARN] No .tif/.tiff files found in {tif_dir}. Nothing to do.")
        return

    print(f"[INFO] Found {len(tif_files)} GeoTIFF(s) in {tif_dir}.")

    for tif_name in tif_files:
        tif_path = os.path.join(tif_dir, tif_name)
        tif_base = os.path.splitext(tif_name)[0]  
        node_file_name = f"nodes_{tif_base}.txt"  # include extension per spec
        node_file_path = os.path.join(node_dir, node_file_name)

        if not os.path.isfile(node_file_path):
            print(f"[SKIP] No node file for {tif_name} (expected {node_file_name}). Skipping.")
            continue

        sub_out_dir = output_dir

        print(f"[RUN] Processing: TIF={tif_name}, NODES={node_file_name}, OUT={sub_out_dir}")
        process_node_ids_parallel(
            node_file_path=node_file_path,
            tif_path=tif_path,
            output_dir=sub_out_dir,
            zoom=zoom,
            max_workers=max_workers,
        )
        print(f"[DONE] Completed {tif_name}. Results in: {sub_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-process a directory of GeoTIFFs. For each TIF, look for nodes_{<tif file name>}.txt in node_dir."
    )
    parser.add_argument("--node_dir", type=str, required=True, help="Directory containing nodes_{<tif file name>}.txt files.")
    parser.add_argument("--tif_dir", type=str, required=True, help="Directory containing GeoTIFF files.")
    parser.add_argument("--output_dir", type=str, default="output", help="Top-level directory to save outputs.")
    parser.add_argument("--zoom", type=int, default=18, help="Zoom level for OSM tile extraction.")
    parser.add_argument("--workers", type=int, default=8, help="Number of threads for parallel processing.")

    args = parser.parse_args()

    process_directory(
        node_dir=args.node_dir,
        tif_dir=args.tif_dir,
        output_dir=args.output_dir,
        zoom=args.zoom,
        max_workers=args.workers,
    )
    print(f"Batch processing complete. All outputs have been saved under: {args.output_dir}")
