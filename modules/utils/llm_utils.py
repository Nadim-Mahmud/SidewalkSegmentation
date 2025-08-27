import random
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import requests
import json



class UserPrompts:
    def __init__(self, prompts_dir):
        self.prompts = pd.read_csv(prompts_dir)
        self.questions = list(self.prompts["prompt"])

    def get_question(self, lat, lon):
        question = random.choice(self.questions)
        return question.format(lat=lat, lon=lon)
    

class FolderDataset:
    """
    Random-access dataset built from a directory of sub-folders:
        <root>/
            - <sample_id>/
                - satellite.png
                - tile.png
                - osm.xml
    Args
    ----
    root_dir : str | Path
        Directory whose *immediate* sub-folders are the samples.
    return_images : bool, default False
        - False -> keep image paths (faster, low memory).  
        - True  -> load and return `PIL.Image` objects.
    """

    FILE_MAP = {
        "node_id": None,
        "sat_image": "satellite.png",
        "tile_image": "tile.png",
        "osm_xml": "osm.xml",
    }

    def __init__(self, root_dir, return_images=False):
        self.root = Path(root_dir)
        if not self.root.is_dir():
            raise ValueError(f"Root directory not found: {self.root}")
        self._dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        if not self._dirs:
            raise ValueError(f"No sub-directories found in {self.root}")
        self.return_images = return_images

    def __len__(self):
        return len(self._dirs)

    def __getitem__(self, idx):
        """Random access by index *or* slice."""
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range.")

        return self._load_sample(self._dirs[idx])

    def _load_sample(self, sample_dir):
        sample = {"node_id": int(os.path.basename(sample_dir))}
        for key, fname in self.FILE_MAP.items():
            if key == "node_id":
                continue
            node_dir = Path(os.path.basename(sample_dir))
            node_file_path = node_dir / fname
            path = sample_dir / fname
            # path = Path(os.path.abspath(path))
            if not path.exists():
                raise FileNotFoundError(f"[{sample_dir.name}] Missing '{fname}'")
            if key.endswith("_image"):
                sample[key] = Image.open(path) if self.return_images else node_file_path
            else:
                sample[key] = path.read_text(encoding="utf-8")
        return sample

def get_lat_lon_from_node_id(node_id):
    url = f"https://api.openstreetmap.org/api/0.6/node/{node_id}"
    try:
        response = requests.get(url, headers={"User-Agent": "OSMNodeLookup/1.0"}, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            node = root.find("node")
            if node is not None:
                lat = float(node.attrib["lat"])
                lon = float(node.attrib["lon"])
                return lat, lon
        elif response.status_code == 404:
            return None
        else:
            raise Exception(f"OSM API error: {response.status_code}")
    except requests.RequestException as e:
        print(f"[ERROR] Request failed for node {node_id}: {e}")
        return None

# Define preprocessing functions
def process_and_save(dataset, output_folder, subset_name, questions_dir, num_images=2):
    json_data_list = []
    subset_folder = os.path.join(output_folder, subset_name)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)

    # For multiple images support, check: https://github.com/haotian-liu/LLaVA/issues/1292
    images_specifier_str = "\n".join(["<image>" for i in range(num_images)]) + "\n"
    prompt_processor = UserPrompts(questions_dir)
    
    for item in tqdm(dataset, desc=f"Building LLaVA JSON ({subset_name})"):
        unique_id = item["node_id"]
        sat_image_path  = str(item["sat_image"])
        tile_image_path = str(item["tile_image"])
        osmxml_output = item["osm_xml"]
    
        lat, lon = get_lat_lon_from_node_id(unique_id)
        prompt = prompt_processor.get_question(lat, lon)
    
        json_data = {
            "id": unique_id,
            "image": [sat_image_path, tile_image_path],
            "conversations": [
                {
                    "from": "human",
                    "value": images_specifier_str + prompt
                },
                {
                    "from": "gpt",
                    "value": osmxml_output
                }
            ]
        }
        json_data_list.append(json_data)
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, 'llava_dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)


def save_dataset(folder_dataset, output_folder, subset_name, questions_dir, val_samples=None):
    # Determine the split for training and validation
    if val_samples is not None and subset_name == 'train':
        train_dataset = folder_dataset[val_samples:]
        val_dataset = folder_dataset[:val_samples]
    else:
        train_dataset = folder_dataset
        val_dataset = []

    # Process and save the datasets
    for subset, data in [('train', train_dataset), ('validation', val_dataset)]:
        if data:
            process_and_save(data, output_folder, subset, questions_dir, 2)