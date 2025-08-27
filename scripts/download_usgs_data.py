import os
from re import A
import sys
import requests
import datetime
import random
import pandas as pd
import json
import csv
import numpy as np
import ast  # For safely parsing dictionary strings
import time
from tqdm import tqdm, trange
from dotenv import load_dotenv

# Import from our custom USGS M2M module
from modules.usgs_m2m import (
    LOGIN_URL, send_request, login, logout, search, 
    download_options, download_request, download_retrieve,
    download_remove, remove_multiple_downloads, get_all_downloads,
    purge_all_available_downloads
)

# Load environment variables
load_dotenv()

# -----------------------------------------
# CONFIGURATION: Set parameters here
# -----------------------------------------
USGS_DOWNLOADS_CONFIG = {
    # API Authentication - defaults to environment variables
    'username': os.getenv('USGS_USERNAME', None),               
    'token': os.getenv('USGS_TOKEN', None),              
    
    # Input/Output settings
    'input_csv_path': 'city_bounding_boxes.csv',
    'download_path': 'data/usgs_tifs',          
    
    # Dataset parameters
    'dataset': 'high_res_ortho',               
    'start_date': '2018-01-01',               
    'end_date': '2020-12-31',                 
    
    # Search and download limits
    'max_downloads': 100,                    
    'downloads_per_city': 100,                          
    
    # Output preferences
    'get_polygon_geojson': False,            
    
    # Image filter preferences
    'res_unit_val': None, # 1.0,                     
    'res_unit_name': None, # 'METER',                  
    'num_bands': None, # 3,
    
    # Download preparation settings
    'wait_for_preparing': True,             # Whether to wait for preparing downloads
    'prepare_check_interval': 10,           # Seconds between checking preparing downloads status  
    'prepare_max_wait_time': 3600,          # Maximum seconds to wait for preparing downloads
    'skip_existing_files': True,            # Skip downloading files that already exist locally
}

def input_csv(csv_path):
    """
    Convert CSV to the desired list format.
    
    Args:
        csv_path (str): Path to the CSV file containing city information.
        
    Returns:
        list: A list of dictionaries containing city names and bounding box data.
    """
    result = []

    # Open the CSV file
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)

        # Iterate over rows in the CSV
        for row in reader:
            city = row['city']
            bounding_box = eval(row['bounding_box'])  # Convert string representation to dict

            # Append the structured dictionary to the result list
            result.append({
                'city': city,
                'bounding_box': bounding_box
            })

    return result

def get_resolutions(result_list):
    """
    Extract resolution values from result list.
    
    Args:
        result_list (list): List of metadata results from USGS API.
        
    Returns:
        list: List of resolution values extracted from the metadata.
    """
    resolutions = []
    for item in result_list:
        # Loop through results for resolution values. Save in a list.
        resolutions.append([
            float(entry['value']) for entry in item
            if entry['fieldName'] == 'Resolution' and entry['value'].replace('.', '', 1).isdigit()
        ])
    return resolutions

def get_data_size(result_list):
    """
    Extract data size values from result list.
    
    Args:
        result_list (list): List of metadata results from USGS API.
        
    Returns:
        list: List of data sizes extracted from the metadata.
    """
    dataset_sizes = []
    for item in result_list:
        # Loop through results for data size values. Save in a list.
        dataset_sizes.append([
            int(entry['value']) for entry in item
            if entry['fieldName'] == 'Dataset Size' and entry['value']
        ])
    return dataset_sizes

def save_spatial_bounds(spat_bounds, property_vals, output_path='spatial_bounds.geojson'):
    """
    Save spatial bounds as GeoJSON.
    
    Args:
        spat_bounds (list): List of spatial bounds to save.
        property_vals (list): List of property values for each spatial bound.
        output_path (str): Path to save the GeoJSON file.
    """
    # Initialize an empty list to hold GeoJSON features
    features = []

    # Iterate over each group of polygons
    idx = 0

    for group in spat_bounds:
        for polygon in group:
            # Wrap each polygon in a GeoJSON Feature
            feature = {
                "type": "Feature",
                "geometry": polygon,
                "properties": property_vals[idx]  # Add properties as needed
            }
            features.append(feature)
            idx += 1

    # Create the GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    # Output as a JSON string (or save to a file)
    geojson_str = json.dumps(geojson, indent=2)

    # Optionally, save the GeoJSON to a file
    with open(output_path, 'w') as file:
        file.write(geojson_str)

    tqdm.write(f"Spatial bounds data saved to {output_path}")

def search_city_geotiffs(api_key, dataset_name, city_bounding_boxes, temporal_filter, config, spat_bounds, property_vals):
    """
    Search for GeoTIFFs from a list of cities.
    
    Args:
        api_key (str): USGS API key for authentication.
        dataset_name (str): Name of the dataset to search.
        city_bounding_boxes (list): List of city dictionaries with bounding boxes.
        temporal_filter (dict): Temporal filter parameters for the search.
        config (dict): Configuration parameters.
        spat_bounds (list): List to store spatial bounds information.
        property_vals (list): List to store property values.
    """
    for item in tqdm(city_bounding_boxes, desc="Processing Cities"):
        # Extract city and bounding box from input csv.
        city = item['city']
        spatial_filter = item['bounding_box']
        
        # Convert bounds to float values if they are np.floats.
        spatial_filter['lowerLeft']['latitude'] = float(spatial_filter['lowerLeft']['latitude'])
        spatial_filter['lowerLeft']['longitude'] = float(spatial_filter['lowerLeft']['longitude'])  
        spatial_filter['upperRight']['latitude'] = float(spatial_filter['upperRight']['latitude'])
        spatial_filter['upperRight']['longitude'] = float(spatial_filter['upperRight']['longitude'])

        # Send search request to M2M API.
        tqdm.write(f"Searching for GeoTIFFs for {city}...")
        search_results = search(api_key, dataset_name, spatial_filter, temporal_filter, maxResults=config['max_downloads'])

        if search_results['results']:

            # Get entity IDs and metadata from the search results.
            entity_ids, results_metadata = extract_entity_ids_and_metadata(search_results)

            tqdm.write(f"Found {len(entity_ids)} Entity IDs for {city}.")

            # Get available download option results from API call.
            options_results = download_options(api_key, dataset_name, entity_ids)

            # Get the IDs of items that are available for download.
            downloads = prepare_downloads(options_results, results_metadata)
            
            # Filter downloads based on criteria before checking for existing files
            original_count = len(downloads)
            downloads = [d for d in downloads if 
                        (config['num_bands'] is None or d['numBands'] == str(config['num_bands']) and
                        (config['res_unit_val'] is None or str(config['res_unit_val']) == d['resolution']) and
                        (config['res_unit_name'] is None or config['res_unit_name'] == d['unitsOfResolution']))]
            filtered_count = original_count - len(downloads)
            if filtered_count > 0:
                tqdm.write(f"Filtered out {filtered_count} files that don't match criteria for {city}")
            
            # Remove any downloads that are already downloaded
            if config.get('skip_existing_files', True):
                original_count = len(downloads)
                downloads = [d for d in downloads if not os.path.exists(os.path.join(config['download_path'], f"{d['imageName']}.zip"))]
                skipped_count = original_count - len(downloads)
                if skipped_count > 0:
                    tqdm.write(f"Skipped {skipped_count} files that already exist for {city}")
            
            # Shuffle the downloads to randomize the order.
            random.shuffle(downloads)
            
            # Select a maximum number of downloads per city.
            if config['downloads_per_city'] is not None:
                downloads = downloads[:int(config['downloads_per_city'])]

            if downloads:
                # Begin zip download process.
                process_city_downloads(api_key, downloads, city, search_results, config, spat_bounds, property_vals)
            else:
                tqdm.write(f"No downloads to process for {city} after filtering.")
        else:
            tqdm.write(f"No results found for {city}.")

def extract_entity_ids_and_metadata(search_results):
    """
    Extract entity IDs and metadata from search results.
    
    Args:
        search_results (dict): Results from the USGS API search.
        
    Returns:
        tuple: (entity_ids, results_metadata) containing extracted IDs and metadata.
    """
    entity_ids = [result['entityId'] for result in search_results['results']]
    results_metadata = [result['metadata'] for result in search_results['results']]
    return entity_ids, results_metadata

def prepare_downloads(options_results, results_metadata):
    """
    Prepare downloads from options results.
    
    Args:
        options_results (list): List of download options from USGS API.
        results_metadata (list): List of metadata for each result.
        
    Returns:
        list: List of download items with extracted metadata.
    """
    downloads = []
    for option in options_results:
        # Extract entity and product IDs.
        image_name = "No Image Name Found"
        units_of_resolution = "No Resolution Units Found"
        number_of_bands = "No Number of Bands Found"
        resolution = "No Resolution Found"

        for item in results_metadata:
            found_entity_ID = False

            # Within the metadata, we need to find the attributes that correspond to the current
            # entity ID from the options_results. 
            for metadata in item:
                if metadata['fieldName'] == "Entity ID" and metadata['value'] == option['entityId']:
                    # Entity ID has been found.
                    found_entity_ID = True
                    continue

                # If the entity ID has been found, search for the specific attributes.
                if found_entity_ID and metadata['fieldName'] == "Image Name":
                    image_name = metadata['value']

                elif found_entity_ID and metadata['fieldName'] == "Units of Resolution":
                    units_of_resolution = metadata['value']

                elif found_entity_ID and metadata['fieldName'] == "Number of Bands":
                    number_of_bands = metadata['value']

                elif found_entity_ID and metadata['fieldName'] == "Resolution":
                    resolution = metadata['value']

        # Append all attributes to the downloads list.
        downloads.append({'entityId': option['entityId'], 'productId': option['id'], 'imageName': image_name,
                          'resolution': resolution, 'unitsOfResolution': units_of_resolution, 'numBands': number_of_bands})
    return downloads

def check_download_status(api_key, download_ids):
    """
    Check the status of downloads that are being prepared.
    
    Args:
        api_key (str): USGS API key for authentication.
        download_ids (list): List of download IDs to check.
        
    Returns:
        dict: Dictionary containing available and still preparing downloads.
    """
    # Create payload for download-retrieve request
    payload = {'downloadId': download_ids}
    
    # Send download-retrieve request to check status
    response = send_request("download-retrieve", payload, api_key)
    
    # Initialize return dictionaries
    available = []
    still_preparing = []
    
    # Check if downloads are available or still preparing
    if 'available' in response:
        available = response['available']
    
    if 'preparing' in response:
        still_preparing = response['preparing']
        
    return {'available': available, 'still_preparing': still_preparing}

def wait_for_downloads(api_key, preparing_downloads, downloads, config):
    """
    Wait for downloads that are being prepared to become available.
    
    Args:
        api_key (str): USGS API key for authentication.
        preparing_downloads (list): List of downloads being prepared.
        downloads (list): List of original download items with metadata.
        config (dict): Configuration parameters.
        
    Returns:
        list: List of downloads that have become available.
    """
    if not preparing_downloads:
        return []
    
    download_ids = [item['downloadId'] for item in preparing_downloads]
    wait_time = 0
    newly_available = []
    
    tqdm.write(f"Waiting for {len(download_ids)} preparing downloads to become available...")
    
    # Create a progress bar for the waiting process
    with tqdm(total=config['prepare_max_wait_time'], desc="Waiting for downloads", unit="s") as pbar:
        while download_ids and wait_time < config['prepare_max_wait_time']:
            # Wait for specified interval
            time.sleep(config['prepare_check_interval'])
            wait_time += config['prepare_check_interval']
            pbar.update(config['prepare_check_interval'])
            
            # Check status of preparing downloads
            status = check_download_status(api_key, download_ids)
            
            # Process newly available downloads
            if status['available']:
                tqdm.write(f"{len(status['available'])} downloads now available")
                
                # Add metadata from original downloads
                for avail in status['available']:
                    for item in downloads:
                        if item['entityId'] == avail['entityId']:
                            avail['imageName'] = item['imageName']
                            avail['resolution'] = item['resolution']
                            avail['unitsOfResolution'] = item['unitsOfResolution']
                            avail['numBands'] = item['numBands']
                            break
                
                newly_available.extend(status['available'])
                
                # Update the list of downloads still being prepared
                download_ids = [item['downloadId'] for item in status['still_preparing']]
    
    if download_ids:
        tqdm.write(f"Timed out waiting for {len(download_ids)} downloads")
    
    return newly_available

def process_city_downloads(api_key, downloads, city, search_results, config, spat_bounds, property_vals):
    """
    Process downloads for a specific city.
    
    Args:
        api_key (str): USGS API key for authentication.
        downloads (list): List of items to download.
        city (str): City name.
        search_results (dict): Results from the USGS API search.
        config (dict): Configuration parameters.
        spat_bounds (list): List to store spatial bounds information.
        property_vals (list): List to store property values.
    """


    label = f"{city}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    payload = {'downloads': downloads, 'label': label}

    # Request available downloads through API call.
    request_results = send_request("download-request", payload, api_key)
    
    # Handle both immediately available downloads and preparing downloads
    available_downloads = request_results.get('availableDownloads', [])
    preparing_downloads = request_results.get('preparingDownloads', [])
    
    tqdm.write(f"Processing {len(available_downloads)} available downloads and {len(preparing_downloads)} preparing downloads for {city}")
    
    # Process immediately available downloads - limit to 1 per city
    if available_downloads:
        process_available_downloads(api_key, available_downloads, downloads, city, search_results, config, spat_bounds, property_vals, max_downloads=100)
    
    # If we couldn't download from available, but have preparing downloads and are configured to wait
    elif config['wait_for_preparing'] and preparing_downloads:
        newly_available = wait_for_downloads(api_key, preparing_downloads, downloads, config)
        if newly_available:
            process_available_downloads(api_key, newly_available, downloads, city, search_results, config, spat_bounds, property_vals, max_downloads=100)


def process_available_downloads(api_key, available_downloads, downloads, city, search_results, config, spat_bounds, property_vals, max_downloads=None):
    """
    Process downloads that are immediately available.
    
    Args:
        api_key (str): USGS API key for authentication.
        available_downloads (list): List of available downloads.
        downloads (list): Original downloads list with metadata.
        city (str): City name.
        search_results (dict): Results from the USGS API search.
        config (dict): Configuration parameters.
        spat_bounds (list): List to store spatial bounds information.
        property_vals (list): List to store property values.
        max_downloads (int, optional): Maximum number of downloads to process. If None, use config value.
    """
    downloads_limit = max_downloads if max_downloads is not None else int(config['downloads_per_city'])
    count = 0
    skipped = 0
    downloaded = 0
    
    for result in tqdm(available_downloads, desc=f"{city} Downloads", leave=False):
        # Stop when download count is reached.
        if downloaded >= downloads_limit:
            tqdm.write(f"Reached download limit of {downloads_limit} for {city}")
            break
            
        # Keep track of current entity ID
        curr_entity_id = result['entityId']

        for item in downloads:
            # If the current entity ID matches the entity ID in the download,
            # add the item attributes to the result.
            if item['entityId'] == curr_entity_id:
                result['imageName'] = item['imageName']
                result['resolution'] = item['resolution']
                result['unitsOfResolution'] = item['unitsOfResolution']
                result['numBands'] = item['numBands']
                break
        else:
            continue  # Skip if no matching item found

        # Count is incremented for every image that passes filters
        count += 1

        if 'url' in result:
            # Check if the file already exists on disk
            filename = f"{result['imageName']}.zip"
            file_path = os.path.join(config['download_path'], filename)
            
            if config.get('skip_existing_files', True) and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 0:  # Only skip if file is not empty
                    tqdm.write(f"Skipping {result['imageName']} for {city} - file already exists ({file_size} bytes)")
                    skipped += 1
                    continue
            
            # Append spatial bounds of the selected city.
            current_bounds = [item['spatialBounds'] for item in search_results['results'] if item['entityId'] == curr_entity_id]

            property_vals.append({'imageName': result['imageName'], 'resolution': result['resolution'], 
                    'unitsOfResolution': result['unitsOfResolution'], 'numBands': result['numBands'], 'entityId': curr_entity_id})
            
            spat_bounds.append(current_bounds)

            # Download the file
            was_downloaded = not download_geotiff(result, city, config['download_path'], api_key=api_key)
            if was_downloaded:
                downloaded += 1
        else:
            tqdm.write(f"No download URL for result: {result['imageName'] if 'imageName' in result else curr_entity_id}")
    
    tqdm.write(f"Processed {count} images for {city}: {downloaded} downloaded, {skipped} skipped")

def download_geotiff(result, city, download_path, skip_existing=True, api_key=None):
    """
    Download a zip file from USGS.
    
    Args:
        result (dict): Result item containing download URL and metadata.
        city (str): City name.
        download_path (str): Path to save the downloaded file.
        skip_existing (bool, optional): Skip download if file already exists. Defaults to True.
        api_key (str, optional): USGS API key for authentication.
        
    Returns:
        bool: True if file was skipped, False otherwise.
    """
    download_url = result['url']
    download_id = result['downloadId']
    filename = f"{result['imageName']}.zip"
    
    # Ensure the download path exists
    os.makedirs(download_path, exist_ok=True)
    
    file_path = os.path.join(download_path, filename)
    
    # Check if file already exists and skip if needed
    if skip_existing and os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        if file_size > 0:  # Only skip if file is not empty
            tqdm.write(f"Skipping {result['imageName']} for {city} - file already exists ({file_size} bytes)")
            return True

    tqdm.write(f"Downloading {result['imageName']} for {city}...")
    
    try:
        file_response = requests.get(download_url, stream=True)
        if file_response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            tqdm.write(f"Downloaded {filename} to {download_path}")
            
            # remove the file from queue on the USGS server
            download_remove(api_key, download_id)
            return True
        else:
            tqdm.write(f"Failed to download {download_id} for {city}: HTTP {file_response.status_code}")
            return False
    except Exception as e:
        tqdm.write(f"Error downloading {download_id} for {city}: {str(e)}")
        return False



def load_city_bounding_boxes(csv_path):
    """
    Load city names and bounding boxes from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        list: A list of dictionaries containing city names and bounding box data.
    """
    if not os.path.isfile(csv_path):
        tqdm.write(f"Error: {csv_path} not found.")
        return []

    # Read CSV
    df = pd.read_csv(csv_path)

    # Convert bounding_box from string to dictionary
    # eval is unsafe but used here for simplicity and because I accidentlly use np.float32; ensure input is trusted.
    df['bounding_box'] = df['bounding_box'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Convert DataFrame to list of dictionaries
    city_bounding_boxes = df.to_dict(orient='records')
    
    return city_bounding_boxes

def download_usgs_images(config=None):
    """
    Main function to download USGS high-resolution orthoimagery based on configuration.
    
    Args:
        config (dict, optional): Configuration dictionary. If None, uses default CONFIG.
        
    Returns:
        int: Number of images successfully downloaded.
    """
    # Use default config if none provided
    if config is None:
        config = USGS_DOWNLOADS_CONFIG
    
    # Handle authentication
    username = config['username']
    token = config['token']
    
    if username is None:
        username = input('Enter your USGS M2M API Username: ')
    
    if token is None:
        import getpass
        token = getpass.getpass('Enter your USGS M2M API Token: ')

    # Dataset name
    dataset_name = config['dataset']

    # Temporal Filter
    temporal_filter = {
        'startDate': config['start_date'],
        'endDate': config['end_date']
    }
    
    # Retrieve bounding box data
    city_bounding_boxes = load_city_bounding_boxes(config['input_csv_path'])
    tqdm.write(f"Found {len(city_bounding_boxes)} cities to process")

    # Shuffle the city bounding boxes to randomize the order
    random.shuffle(city_bounding_boxes)

    # Login to the USGS API
    start_time = datetime.datetime.now()
    api_key = login(username, token)
    tqdm.write("API Key obtained.")
    
    # Purge all available downloads before starting if configured
    if config.get('purge_existing_downloads', True):
        tqdm.write("Purging any existing downloads...")
        purged = purge_all_downloads(api_key)
    
    # Default list for spatial bounds of results.
    spat_bounds = []
    property_vals = []
    
    total_downloaded = 0
    
    try:
        # Search for geotiff data of the specified cities from csv and begin download process.
        search_city_geotiffs(api_key, dataset_name, city_bounding_boxes, temporal_filter, 
                            config, spat_bounds, property_vals)

    finally:
        if config['get_polygon_geojson'] and spat_bounds:
            # Save spatial bounds of the downloads in one geojson file if requested.
            save_spatial_bounds(spat_bounds, property_vals)

        # Logout from the API
        logout(api_key)
        
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        tqdm.write(f"Download process completed in {elapsed_time:.1f} seconds.")
    
    return total_downloaded

def get_all_downloads(api_key):
    """
    Get all available downloads for the user.
    
    Args:
        api_key (str): USGS API key for authentication.
        
    Returns:
        list: List of available downloads.
    """
    try:
        # Send the download-retrieve request without specifying any download IDs
        response = send_request("download-retrieve", {}, api_key)
        
        # Extract available downloads
        available = response.get('available', [])
        preparing = response.get('preparing', [])
        
        return {'available': available, 'preparing': preparing}
    except Exception as e:
        tqdm.write(f"Error retrieving downloads: {str(e)}")
        return {'available': [], 'preparing': []}

def purge_all_downloads(api_key):
    """
    Remove all pending and available downloads for the user.
    
    Args:
        api_key (str): USGS API key for authentication.
        
    Returns:
        int: Number of downloads purged.
    """
    try:
        # Use the function from usgs_m2m.py
        result = purge_all_available_downloads(api_key)
        
        removed = result.get('removed', 0)
        tqdm.write(f"Successfully purged {removed} download(s)")
        return removed
    except Exception as e:
        tqdm.write(f"Error purging downloads: {str(e)}")
        return 0

if __name__ == '__main__':
    # Use the default configuration and simply call the main function
    download_usgs_images(USGS_DOWNLOADS_CONFIG)