import os
import requests
from dotenv import load_dotenv

# Load environment variables for USGS API credentials
load_dotenv()

# Base URL for the M2M API
BASE_URL = 'https://m2m.cr.usgs.gov/api/api/json/stable/'
LOGIN_URL = 'https://ers.cr.usgs.gov/profile/access'

class USGSApiError(Exception):
    """Exception raised for errors in the USGS API responses."""
    pass

def send_request(endpoint, payload, api_key=None):
    """
    Send a request to the USGS M2M API.
    
    Args:
        endpoint (str): API endpoint to call
        payload (dict): Request data
        api_key (str, optional): Authentication token
        
    Returns:
        dict: Response data
        
    Raises:
        USGSApiError: If the API returns an error or the HTTP request fails
    """
    headers = {}
    if api_key:
        headers['X-Auth-Token'] = api_key
    url = BASE_URL + endpoint
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data.get('errorCode') is None:
            return data['data']
        else:
            error_msg = f"API Error: {data['errorCode']} - {data['errorMessage']}"
            raise USGSApiError(error_msg)
    else:
        error_msg = f"HTTP Error {response.status_code}: {response.text}"
        raise USGSApiError(error_msg)

def login(username, token):
    """
    Login to the USGS M2M API using username and token.
    
    Args:
        username (str): USGS username
        token (str): USGS token or password
        
    Returns:
        str: API key for authenticated requests
        
    Raises:
        USGSApiError: If login fails
    """
    payload = {
        'username': username,
        'token': token
    }
    data = send_request('login-token', payload)
    return data  # Returns the API key

def logout(api_key):
    """Logout from the USGS M2M API."""
    payload = {}
    send_request('logout', payload, api_key)

def search(api_key, datasetName, spatialFilter, temporalFilter, maxResults=10, startingNumber=1, sortOrder='ASC'):
    """Search for scenes in the USGS catalog."""
    scene_filter = {'spatialFilter': spatialFilter}
    payload = {
        'datasetName': datasetName,
        'sceneFilter': scene_filter,
        'temporalFilter': temporalFilter,
        'maxResults': maxResults,
        'startingNumber': startingNumber,
        'sortOrder': sortOrder
    }
    data = send_request('scene-search', payload, api_key)
    return data

def download_options(api_key, datasetName, entityIds):
    """Get download options for the specified entities."""
    payload = {
        'datasetName': datasetName,
        'entityIds': entityIds
    }
    data = send_request('download-options', payload, api_key)
    return data

def download_request(api_key, payload):
    """Request downloads for the specified entities."""
    data = send_request('download-request', payload, api_key)
    return data

def download_retrieve(api_key):
    """
    Retrieve information about requested downloads.
    
    Args:
        api_key (str): Authentication token for the USGS M2M API.
        download_ids (list, optional): List of download IDs to retrieve info for.
            If None, retrieves info for all downloads.
            
    Returns:
        dict: Contains lists of 'available' and 'preparing' downloads.
        
    Raises:
        USGSApiError: If the API returns an error
    """
    payload = {}
    
    data = send_request("download-retrieve", payload, api_key)
    return data

def download_remove(api_key, download_id):
    """
    Remove a specific download from the download queue.
    
    Args:
        api_key (str): Authentication token for the USGS M2M API.
        download_id (int): Single download ID to remove.
            
    Returns:
        dict: Contains the result of the removal operation.
        
    Raises:
        USGSApiError: If the API returns an error
    """
    payload = {'downloadId': download_id}
    data = send_request("download-remove", payload, api_key)
    return data

def remove_multiple_downloads(api_key, download_ids):
    """
    Remove multiple downloads from the queue one by one.
    
    Args:
        api_key (str): Authentication token for the USGS M2M API.
        download_ids (list): List of download IDs to remove.
            
    Returns:
        dict: Contains the total number of downloads removed.
        
    Raises:
        USGSApiError: If the API returns an error for any download ID
    """
    total_removed = 0
    failed_ids = []
    
    for download_id in download_ids:
        try:
            result = download_remove(api_key, download_id)
            if result:
                total_removed += 1
        except USGSApiError:
            failed_ids.append(download_id)
    
    # If any failed, provide information but continue with what worked
    if failed_ids:
        print(f"Warning: Failed to remove {len(failed_ids)} download(s). IDs: {failed_ids}")
    
    return {'removed': total_removed, 'failed': failed_ids}

def remove_pending_downloads(api_key, label):
    """
    Remove pending downloads with the specified label.
    
    Args:
        api_key (str): Authentication token for the USGS M2M API.
        label (str): Label of downloads to remove.
            
    Returns:
        dict: Contains the result of the removal operation.
        
    Raises:
        USGSApiError: If the API returns an error
    """
    # Step 1: Retrieve pending downloads
    try:
        pending_downloads = get_all_downloads(api_key)
    except USGSApiError as e:
        print(f"Error retrieving pending downloads: {e}")
        return {'error': str(e)}

    if not pending_downloads:
        print("No pending downloads to remove.")
        return {}
    
    remove_payload = {"label": label}  # Adjust key to match the API's payload requirements

    # Step 3: Remove downloads
    result = send_request("download-order-remove", remove_payload, api_key)
    print("Removal Result:", result)
    return result

def get_all_downloads(api_key):
    """
    Get all available and preparing downloads.
    
    Args:
        api_key (str): Authentication token for the USGS M2M API.
        
    Returns:
        dict: Contains lists of 'available' and 'preparing' downloads.
        
    Raises:
        USGSApiError: If the API returns an error
    """
    return download_retrieve(api_key)

def purge_all_available_downloads(api_key):
    """
    Remove all available downloads from the queue.
    
    Args:
        api_key (str): Authentication token for the USGS M2M API.
        
    Returns:
        dict: Contains the number of downloads removed.
        
    Raises:
        USGSApiError: If the API returns an error
    """
    # Get all downloads
    try:
        downloads = get_all_downloads(api_key)
    except USGSApiError as e:
        print(f"Error retrieving downloads: {e}")
        return {'removed': 0, 'error': str(e)}
    
    # If there are available downloads, remove them
    if 'available' in downloads and downloads['available']:
        download_ids = [item['downloadId'] for item in downloads['available']]
        try:
            return remove_multiple_downloads(api_key, download_ids)
        except USGSApiError as e:
            print(f"Error removing downloads: {e}")
            return {'removed': 0, 'error': str(e)}
    
    return {'removed': 0}


