'''
A script for creating the directory structure for satellite image patches and their corresponding mask images.
The resulting folder will contain subfolders named after the original folders, 
with satellite images renamed to `*_sat.jpg` and mask images renamed to `*_mask.png`.
'''

import os
from PIL import Image
import shutil

def reorganize_and_rename(sat_dir, mask_dir, output_dir):
    """
    Reorganize satellite patches and mask images into a new directory structure.
    
    Args:
        sat_dir (str): Path to the directory containing satellite image folders.
        mask_dir (str): Path to the directory containing mask image folders.
        output_dir (str): Path where the new organized directory will be created.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each folder in the satellite directory
    for folder_name in os.listdir(sat_dir):
        sat_folder_path = os.path.join(sat_dir, folder_name)
        mask_folder_path = os.path.join(mask_dir, folder_name)
        
        # Check if it's a directory and corresponding mask folder exists
        if os.path.isdir(sat_folder_path) and os.path.isdir(mask_folder_path):
            # Create new folder in output_dir
            new_folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Process satellite image
            sat_img_path = os.path.join(sat_folder_path, "satellite.png")
            if os.path.isfile(sat_img_path):
                img = Image.open(sat_img_path)
                jpg_path = os.path.join(new_folder_path, f"{folder_name}_sat.jpg")
                img.convert("RGB").save(jpg_path, "JPEG")
            else:
                print(f"Warning: {sat_img_path} not found.")
            
            # Process mask image
            mask_img_path = os.path.join(mask_folder_path, "satellite.png")
            if os.path.isfile(mask_img_path):
                new_mask_path = os.path.join(new_folder_path, f"{folder_name}_mask.png")
                shutil.copy(mask_img_path, new_mask_path)
            else:
                print(f"Warning: {mask_img_path} not found.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reorganize satellite and mask images.")
    parser.add_argument("--sat_dir", default='data/manhat_data_sat_no_road', help="Directory with satellite image folders")
    parser.add_argument("--mask_dir", default='data/manhat_data_masks_no_road', help="Directory with mask image folders")
    parser.add_argument("--output_dir", default='data/data_patches_manhat', help="Directory where organized folder structure will be created")
    
    args = parser.parse_args()
    reorganize_and_rename(args.sat_dir, args.mask_dir, args.output_dir)
    
    print("Reorganization complete!")
