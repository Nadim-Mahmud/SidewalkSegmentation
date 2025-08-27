import os
import sys
from PIL import Image

'''
This is a script that is meant to convert all tif files in a directory to
.jpg. 
'''

# Constant config list.
CONFIG_DIRS = [
    # Example directories; update these paths according to your needs
    '/home/hanew/your_project_folder/trace/data/sample_usgs_images',
    'data/full_50/oxford_jpgs'
]
def convert_tif_to_jpg(input_dir, output_dir):
    # Check if the provided path is a valid directory
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid directory.")
        return

    # Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        # Process files ending with .tif or .tiff (case-insensitive)
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            try:
                # Open the image file
                with Image.open(input_path) as im:
                    # Ensure the image is in RGB mode (JPEG cannot handle alpha channels)
                    rgb_im = im.convert('RGB')
                    # Construct the new filename by appending "_sat.jpg" to the base name
                    base_name = os.path.splitext(filename)[0]
                    output_filename = f"{base_name}_sat.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    # Save the image in JPEG format
                    rgb_im.save(output_path, 'JPEG')
                    print(f"Converted '{filename}' to '{output_filename}'")
            except Exception as e:
                print(f"Failed to convert '{filename}': {e}")

if __name__ == '__main__':
    directories = CONFIG_DIRS
    input_dir = directories[0]
    output_dir = directories[1]
    convert_tif_to_jpg(input_dir, output_dir)
