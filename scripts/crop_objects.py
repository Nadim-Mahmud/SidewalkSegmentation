'''
Ethan Han

A script for cropping out the objects from satellite images
given a respective mask png.
'''

from PIL import Image
import numpy as np

CONFIG = {'sat_im_path': 'data_patches/196560160/196560160_sat.jpg',
        'sat_mask_path': 'data_patches/196560160/196560160_mask.png', 
        'output_dir': '.'}

def crop_from_mask(sat_im_path, sat_mask_path, output_dir):
    # Load your satellite image and mask
    sat_image = Image.open(sat_im_path).convert("RGB")
    mask_image = Image.open(sat_mask_path).convert("L")  # Convert to grayscale

    # Convert images to NumPy arrays
    sat_array = np.array(sat_image)
    mask_array = np.array(mask_image)

    # Apply mask: keep only pixels where the mask is white (255), else black them out
    masked_array = np.where(mask_array[:, :, None] == 255, sat_array, 0)

    # Convert masked array back to an image
    masked_image = Image.fromarray(masked_array.astype(np.uint8))
    sat_image.putalpha(mask_image)


    # Save or show the result
    sat_image.save(f"{output_dir}/masked_output.png")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop from satellite image based on mask images.")
    parser.add_argument("--sat_dir", default=CONFIG['sat_im_path'], help="Path to the satellite image patch.")
    parser.add_argument("--mask_dir", default=CONFIG['sat_mask_path'], help="Path to the mask image.")
    parser.add_argument("--output_dir", default=CONFIG['output_dir'], help="Directory where cropped image will be saved.")
    
    args = parser.parse_args()
    crop_from_mask(args.sat_dir, args.mask_dir, args.output_dir)
