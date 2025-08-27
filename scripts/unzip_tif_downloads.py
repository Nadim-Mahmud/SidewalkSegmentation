import os
import zipfile
from pathlib import Path

def unzip_tifs_only(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Walk through all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".zip"):
                zip_path = Path(root) / file
                print(f"Scanning: {zip_path}")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            if member.lower().endswith((".tif", ".tiff")):
                                target_path = output_dir / Path(member).name
                                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                    target.write(source.read())
                                print(f"Extracted: {target_path}")
                except zipfile.BadZipFile:
                    print(f"Skipping invalid zip file: {zip_path}")

    print(f"All valid .tif files have been extracted to: {output_dir}")

unzip_tifs_only("/home/hanew/your_project_folder/omniacc/data/usgs_tifs_zips", "/home/hanew/your_project_folder/omniacc/data/tifs/usgs_manhattan")
