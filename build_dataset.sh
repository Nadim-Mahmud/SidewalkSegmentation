#!/bin/bash 

python build_dataset.py \
    --node_dir  /home/hanew/your_project_folder/omniacc/data/tifs/manhattan/intersection_nodes/ids \
    --tif_dir /home/hanew/your_project_folder/omniacc/data/tifs/manhattan/images \
    --output_dir /home/hanew/your_project_folder/omniacc/data/manhat_data_sat_intersections \
    --zoom 18 \
    --workers 8
