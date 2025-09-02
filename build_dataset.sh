#!/bin/bash 

python build_dataset.py \
    --node_dir  /data/tifs/manhattan/intersection_nodes/ids \
    --tif_dir /data/tifs/manhattan/images \
    --output_dir /data/manhat_data_sat_intersections \
    --zoom 18 \
    --workers 8