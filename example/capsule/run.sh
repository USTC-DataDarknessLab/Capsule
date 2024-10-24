#!/bin/bash

# First, we need to process the data to obtain it in binary format.
python capsule/src/datagen/products.py

# Then, we need to partition the raw data
python capsule/src/datapart/trans2subG.py --dataset=PD --force_partiton --partNUM=4

# In the ./config , we provide a configuration file for running the PD dataset. 
# We can use this configuration file to run the program
python capsule/src/train/capsule/capsule_dgl_train.py --json_path="capsule/config/PD_dgl.json"

# Note, if you want to run in pyg format, you need to set 'framework' to 'pyg' in the JSON file.
