#!/bin/bash

# For PyG, relevant parameters can be passed through args. 
# Specific parameters can be found by checking the source files. 
# Generally, we would set the dataset, fanout, and the training model.

python ../src/train/pyg/pyg_train.py --dataset ogb-products