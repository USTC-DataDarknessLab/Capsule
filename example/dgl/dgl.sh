#!/bin/bash

# For DGL, relevant parameters can be passed through args. 
# Specific parameters can be found by checking the source files. 
# Generally, we would set the dataset, fanout, and the training model.

python ../src/train/dgl/dgl_train.py --dataset ogb-products