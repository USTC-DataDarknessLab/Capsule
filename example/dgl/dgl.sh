#!/bin/bash

# For DGL, relevant parameters can be passed through args. 
# Specific parameters can be found by checking the source files. 
# Generally, we would set the dataset, fanout, and the training model.

# PD
python ../src/train/dgl/dgl_train.py --dataset ogb-products
# RD
python ../src/train/dgl/dgl_train.py --dataset Reddit
# PA
python ../src/train/dgl/dgl_train.py --dataset ogb-papers100M
# WB
python ../src/train/dgl/dgl_train.py --dataset wb2001
# UK
python ../src/train/dgl/dgl_train.py --dataset ogb-products
# FR
python ../src/train/dgl/dgl_train.py --dataset uk-2006-05