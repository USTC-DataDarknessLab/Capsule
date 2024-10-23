"""
Partial code reference: https://github.com/SJTU-IPADS/gnnlab
"""

import torch
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import random
import pickle
import dgl
import os


RAW_DATA_DIR = "../../data/dataset/"
OUTPUT_DATA_DIR = "../../data/raw/products"
if not os.path.exists(OUTPUT_DATA_DIR):
    os.makedirs(OUTPUT_DATA_DIR)

def convert_data_tobin():
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root=RAW_DATA_DIR))
    g = dataset[0]
    src = g.edges()[0]
    dst = g.edges()[1]
    src.numpy().tofile(OUTPUT_DATA_DIR + "/srcList.bin")
    dst.numpy().tofile(OUTPUT_DATA_DIR + "/dstList.bin")
    torch.stack((src,dst),dim=1).to(torch.int32).reshape(-1).numpy().tofile(OUTPUT_DATA_DIR + "/graph.bin")
    g.ndata['feat'].numpy().tofile(OUTPUT_DATA_DIR + "/feat.bin")
    g.ndata['label'].numpy().tofile(OUTPUT_DATA_DIR + "/labels.bin")
    torch.nonzero(g.ndata['train_mask']).squeeze().numpy().tofile(OUTPUT_DATA_DIR + "/trainIds.bin")
    torch.nonzero(g.ndata['val_mask']).squeeze().numpy().tofile(OUTPUT_DATA_DIR + "/valIds.bin")
    torch.nonzero(g.ndata['test_mask']).squeeze().numpy().tofile(OUTPUT_DATA_DIR + "/testIds.bin")

if __name__ == '__main__':
    convert_data_tobin()