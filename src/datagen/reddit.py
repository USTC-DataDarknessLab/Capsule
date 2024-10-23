import torch
import dgl
import numpy as np
import random
import pickle
from dgl.data import RedditDataset
import os

RAW_DATA_DIR = "../../data/dataset/"
def load_reddit(self_loop=True):
    data = RedditDataset(self_loop=self_loop,raw_dir=RAW_DATA_DIR)
    g = data[0]
    g.ndata['feat'] = g.ndata.pop('feat')
    g.ndata['label'] = g.ndata.pop('label')
    train_idx = []
    val_idx = []
    test_idx = []
    for index in range(len(g.ndata['train_mask'])):
        if g.ndata['train_mask'][index] == 1:
            train_idx.append(index)
    for index in range(len(g.ndata['val_mask'])):
        if g.ndata['val_mask'][index] == 1:
            val_idx.append(index)
    for index in range(len(g.ndata['test_mask'])):
        if g.ndata['test_mask'][index] == 1:
            test_idx.append(index)
    return g, data,train_idx,val_idx,test_idx

OUTPUT_DATA_DIR = "../../data/raw/products"
if not os.path.exists(OUTPUT_DATA_DIR):
    os.makedirs(OUTPUT_DATA_DIR)

g, dataset,train_idx,val_idx,test_idx= load_reddit()

src = g.edges()[0]
dst = g.edges()[1]
src.numpy().tofile(OUTPUT_DATA_DIR + "/srcList.bin")
dst.numpy().tofile(OUTPUT_DATA_DIR + "/dstList.bin")
torch.stack((src,dst),dim=1).to(torch.int32).reshape(-1).numpy().tofile(OUTPUT_DATA_DIR + "/graph.bin")
feat = g.ndata['feat'].numpy().tofile(OUTPUT_DATA_DIR + "/feat.bin")
label = g.ndata['label'].numpy().tofile(OUTPUT_DATA_DIR + "/labels.bin")

torch.Tensor(train_idx).to(torch.int64).numpy().tofile(OUTPUT_DATA_DIR + "/trainIds.bin")
torch.Tensor(val_idx).to(torch.int64).numpy().tofile(OUTPUT_DATA_DIR + "/valIds.bin")
torch.Tensor(test_idx).to(torch.int64).numpy().tofile(OUTPUT_DATA_DIR + "/testIds.bin")
