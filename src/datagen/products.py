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

DATA_PATH = 'capsule/sgnn'

DOWNLOAD_URL = 'http://snap.stanford.edu/ogb/data/nodeproppred/products.zip'
RAW_DATA_DIR = DATA_PATH +'/raw_dataset'
PAPERS_RAW_DATA_DIR = f'{RAW_DATA_DIR}/products'
OUTPUT_DATA_DIR = DATA_PATH + '/dataset/products'


def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/products.zip'):
        assert(os.system(
            f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/products.zip') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{PAPERS_RAW_DATA_DIR}/unzipped'):
        assert(os.system(
            f'cd {RAW_DATA_DIR}; unzip {RAW_DATA_DIR}/products.zip') == 0)
        assert(os.system(f'touch {PAPERS_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')

def write_meta():
    print('Writing meta file...')
    with open(f'{OUTPUT_DATA_DIR}/meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE', 2449029))
        f.write('{}\t{}\n'.format('NUM_EDGE', 123718280))
        f.write('{}\t{}\n'.format('FEAT_DIM', 100))
        f.write('{}\t{}\n'.format('NUM_CLASS', 47))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', 196615))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', 39323))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', 2213091))

def convert_data_tobin():
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root=RAW_DATA_DIR))
    g = dataset[0]
    src = g.edges()[0].numpy()
    dst = g.edges()[1].numpy()
    src.tofile(OUTPUT_DATA_DIR + "/srcList.bin")
    dst.tofile(OUTPUT_DATA_DIR + "/dstList.bin")
    graph = torch.stack((src,dst),dim=1).reshape(-1).numpy().tofile(OUTPUT_DATA_DIR + "/graph.bin")
    g.ndata['feat'].numpy().tofile(OUTPUT_DATA_DIR + "/feat.bin")
    g.ndata['label'].numpy().tofile(OUTPUT_DATA_DIR + "/labels.bin")
    torch.nonzero(g.ndata['train_mask']).squeeze().numpy().tofile(OUTPUT_DATA_DIR + "/trainIds.bin")
    torch.nonzero(g.ndata['val_mask']).squeeze().numpy().tofile(OUTPUT_DATA_DIR + "/valIds.bin")
    torch.nonzero(g.ndata['test_mask']).squeeze().numpy().tofile(OUTPUT_DATA_DIR + "/testIds.bin")

if __name__ == '__main__':
    convert_data_tobin()