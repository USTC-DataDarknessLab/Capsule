import argparse
from ogb.nodeproppred import PygNodePropPredDataset
import scipy
import numpy as np
import json
import torch
import os
import time


# Parse arguments
total_time = time.time()
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn_products')
args = argparser.parse_args()

ds = ['papers100M','ogbn_products','reddit','com_fr','uk-2006-05','wb2001']

# Download/load dataset
print('Loading dataset...')
root = '/path/to/Ginex'
os.makedirs(root, exist_ok=True)

raw_path = '/raid/bear/data/raw/' + args.dataset


featbin = raw_path + '/feat.bin'
graphbin = raw_path + '/graph.bin'
labelbin = raw_path + '/labels.bin'
train_idx_bin = raw_path + '/trainIds.bin'
val_idx_bin = raw_path+'/valIds.bin'
test_idx_bin = raw_path+'/testIds.bin'

train_idx = np.fromfile(train_idx_bin,dtype=np.int64)
val_idx = np.fromfile(val_idx_bin,dtype=np.int64)
test_idx = np.fromfile(test_idx_bin,dtype=np.int64)
edges = np.fromfile(graphbin,dtype=np.int32)
featlen = 0
classes = 0
num_nodes = 0

print(args.dataset)
ds_dic={
    'papers100M': 'PA',
    'ogbn_products': 'PD',
    'reddit': 'RD',
    'twitter': 'TW',
    'com_fr': 'FR',
    'uk-2006-05': 'UK',
    'wb2001': 'WB'
}

# like capsule datasetInfo to get dataset info
with open('/path/to/datasetInfo.json', 'r', encoding='utf-8') as file:
    dataset_info = json.load(file)

featlen = dataset_info[ds_dic[args.dataset]]["featLen"]
classes = dataset_info[ds_dic[args.dataset]]["classes"]
num_nodes = dataset_info[ds_dic[args.dataset]]["nodes"]



#dataset = PygNodePropPredDataset(args.dataset, root)


dataset_path = os.path.join(root, args.dataset + '-ginex')

# Construct sparse formats
#num_nodes = dataset[0].num_nodes
srcs = edges[::2]
dsts = edges[1::2]
coo = np.stack((srcs,dsts), axis=0)
#coo = dataset[0].edge_index.numpy()

v = np.ones_like(coo[0])
coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))

csc = coo.tocsc()
csr = coo.tocsr()


# Save csc-formatted dataset
indptr = csc.indptr.astype(np.int64)
indices = csc.indices.astype(np.int64)

features = torch.tensor(np.fromfile(featbin,dtype=np.float32).reshape(-1,featlen))
#features = dataset[0].x

labels = torch.tensor(np.fromfile(labelbin,dtype=np.int64)).reshape(-1,1)
#labels = dataset[0].y



os.makedirs(dataset_path, exist_ok=True)
indptr_path = os.path.join(dataset_path, 'indptr.dat')
indices_path = os.path.join(dataset_path, 'indices.dat')
features_path = os.path.join(dataset_path, 'features.dat')
labels_path = os.path.join(dataset_path, 'labels.dat')
conf_path = os.path.join(dataset_path, 'conf.json')
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

print('Saving indptr...')
indptr_mmap = np.memmap(indptr_path, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
indptr_mmap[:] = indptr[:]
indptr_mmap.flush()
print('Done!')

print('Saving indices...')
indices_mmap = np.memmap(indices_path, mode='w+', shape=indices.shape, dtype=indices.dtype)
indices_mmap[:] = indices[:]
indices_mmap.flush()
print('Done!')

print('Saving features...')
features_mmap = np.memmap(features_path, mode='w+', shape=features.shape, dtype=np.float32)
features_mmap[:] = features[:]
features_mmap.flush()
print('Done!')

print('Saving labels...')
labels = labels.type(torch.float32)

labels_mmap = np.memmap(labels_path, mode='w+', shape=labels.shape, dtype=np.int64)
labels_mmap[:] = labels[:]
labels_mmap.flush()
print('Done!')

print('Making conf file...')
mmap_config = dict()
mmap_config['num_nodes'] = num_nodes
mmap_config['indptr_shape'] = tuple(indptr.shape)
mmap_config['indptr_dtype'] = str(indptr.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['features_shape'] = tuple(features_mmap.shape)
mmap_config['features_dtype'] = str(features_mmap.dtype)
mmap_config['labels_shape'] = tuple(labels_mmap.shape)
mmap_config['labels_dtype'] = str(labels_mmap.dtype)
mmap_config['num_classes'] = classes
json.dump(mmap_config, open(conf_path, 'w'))
print('Done!')


print('Saving split index...')
idx = dict()
idx['train'] = torch.tensor(train_idx)
idx['test'] = torch.tensor(val_idx)
idx['valid'] =torch.tensor(test_idx)


torch.save(idx, split_idx_path)

# Calculate and save score for neighbor cache construction
print('Calculating score for neighbor cache construction...')
score_path = os.path.join(dataset_path, 'nc_score.pth')
csc_indptr_tensor = torch.from_numpy(csc.indptr.astype(np.int64))
csr_indptr_tensor = torch.from_numpy(csr.indptr.astype(np.int64))

eps = 0.00000001
in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1]) + eps
out_num_neighbors = (csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]) + eps


score = out_num_neighbors / in_num_neighbors
print('Saving score...')
torch.save(score, score_path)
print('Done!')

print("total time used {:.4f}s".format(time.time() - total_time))
