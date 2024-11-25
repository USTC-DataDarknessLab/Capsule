import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
#from torch.utils.data import Dataset, DataLoader
import ast
import random
import copy
import tqdm
import argparse
import sklearn.metrics
import numpy as np
import time
import sys
import os
import json
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler
from capsule_model import DGL_SAGE, DGL_GCN, DGL_GAT

import os
# Reduced OOM OOM caused by GPU cache, uncertain impact on system performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

curDir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curDir+"/../../"+"load")
from loader import CustomDataset

def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )

def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )

def train(dataset, model,basicLoop=0,loop=10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=dataset.batchsize, collate_fn=collate_fn)#,pin_memory=True)
    for epoch in range(loop):
        startTime = time.time()
        total_loss = 0
        model.train()
        for it,(graph,feat,label,number) in enumerate(train_loader):
            feat = feat.cuda()
            y_hat = model(graph, feat)
            label = label.to(torch.int64)
            loss = F.cross_entropy(y_hat[:number], label[:number].to('cuda:0'))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print("| Epoch {:03d} | Loss {:.4f} | Time {:.3f}s |".format(basicLoop+epoch, total_loss / (it+1), time.time()-startTime))

def collate_fn(data):
    return data[0]

def load_reddit(self_loop=True):
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=self_loop,raw_dir="capsule/data/dataset")
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

def Testing(model,name,data,device="cuda:0",tid=None,num_classes=0):
    if name == "PD":
        g = data[0]
        acc = layerwise_infer(device, g, data.test_idx, model, num_classes, batch_size=4096)
    elif name == "RD":
        acc = layerwise_infer(device, data, tid, model, num_classes, batch_size=4096)  
    print("Test Accuracy {:.4f}".format(acc.item()))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument('--json_path', type=str, default='../../../config/PD_dgl.json', help='Dataset name')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    data = None
    with open(args.json_path, 'r') as json_file:
        data = json.load(json_file)
    
    arg_fanout = data["fanout"]
    arg_layers = len(arg_fanout)

    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda:0')
    if data["model"] == "SAGE":
        model = DGL_SAGE(data['featlen'], 256, data['classes'],arg_layers).to('cuda:0')
    elif data["model"] == "GCN":
        model = DGL_GCN(data['featlen'], 256, data['classes'] ,arg_layers,F.relu,0.5).to('cuda:0')
    elif data["model"] == "GAT":
        model = DGL_GAT(data['featlen'], 256, data['classes'], heads=[4,1]).to('cuda:0')
    else:
        print("Invalid model option. Please choose from 'SAGE', 'GCN', or 'GAT'.")
        sys.exit(1)
    
    print('Training...')
    dataset = CustomDataset(args.json_path)  # Use args.json_path as the JSON file path
    epochInterval = data["epochInterval"]
    maxEpoch = dataset.maxEpoch
    epoch = 0
    
    for BLoop in range(0,maxEpoch,epochInterval):
        train(dataset, model,basicLoop=BLoop,loop=epochInterval)
    # torch.save(model.state_dict(), 'model_parameters.pth')
    # model.load_state_dict(torch.load('model_parameters.pth'))
    if data["dataset"] == "PD":
        TestDataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root="capsule/data/dataset"))
        Testing(model,data["dataset"],TestDataset,num_classes=data['classes'])
    elif data["dataset"] == "RD":
        g, Testdata,train_idx,val_idx,test_idx = load_reddit()
        Testing(model,data["dataset"],g,device="cuda:0",tid=test_idx,num_classes=data['classes'])
    elif data["dataset"] == "PA":
        if arg_layers == 2:
            sampler_test = NeighborSampler([100,100])
        elif arg_layers == 3:
            sampler_test = NeighborSampler([20,50,50])
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-papers100M',root="capsule/data/dataset"))
        g = dataset[0]
        test_dataloader = dgl.dataloading.DataLoader(g, dataset.test_idx, sampler_test, device=device,
                                batch_size=4096, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=True)
        acc = evaluate(model, g, test_dataloader,num_classes=data['classes'])
        print("Test Accuracy {:.4f}".format(acc.item()))
        
