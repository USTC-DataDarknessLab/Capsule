import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from dgl_model import SAGE, GCN, GAT
import tqdm
import argparse
import ast
import sklearn.metrics
import numpy as np
import time
import sys


def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'].cpu().numpy())
            y_hats.append(model(blocks, x).argmax(1).cpu().numpy())
        predictions = np.concatenate(y_hats)
        labels = np.concatenate(ys)
    return sklearn.metrics.accuracy_score(labels, predictions)

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
    return sklearn.metrics.accuracy_score(label.cpu().numpy(), pred.argmax(1).cpu().numpy())

def train(args, device, g, dataset, model,data=None ,basicLoop=0,loop=10):
    # create sampler & dataloader
    if data != None:
        train_idx,val_idx,test_idx = data 
    else:
        train_idx = dataset.train_idx.to(device)
        val_idx = dataset.val_idx.to(device)
        test_idx = dataset.test_idx.to(device)
    # sampler = NeighborSampler(args.fanout,  # fanout for [layer-0, layer-1, layer-2]
    #                         prefetch_node_feats=['feat'],
    #                         prefetch_labels=['label'])
    sampler = NeighborSampler(args.fanout)
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=args.bs, shuffle=False,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)
    if val_idx != []:
        val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                    batch_size=args.bs, shuffle=True,
                                    drop_last=False, num_workers=0,
                                    use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    for epoch in range(loop):
        model.train()
        total_loss = 0
        startTime = time.time()
        count = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            y = y.to(torch.int64)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            count = it
        trainTime = time.time()-startTime
        if val_idx != []:
            acc = evaluate(model, g, val_dataloader)
        else:
            acc = torch.Tensor([0.00])
        print("| Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time {:.3f}s | Count {} |"
              .format(basicLoop+epoch, total_loss / (it+1), acc.item(), trainTime, count))
    
def load_reddit(self_loop=True):
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=self_loop,raw_dir='capsule/data/dataset/')
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

def load_dataset(dataset,path,featlen,mode=None):
    graphbin = "%s/%s/graph.bin" % (path,dataset)
    labelbin = "%s/%s/labels.bin" % (path,dataset)
    featsbin = "%s/%s/feat.bin" % (path,dataset)
    trainbin = "%s/%s/trainIds.bin" % (path,dataset)
    edges = np.fromfile(graphbin,dtype=np.int32)
    
    srcs = edges[::2]
    dsts = edges[1::2]
    feats = np.fromfile(featsbin,dtype=np.float32).reshape(-1,100)
    

    label = np.fromfile(labelbin,dtype=np.int64)

    g = dgl.graph((srcs,dsts))
    feats_tmp = feats[:g.num_nodes()]
    g.ndata['feat'] = torch.tensor(feats_tmp)
    g.ndata['label'] = torch.tensor(label[:g.num_nodes()])

    train_idx = np.fromfile(trainbin,dtype=np.int64)
    return g,train_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument('--fanout', type=ast.literal_eval, default=[10, 10, 10], help='Fanout value')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dataset', type=str, default='ogb-papers100M', help='Dataset name')
    parser.add_argument('--maxloop', type=int, default=20, help='max loop number')
    parser.add_argument('--model', type=str, default="SAGE", help='train model')
    parser.add_argument('--bs', type=int, default=1024, help='batchsize')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    default_datasetpath = "capsule/data/raw"
    print('Loading data')
    out_size = 0
    if args.dataset == 'ogb-products':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root="capsule/data/dataset"))
        g = dataset[0]
        data = None
    elif args.dataset == 'Reddit':
        g, dataset,train_idx,val_idx,test_idx= load_reddit()
        data = (train_idx,val_idx,test_idx)
    elif args.dataset == 'ogb-papers100M':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-papers100M',root="capsule/data/dataset"))
        g = dataset[0]
        data = None
    elif args.dataset == 'com_fr':
        g,train_idx = load_dataset(args.dataset,default_datasetpath,100,'id_ordered')
        out_size = 150
        data = (train_idx,[],[])
        dataset = None
    elif args.dataset == 'wb2001':
        g,train_idx = load_dataset(args.dataset,default_datasetpath,100,'id_ordered')
        out_size = 150
        data = (train_idx,[],[])
        dataset = None
    elif args.dataset == 'uk-2006-05':
        g,train_idx = load_dataset(args.dataset,default_datasetpath,100)
        train_idx = torch.tensor(train_idx).to(torch.int64)
        out_size = 150
        data = (train_idx,[],[])
        dataset = None
    else:
        exit(0)
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    
    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes if out_size == 0 else out_size
    if args.model == "SAGE":
        model = SAGE(in_size, 256, out_size,args.layers).to(device)
    elif args.model == "GCN":
        model = GCN(in_size, 256, out_size,args.layers,F.relu,0.5).to(device)
    elif args.model == "GAT":
        model = GAT(in_size, 256, out_size,heads=[4,1]).to(device)
    else:
        print("Invalid model option. Please choose from 'SAGE', 'GCN', or 'GAT'.")
        sys.exit(1)  # Use sys.exit(1) to exit the program and return an error status code
    # model.load_state_dict(torch.load("model.pth"))

    # model training
    print('Training...')
    loopList = [0,5,10,20,30,50,100,150,200]
    for index in range(1,len(loopList)):
        if loopList[index] > args.maxloop:
            break
        _loop = loopList[index] - loopList[index - 1]
        train(args, device, g, dataset, model,data=data,basicLoop=loopList[index - 1],loop=_loop)
        #print('Testing with after loop {}:...'.format(loopList[index]))
        #model.load_state_dict(torch.load('model_parameters.pth'))
        if args.dataset == 'ogb-products':
            acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
        elif args.dataset == 'Reddit':
            acc = layerwise_infer(device, g, test_idx, model, batch_size=4096) 
        elif args.dataset == 'ogb-papers100M':
            acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096) 
        if args.dataset in ['ogb-products','Reddit','ogb-papers100M']:
            print("Test Accuracy {:.4f}".format(acc.item()))
            print("-"*20)