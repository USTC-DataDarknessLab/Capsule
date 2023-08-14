import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.nn.pytorch import GraphConv
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
import sklearn.metrics
import numpy as np
import time
class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.hid_size = n_hidden
        self.out_size = n_classes
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        for _ in range(1, n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        self.layers.append(
            GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y
    
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

def train(args, device, g, dataset, model,data=None):
    # create sampler & dataloader
    if data != None:
        train_idx,val_idx,test_idx = data 
    else:
        train_idx = dataset.train_idx.to(device)
        val_idx = dataset.val_idx.to(device)
        # train_idx = dataset.train_idx
        # val_idx = dataset.val_idx
    sampler = NeighborSampler([25,10],  # fanout for [layer-0, layer-1, layer-2]
                            prefetch_node_feats=['feat'],
                            prefetch_labels=['label'])
    use_uva = False
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=256, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=256, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)
    # train_dataloader = DataLoader(g, train_idx, sampler, device=device,
    #                             batch_size=1024, shuffle=True,
    #                             drop_last=False, num_workers=0)

    # val_dataloader = DataLoader(g, val_idx, sampler, device=device,
    #                             batch_size=1024, shuffle=True,
    #                             drop_last=False, num_workers=0)
    # print("first")
    # for index , info in enumerate(train_dataloader):
    #     print(info)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    print("second")
    for epoch in range(50):
        start = time.time()
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        acc = evaluate(model, g, val_dataloader)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))
        print("time :",time.time()-start)

def load_reddit(self_loop=True):
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=self_loop)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='puregpu', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]
    #g, dataset,train_idx,val_idx,test_idx= load_reddit()
    #data = (train_idx,val_idx,test_idx)
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    model = GCN(in_size, 256, out_size,2,F.relu,0.5).to(device)

    # model training
    print('Training...')
    train(args, device, g, dataset, model)
    #train(args, device, g, dataset, model,data=data)

    # test the model
    print('Testing...')
    #acc = layerwise_infer(device, g, test_idx, model, batch_size=4096)
    acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
