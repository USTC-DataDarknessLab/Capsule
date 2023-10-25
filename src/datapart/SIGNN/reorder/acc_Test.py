import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
import sklearn.metrics
import numpy as np
partNUM = 4

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        #self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
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

def train(args, device, g, train_idx, model):
    sampler = NeighborSampler([10, 25],  # fanout for [layer-0, layer-1, layer-2]
                              prefetch_node_feats=['feat'],
                              prefetch_labels=['label'])
    use_uva = (args.mode == 'mixed')
    train_dataloader_list = []
    for i in range(partNUM):
        train_dataloader_list.append(
            DataLoader(g[i], train_idx[i], sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)
        )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    for epoch in range(20):
        model.train()
        total_loss = 0
        for i in range(partNUM):
            train_dataloader = train_dataloader_list[i]
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label']
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                #accuracy = sklearn.metrics.accuracy_score(y.cpu().numpy(), y_hat.argmax(1).detach().cpu().numpy())
            acc = torch.Tensor([0.00])
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    graph_dir = '/home/bear/workspace/single-gnn/data/products/raw-products_4/'#'data_4/'
    part_config = graph_dir + 'ogb-product.json'
    print('loading partitions')
    
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    model = SAGE(100, 256, 47).to(device)
    g_list = []
    train_list = []
    PATH = "/home/bear/workspace/single-gnn/data/partition/PD/processed/part"
    for i in range(partNUM):
        indices = np.fromfile(PATH + f"{i}/indices.bin",dtype=np.int32)
        indptr = np.fromfile(PATH + f"{i}/indptr.bin",dtype=np.int32)
        indptr = torch.tensor(indptr).to(torch.int64)
        indices = torch.tensor(indices).to(torch.int64)
        feat = np.fromfile(PATH + f"{i}/feat.bin",dtype=np.float32).reshape(-1,100)
        trainIds = np.fromfile(PATH + f"{i}/trainIds.bin",dtype=np.int32)
        trainIds = torch.tensor(trainIds).to(torch.int64)
        labels = np.fromfile(PATH + f"{i}/labels.bin",dtype=np.int64)
        device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
        g = dgl.graph(('csr', (indptr, indices, [])))
        
        # g.nodes['_N'] = g.nodes['_N'].to(torch.int64)
        
        feat = torch.tensor(feat)
        g.ndata['feat'] = feat
        # print(g.nodes('_N'))
        labels = torch.tensor(labels)
        g.ndata['label'] = labels
        g_list.append(g)
        train_list.append(trainIds)
    print('Training...')
    train(args, device, g_list, train_list , model)

    
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root="/home/bear/workspace/single-gnn/data/dataset"))
    g = dataset[0]
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    # test the model
    print('Testing...')
    acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))
