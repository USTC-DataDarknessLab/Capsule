import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import sys
import os
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../../"+"load")
from loader import CustomDataset

class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
 
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            print("x:",x.dtype)
            print("edge_index:",edge_index.dtype)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                pbar.update(1)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all


def run(rank, world_size, dataset):
    train_loader = DataLoader(dataset=dataset, batch_size=1024, collate_fn=collate_fn,pin_memory=True)
    torch.manual_seed(12345)
    model = SAGE(100, 256, 47).to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    count = 0
    for epoch in range(1, 2):
        model.train()    
        for graph,feat,label,number in train_loader:        
            count += 1
            if count > 5:
                break
            optimizer.zero_grad()     
            out = model(feat.to('cuda:0'), graph.to('cuda:0'))[1:number+1]
            loss = F.cross_entropy(out, label[:number].to(torch.int64).to('cuda:0'))
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]

if __name__ == '__main__':
    # dataset = Reddit('.')
    # world_size = torch.cuda.device_count()
    world_size = 1
    print('Let\'s use', world_size, 'GPUs!')
    dataset = CustomDataset("./../../load/graphsage.json")
    run(0, world_size, dataset)
    
    #mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)