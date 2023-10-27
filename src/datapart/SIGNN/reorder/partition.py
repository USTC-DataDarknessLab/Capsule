from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import numpy as np 
import torch
import dgl
import time 
import copy
import os



### config 
datasets = {
    "FR": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/com_fr",
        "maxID": 65608366
    },
    "PA": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/papers100M",
        "maxID": 111059956
    },
    "PD": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/ogbn_products",
        "maxID": 2449029
    },
    "TW": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/twitter",
        "maxID": 41652230
    },
    "UK": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/uk-2006-05",
        "maxID": 77741046
    }
}


def acc_ana(tensor):
    num_ones = torch.sum(tensor == 1).item()  
    total_elements = tensor.numel()  
    percentage_ones = (num_ones / total_elements) * 100 
    print(f"only use by one train node : {percentage_ones:.2f}%")
    num_greater_than_1 = torch.sum(tensor > 1).item() 
    percentage_greater_than_1 = (num_greater_than_1 / total_elements) * 100
    print(f"use by multi train nodes : {percentage_greater_than_1:.2f}%")
    # edgeNUM = edgeTable.cpu().sum() - edgeNUM
    # print(f"edge add to subG : {edgeNUM} , {edgeNUM * 1.0 / allEdgeNUM * 100 :.2f}% of total edges")
    # print(f"after {index} BFS has {torch.nonzero(nodeTable).size(0)} nodes, "
    # f"{torch.nonzero(nodeTable).size(0) * 1.0 / maxID * 100 :.2f}% of total nodes")


def checkFilePath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"file '{path}' exist...")

def saveBin(tensor,savePath,addSave=False):
    if addSave :
        with open(savePath, 'ab') as f:
            if isinstance(tensor, torch.Tensor):
                tensor.numpy().tofile(f)
            elif isinstance(tensor, np.ndarray):
                tensor.tofile(f)
    else:
        if isinstance(tensor, torch.Tensor):
            tensor.numpy().tofile(savePath)
        elif isinstance(tensor, np.ndarray):
            tensor.tofile(savePath)

RUNTIME = 0
SAVETIME = 0
## bfs 遍历获取基础子图
def analysisG(graph,maxID,partID,trainId=None,savePath=None):
    global RUNTIME
    global SAVETIME
    dst = torch.tensor(graph[::2])
    src = torch.tensor(graph[1::2])
    if trainId == None:
        trainId = torch.arange(int(maxID*0.01),dtype=torch.int64)
    nodeTable = torch.zeros(maxID,dtype=torch.int32)
    nodeTable[trainId] = 1

    batch_size = 2
    src_batches = torch.chunk(src, batch_size, dim=0)
    dst_batches = torch.chunk(dst, batch_size, dim=0)
    batch = [src_batches, dst_batches]

    repeats = 3
    acc = True
    start = time.time()
    edgeTable = torch.zeros_like(src,dtype=torch.int32).cuda()
    edgeNUM = 0
    allEdgeNUM = src.numel()
    for index in range(1,repeats+1):
        acc_tabel = torch.zeros_like(nodeTable,dtype=torch.int32)
        # print(f"before {index} BFS has {torch.nonzero(nodeTable).size(0)} nodes, "
        #     f"{torch.nonzero(nodeTable).size(0) * 1.0 / maxID * 100 :.2f}% of total nodes")
        offset = 0
        for src_batch,dst_batch in zip(*batch):
            tmp_nodeTabel = copy.deepcopy(nodeTable)
            tmp_nodeTabel = tmp_nodeTabel.cuda()
            src_batch = src_batch.cuda()
            dst_batch = dst_batch.cuda()
            dgl.fastFindNeigEdge(tmp_nodeTabel,edgeTable,src_batch, dst_batch, offset)
            offset += len(src_batch)
            tmp_nodeTabel = tmp_nodeTabel.cpu()
            acc_tabel = acc_tabel | tmp_nodeTabel
        print("end bfs...")
        #acc_ana(acc_tabel)

        nodeTable = acc_tabel
        
        print('-'*10)
    edgeTable = edgeTable.cpu()
    ## merge edge
    graph = graph.reshape(-1,2)
    # print("edgeTable:",edgeTable)
    processPath = ""
    nodeSet =  torch.nonzero(nodeTable).reshape(-1).to(torch.int32)
    edgeTable = torch.nonzero(edgeTable).reshape(-1).to(torch.int32)
    selfLoop = np.repeat(trainId.to(torch.int32), 2)
    subGEdge = graph[edgeTable]
    RUNTIME += time.time()-start

    saveTime = time.time()
    checkFilePath(savePath + processPath)
    DataPath = savePath + processPath + f"/raw_G.bin"
    TrainPath = savePath + processPath + f"/raw_trainIds.bin"
    NodePath = savePath + processPath + f"/raw_nodes.bin"
    saveBin(nodeSet,NodePath)
    saveBin(selfLoop,DataPath)
    saveBin(subGEdge,DataPath,addSave=True)
    saveBin(trainId,TrainPath)
    SAVETIME += time.time()-saveTime
    return RUNTIME,SAVETIME

if __name__ == '__main__':
    selected_dataset = ["PA"] 
    for NAME in selected_dataset:
        dataset = datasets[NAME]
        GRAPHPATH = dataset["GRAPHPATH"]
        maxID = dataset["maxID"]
    #trainId = torch.arange(int(maxID*0.01),dtype=torch.int64)
    #trainId = torch.arange(196615,dtype=torch.int64)
    trainId = np.fromfile("/home/bear/workspace/single-gnn/data/raid/papers100M/trainIDs.bin",dtype=np.int64)
    trainId = torch.tensor(trainId)
    batch_size = 8
    trainBatch = torch.chunk(trainId, batch_size, dim=0)
    graph = np.fromfile(GRAPHPATH+"/graph.bin",dtype=np.int32)
    subGSavePath = "/home/bear/workspace/single-gnn/data/partition/PA"
    for index,trainids in enumerate(trainBatch):
        analysisG(graph,maxID,index,trainId=trainids,savePath=subGSavePath+f"/part{index}")
    
    print(f"run time cost:{RUNTIME:.3f}")
    print(f"save time cost:{SAVETIME:.3f}")