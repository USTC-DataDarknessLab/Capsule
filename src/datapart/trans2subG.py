import numpy as np
import dgl
import torch
import os
import copy
import gc
import time
from scipy.sparse import csr_matrix,coo_matrix
import json
from tools import *
from memory_profiler import profile
import sys
import argparse
from subCluster import *
# =============== 1.partition
# WARNING : EDGENUM < 32G Otherwise, it cannot be achieved.
# G_MEM: 16G
MAXEDGE = 900000000    # 
MAXSHUFFLE = 30000000   # 
#################

## pagerank+label Traversal gets the base subgraph
#@profile
def PRgenG(RAWPATH,nodeNUM,originPartNUM,savePath=None):
    GRAPHPATH = RAWPATH + "/graph.bin"
    TRAINPATH = RAWPATH + "/trainIds.bin"
    graph = torch.from_numpy(np.fromfile(GRAPHPATH,dtype=np.int32))
    src,dst = graph[::2],graph[1::2]
    edgeNUM = len(src)
    trainIds = torch.from_numpy(np.fromfile(TRAINPATH,dtype=np.int64))
    
    template_array = torch.zeros(nodeNUM,dtype=torch.int32)

    # Streaming edge data
    batch_size = len(src) // MAXEDGE + 1
    src_batches = torch.chunk(src, batch_size, dim=0)
    dst_batches = torch.chunk(dst, batch_size, dim=0)
    batch = [src_batches, dst_batches]

    inNodeTable = torch.zeros(nodeNUM,dtype=torch.int32,device="cuda")
    outNodeTable = torch.zeros(nodeNUM,dtype=torch.int32,device="cuda")
    nodeInfo = torch.zeros(nodeNUM,dtype=torch.int32,device="cuda") - 1
    trainIds = trainIds.cuda()
    nodeInfo[trainIds] = trainIds.to(torch.int32)

    for src_batch,dst_batch in zip(*batch):
        src_batch,dst_batch = src_batch.cuda(),dst_batch.cuda()
        dgl.lpGraph(src_batch,dst_batch,nodeInfo,inNodeTable,outNodeTable)  # Calculate the access with the 1-hop neighbor
    src_batch,dst_batch = None,None
    outNodeTable = outNodeTable.cpu() # innodeTable still in GPU for next use

    nodeValue = template_array.clone()
    # value setting
    nodeValue[trainIds] = 100000

    # random method
    print("start greedy cluster ...")
    # trainIdsInPart Indicates which label each training point should be in
    trainIdsInPart = genSmallCluster(trainIds,nodeInfo,originPartNUM)   
    TableNUM = 30   # Indicates that an int32 table stores a maximum of 30 labels
    labelTableLen = int((originPartNUM-1)/TableNUM + 1)
    
    nodeInfo = transPartId2Bit(trainIdsInPart,trainIds,nodeNUM,TableNUM,labelTableLen)

    emptyCache()
    nodeLayerInfo = []
    for _ in range(3):
        offset = 0
        acc_nodeValue = torch.zeros_like(nodeValue,dtype=torch.int32)
        acc_nodeInfo = torch.zeros_like(nodeInfo,dtype=torch.int32)
        for src_batch,dst_batch in zip(*batch):  
            tmp_nodeValue,tmp_nodeInfo = nodeValue.clone().cuda(),nodeInfo.clone().cuda() 
            src_batch,dst_batch = src_batch.cuda(), dst_batch.cuda()  
            dgl.per_pagerank(dst_batch,src_batch,inNodeTable,tmp_nodeValue,tmp_nodeInfo,labelTableNUM=labelTableLen)
            tmp_nodeValue, tmp_nodeInfo = tmp_nodeValue.cpu(),tmp_nodeInfo.cpu()
            acc_nodeValue += tmp_nodeValue - nodeValue
            acc_nodeInfo = acc_nodeInfo | tmp_nodeInfo
            offset += len(src_batch)
        nodeValue = nodeValue + acc_nodeValue
        nodeInfo = acc_nodeInfo
        tmp_nodeValue,tmp_nodeInfo=None,None
        nodeLayerInfo.append(nodeInfo.clone())
    src_batch,dst_batch,inNodeTable = None,None,None
    outlayer = torch.bitwise_xor(nodeLayerInfo[-1], nodeLayerInfo[-2]) # The outermost point will not have a connecting edge
    nodeLayerInfo = None
    emptyCache()

    # Just synthesize nodeInfo into the 64 version
    if (labelTableLen > 1):
        nodeInfo1 = nodeInfo[::2].to(torch.int64)
        nodeInfo2 = nodeInfo[1::2].to(torch.int64) << TableNUM
        nodeInfo = nodeInfo1 | nodeInfo2

        outlayer1 = outlayer[::2].to(torch.int64)
        outlayer2 = outlayer[1::2].to(torch.int64) << TableNUM
        outlayer = outlayer1 | outlayer2

    nodeInfo = nodeInfo.cuda()
    outlayer = outlayer.cuda()
    averDegree = edgeNUM / nodeNUM
    
    print("cluster start....")
    # mergeBound indicates the maximum number of initial partitions that can be combined for the final partition. 
    # For example, 32 -> 8 indicates that the four original partitions are merged into a new partition, which is 4
    # startCluster should return: 1.nodeInfo, 2. Partition map of the new subgraph, 3. Initial subgraph merge route (used to generate trainBatch)
    originNodeInfo = nodeInfo.clone().cuda()
    nodeInfo,subMap,subTrack = startCluster(nodeInfo, originPartNUM, 12000, (originPartNUM,averDegree,100))

    # Modify the three-hop node to the merged partition
    # Suppose partition B is merged into partition A. So B + A -> A
    # Note that when the node meets any of the following conditions, it serves as the three-hop node of the new partition A
    for index,bit_position in enumerate(subMap):
        track = torch.tensor(subTrack[bit_position],dtype = torch.int32, device = 'cuda')
        originPart = track[0]
        track = track[1:]
        curPart = 1 << originPart
        for mergePart in track:
            # A bound and B bound
            mergeNode1 = (((outlayer >> originPart) & 1) != 0) & (((outlayer >> mergePart) & 1) != 0)
            # A bound and not in B
            mergeNode2 = (((outlayer >> originPart) & 1) != 0) & ((originNodeInfo & (1 << mergePart)) == 0)
            # B bound and not in A
            mergeNode3 = ((originNodeInfo & (curPart)) == 0) & (((outlayer >> mergePart) & 1) != 0)
            # All bound node after merge
            mergeNodes = mergeNode1 | mergeNode2 | mergeNode3
            # These nodes serve as the final three-hop nodes of partition A
            # In addition, all other nodes in partition A are no longer three-hop nodes
            outlayer = outlayer & ~(1 << originPart)
            outlayer[mergeNodes] = outlayer[mergeNodes] | (1 << originPart)
            curPart = curPart | (1 << mergePart)
    originNodeInfo = None

    trainIds = trainIds.cpu()
    trainIdsInPart = trainIdsInPart.to(torch.int32).cuda()
    for index,bit_position in enumerate(subMap):
        # GPU : nodeIndex,outIndex
        nodeIndex = (nodeInfo & (1 << bit_position)) != 0
        outIndex  =  (outlayer & (1 << bit_position)) != 0  # Indicates whether it is a three-hop point
        nid = torch.nonzero(nodeIndex).reshape(-1).to(torch.int32).cpu()
        PATH = savePath + f"/part{index}"
        checkFilePath(PATH)
        DataPath = PATH + f"/raw_G.bin"
        NodePath = PATH + f"/raw_nodes.bin"
        PRvaluePath = PATH + f"/sortIds.bin"

        track = torch.tensor(subTrack[bit_position],dtype = torch.int32, device = 'cuda')
        subTrainIds = torch.zeros(0, dtype = torch.int32)
        for t in track:
            subTrainIds = torch.cat((subTrainIds, trainIds[torch.nonzero(trainIdsInPart == t).reshape(-1)]), dim = 0)
        TrainPath = PATH + f"/raw_trainIds.bin"
        saveBin(subTrainIds,TrainPath)

        selfLoop = np.repeat(subTrainIds.to(torch.int32), 2)
        saveBin(nid,NodePath)
        saveBin(selfLoop,DataPath)
        graph = graph.reshape(-1,2)
        sliceNUM = (edgeNUM-1) // (MAXEDGE//2) + 1
        offsetSize = (edgeNUM-1) // sliceNUM + 1
        offset = 0
        start = time.time()
        for i in range(sliceNUM):
            sliceLen = min((i+1)*offsetSize,edgeNUM)
            g_gpu = graph[offset:sliceLen]                  # part of graph
            g_gpu = g_gpu.cuda()
            gsrc,gdst = g_gpu[:,0],g_gpu[:,1]
            gsrcMask = nodeIndex[gsrc.to(torch.int64)]
            gdstMask = nodeIndex[gdst.to(torch.int64)]
            idx_gpu = torch.bitwise_and(gsrcMask, gdstMask) # This time also includes a triple jump side
            IsoutNode = outIndex[gdst.to(torch.int64)]
            idx_gpu = torch.bitwise_and(idx_gpu, ~IsoutNode) # The three-hop edge has been deleted
            subEdge = g_gpu[idx_gpu].cpu()
            saveBin(subEdge,DataPath,addSave=True)
            offset = sliceLen                       
        print(f"time :{time.time()-start:.3f}s")    
        partValue = nodeValue[nodeIndex]  
        _ , sort_indice = torch.sort(partValue,dim=0,descending=True)
        sort_nodeid = nid[sort_indice]
        saveBin(sort_nodeid,PRvaluePath)
    return subMap.shape[0]

# =============== 2.graphToSub    
def nodeShuffle(raw_node,raw_graph):
    srcs, dsts = raw_graph[::2], raw_graph[1::2]
    raw_node = convert_to_tensor(raw_node, dtype=torch.int32).cuda()
    srcs_tensor = convert_to_tensor(srcs, dtype=torch.int32)
    dsts_tensor = convert_to_tensor(dsts, dtype=torch.int32)
    uniTable = torch.ones(len(raw_node),dtype=torch.int32,device="cuda")
    batch_size = len(srcs) // (MAXEDGE//2) + 1
    src_batches = list(torch.chunk(srcs_tensor, batch_size, dim=0))
    dst_batches = list(torch.chunk(dsts_tensor, batch_size, dim=0))
    batch = [src_batches, dst_batches]
    src_emp,dst_emp = raw_node[:1].clone(), raw_node[:1].clone()    # padding , no use
    srcShuffled,dstShuffled,uniTable = dgl.mapByNodeSet(raw_node,uniTable,src_emp,dst_emp,rhsNeed=False,include_rhs_in_lhs=False)
    raw_node = raw_node.cpu()
    remap = None
    for index,(src_batch,dst_batch) in enumerate(zip(*batch)):
        srcShuffled,dstShuffled,remap = remapEdgeId(uniTable,src_batch,dst_batch,remap=remap,device=torch.device('cuda:0'))
        src_batches[index] = srcShuffled
        dst_batches[index] = dstShuffled 
    srcShuffled,dstShuffled=None,None
    srcs_tensor = torch.cat(src_batches).cpu()
    dsts_tensor = torch.cat(dst_batches).cpu()
    uniTable = uniTable.cpu()
    return srcs_tensor,dsts_tensor,uniTable

def trainIdxSubG(subGNode,trainSet):
    trainSet = torch.as_tensor(trainSet).to(torch.int32)
    Lid = torch.zeros_like(trainSet).to(torch.int32).cuda()
    dgl.mapLocalId(subGNode.cuda(),trainSet.cuda(),Lid)
    Lid = Lid.cpu().to(torch.int64)
    return Lid

dataInfo = {}
def rawData2GNNData(RAWDATAPATH,partitionNUM,LABELPATH):
    labels = np.fromfile(LABELPATH,dtype=np.int64)  
    for rank in range(partitionNUM):
        partProcess(rank,RAWDATAPATH,labels)
        emptyCache()

def partProcess(rank,RAWDATAPATH,labels):
    startTime = time.time()
    PATH = RAWDATAPATH + f"/part{rank}" 
    rawDataPath = PATH + f"/raw_G.bin"
    rawTrainPath = PATH + f"/raw_trainIds.bin"
    rawNodePath = PATH + f"/raw_nodes.bin"
    PRvaluePath = PATH + f"/sortIds.bin"
    SubTrainIdPath = PATH + "/trainIds.bin"
    SubIndptrPath = PATH + "/indptr.bin"
    SubIndicesPath = PATH + "/indices.bin"
    SubLabelPath = PATH + "/labels.bin"
    checkFilePath(PATH)
    coostartTime = time.time()
    data = np.fromfile(rawDataPath,dtype=np.int32)
    node = np.fromfile(rawNodePath,dtype=np.int32)
    trainidx = np.fromfile(rawTrainPath,dtype=np.int64)  
    print(f"loading data time : {time.time()-coostartTime:.4f}s")
    
    coostartTime = time.time()
    remappedSrc,remappedDst,uniNode = nodeShuffle(node,data)
    subLabel = labels[uniNode.to(torch.int64)]
    indptr, indices = cooTocsc(remappedSrc,remappedDst,sliceNUM=(len(data) // (MAXEDGE//2))) 
    print(f"coo data time : {time.time()-coostartTime:.4f}s")

    coostartTime = time.time()
    trainidx = trainIdxSubG(uniNode,trainidx)
    saveBin(subLabel,SubLabelPath)
    saveBin(trainidx,SubTrainIdPath)
    saveBin(indptr,SubIndptrPath)
    saveBin(indices,SubIndicesPath)
    print(f"save time : {time.time()-coostartTime:.4f}s")
    
    pridx = torch.as_tensor(np.fromfile(PRvaluePath,dtype=np.int32))
    remappedSrc,_,_ = remapEdgeId(uniNode,pridx,None,device=torch.device('cuda:0'))
    saveBin(remappedSrc,PRvaluePath)

    dataInfo[f"part{rank}"] = {'nodeNUM': len(node),'edgeNUM':len(data) // 2}
    print(f"map data time : {time.time()-startTime:.4f}s")
    print("-"*20)



# =============== 3.featTrans
def featSlice(FEATPATH,beginIndex,endIndex,featLen):
    blockByte = 4 # float32 4byte
    offset = (featLen * beginIndex) * blockByte
    subFeat = torch.as_tensor(np.fromfile(FEATPATH, dtype=np.float32, count=(endIndex - beginIndex) * featLen, offset=offset))
    return subFeat.reshape(-1,featLen)

def sliceIds(Ids,sliceTable):
    # Cut Ids into the range specified by sliceTable
    # Ids can only be the sorted result
    beginIndex = 0
    ans = []
    for tar in sliceTable[1:]:
        position = torch.searchsorted(Ids, tar)
        slice = Ids[beginIndex:position]
        ans.append(slice)
        beginIndex = position
    return ans

def genSubGFeat(SAVEPATH,FEATPATH,partNUM,nodeNUM,sliceNUM,featLen):
    # get slices
    emptyCache()
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM

    idsSliceList = [[] for i in range(partNUM)]
    for i in range(partNUM):
        file = SAVEPATH + f"/part{i}/raw_nodes.bin"
        ids = torch.as_tensor(np.fromfile(file,dtype=np.int32))
        idsSliceList[i] = sliceIds(ids,boundList)
    
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice(FEATPATH,beginIdx,endIdx,featLen).cuda()
        for index in range(partNUM):
            fileName = SAVEPATH + f"/part{index}/feat.bin"
            SubIdsList = idsSliceList[index][sliceIndex]
            t_SubIdsList = SubIdsList - beginIdx
            subFeat = sliceFeat[t_SubIdsList.to(torch.int64).cuda()]
            subFeat = subFeat.cpu()
            saveBin(subFeat,fileName,addSave=sliceIndex)

def genAddFeat(beginId,addIdx,SAVEPATH,FEATPATH,partNUM,nodeNUM,sliceNUM,featLen):
    # addIdx now in CUDA
    emptyCache()
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM

    file = SAVEPATH + f"/part{beginId}/raw_nodes.bin"
    ids = torch.as_tensor(np.fromfile(file,dtype=np.int32),device="cuda")
    addIdx.append(ids)  # Increases all indexes of the original loaded subgraph

    for i in range(partNUM+1):
        addIdx[i] = sliceIds(addIdx[i],boundList)

    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice(FEATPATH,beginIdx,endIdx,featLen).cuda()
        for index in range(partNUM + 1):
            if index == partNUM:
                fileName = SAVEPATH + f"/part{beginId}/feat.bin"
            else:
                fileName = SAVEPATH + f"/part{index}/addfeat.bin"
            SubIdsList = addIdx[index][sliceIndex]
            t_SubIdsList = SubIdsList - beginIdx
            addFeat = sliceFeat[t_SubIdsList.to(torch.int64)]   # t_SubIdsList is in CUDA
            addFeat = addFeat.cpu()
            saveBin(addFeat,fileName,addSave=sliceIndex)

# =============== 4.addFeat
cur ,res = [] ,[]
cur_sum, res_sum = 0, -1

def dfs(part_num,diffMatrix):
    global cur,res,cur_sum,res_sum
    if (len(cur) == part_num):
        if res_sum == -1:
            res = cur[:]
            res_sum = cur_sum
        elif cur_sum < res_sum:
            res_sum = cur_sum
            res = cur[:]
        return
    for i in range(0, part_num):
        if (i in cur or (res_sum != -1 and len(cur) > 0 and cur_sum + diffMatrix[cur[-1]][i] > res_sum)):
            continue
        if len(cur) != 0:
            cur_sum += diffMatrix[cur[-1]][i] 
        cur.append(i)
        dfs(part_num,diffMatrix)
        cur = cur[:-1]
        if len(cur) != 0:
            cur_sum -= diffMatrix[cur[-1]][i]

def cal_min_path(diffMatrix, nodesList, part_num, base_path):
    base_path += '/part'
    start = time.time()
    maxNodeNum = 0
    for i in range(part_num):
        path = base_path + str(i) + '/raw_nodes.bin'
        nodes = torch.as_tensor(np.fromfile(path, dtype=np.int32)).cuda()
        maxNodeNum = max(maxNodeNum, nodes.shape[0])
        nodesList.append(nodes)
    
    res1 = torch.zeros(maxNodeNum, dtype=torch.int32,device="cuda")
    res2 = torch.zeros(maxNodeNum, dtype=torch.int32,device="cuda")
    print(f"load all nodes {time.time() - start:.4f}s")
    for i in range(part_num):
        for j in range(i + 1,part_num):
            node1 = nodesList[i]
            node2 = nodesList[j]
            res1.fill_(0)
            res2.fill_(0)
            dgl.findSameNode(node1, node2, res1, res2)
            sameNum = torch.sum(res1).item()
            diffMatrix[i][j] = node2.shape[0] - sameNum # The additional loading required for j relative to i
            diffMatrix[j][i] = node1.shape[0] - sameNum

            # print("part{} shape:{},part{} shape:{}, identical nodes :{}".format(i,node1.shape,j,node2.shape,sameNum))

    start = time.time()
    dfs(part_num ,diffMatrix)
    print("dfs time: {}".format(time.time() - start))
    return maxNodeNum, res

def genFeatIdx(part_num, base_path, nodeList, part_seq, featLen, maxNodeNum):
    res1 = torch.zeros(maxNodeNum, dtype=torch.int32,device="cuda")
    res2 = torch.zeros(maxNodeNum, dtype=torch.int32,device="cuda")
    base_path += '/part'
    addIndex = [[] for _ in range(part_num)]
    for i in range(0, part_num):
        cur_part = part_seq[i]
        next_part = part_seq[(i+1) % part_num]
        curNode = nodeList[cur_part].cuda()
        nextNode = nodeList[next_part].cuda()
        curLen = curNode.shape[0]
        nextLen = nextNode.shape[0]
        
        res1.fill_(0)
        res2.fill_(0)
        dgl.findSameNode(curNode, nextNode, res1, res2)
        same_num = torch.sum(res1).item()
        
        # index
        maxlen = max(curLen,nextLen)
        res1_zero = torch.nonzero(res1[:maxlen] == 0).reshape(-1).to(torch.int32)
        res2_zero = torch.nonzero(res2[:maxlen] == 0).reshape(-1).to(torch.int32)
        res1_one = torch.nonzero(res1[:maxlen] == 1).reshape(-1).to(torch.int32)
        res2_one = torch.nonzero(res2[:maxlen] == 1).reshape(-1).to(torch.int32)

        if (nextLen > same_num):
            if(curLen < nextLen or curLen == nextLen):
                replaceIdx = res2_zero.cuda()
            elif(curLen > nextLen):
                replaceIdx = res2_zero[:nextLen - same_num].cuda()  # If the feat index is not clipped, the feat index is out of bounds
        else:
            replaceIdx = torch.Tensor([]).to(torch.int32)
            

        nextPath = base_path + str(next_part)
        sameNodeInfoPath = nextPath + '/sameNodeInfo.bin'
        diffNodeInfoPath = nextPath + '/diffNodeInfo.bin'
        if res1_one.shape[0] != 0:
            sameNode = torch.cat((res1_one, res2_one), dim = 0)
        else:
            sameNode = torch.Tensor([]).to(torch.int32)
        
        if res1_zero.shape[0] != 0:
            diffNode = torch.cat((res1_zero, res2_zero), dim = 0)
        else:
            diffNode = torch.Tensor([]).to(torch.int32)
        saveBin(sameNode.cpu(), sameNodeInfoPath)
        saveBin(diffNode.cpu(), diffNodeInfoPath)
        sameNode, diffNode = None, None

        # Cache the index that needs to be reloaded
        addIndex[next_part] = nodeList[next_part][replaceIdx.to(torch.int64)]
    return addIndex

def writeJson(path):
    with open(path, "w") as json_file:
        json.dump(dataInfo, json_file,indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PD', help='dataset name')
    parser.add_argument('--partNUM', type=int, default=8, help='Number of layers')
    args = parser.parse_args()

    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(current_file_path)
    relative_json_path = "../../datasetInfo.json"

    JSONPATH = os.path.join(root_dir, relative_json_path)
    print("JSON file path:", JSONPATH)
    partitionNUM = args.partNUM
    sliceNUM = 8
    with open(JSONPATH, 'r') as file:
        data = json.load(file)
    datasetName = [args.dataset] 

    for NAME in datasetName:
        GRAPHPATH = data[NAME]["rawFilePath"]
        maxID = data[NAME]["nodes"]
        subGSavePath = data[NAME]["processedPath"]
        
        startTime = time.time()
        partitionNUM = PRgenG(GRAPHPATH,maxID,partitionNUM,savePath=subGSavePath)
        print(f"partition all cost:{time.time()-startTime:.3f}s")

        RAWDATAPATH = data[NAME]["processedPath"]
        FEATPATH = data[NAME]["rawFilePath"] + "/feat.bin"
        LABELPATH = data[NAME]["rawFilePath"] + "/labels.bin"
        SAVEPATH = data[NAME]["processedPath"]
        nodeNUM = data[NAME]["nodes"]
        featLen = data[NAME]["featLen"] 
        MERGETIME = time.time()
        rawData2GNNData(RAWDATAPATH,partitionNUM,LABELPATH)
        print(f"trans graph cost time{time.time() - MERGETIME:.3f}s ...")
        
        diffMatrix = [[0 for _ in range(partitionNUM)] for _ in range(partitionNUM)]
        nodeList = []
        maxNodeNum,minPath = cal_min_path(diffMatrix , nodeList, partitionNUM, data[NAME]["processedPath"])
        
        dataInfo['path'] = res
        writeJson(SAVEPATH+f"/{NAME}.json")

        addIdx = genFeatIdx(partitionNUM, SAVEPATH, nodeList, res, featLen, maxNodeNum)
        genAddFeat(res[0],addIdx,SAVEPATH,FEATPATH,partitionNUM,nodeNUM,sliceNUM,featLen)