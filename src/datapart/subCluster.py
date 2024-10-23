import numpy as np
import os
import torch
import dgl
import gc
import time
import torch
import math

# The above process is completed in CUDA

subGNUM = []    # 
subGCost = []   #subGCost[i] Indicates the total cost of subgraph i
subGMap = []    #subGMap[i] represents the actual subgraph location of the current subgraph i (for merge operations)
subGWeight = [] #weight indicates how many initial subpartitions the partition is combined from
subGTrack = []  #subGTrack[i] = [] Indicates which initial partitions constitute partition i

mergeBound = 0
averDegree = 0
featLen = 0

def findMinPart(partInfo):
    min_value, min_index = torch.min(partInfo, dim=0)
    return min_value, min_index

def findMaxPart(partInfo):
    max_value, max_index = torch.max(partInfo, dim=0)
    return max_value, max_index

def changeInfo(partInfo,changeIdx,changeValue,acc=True):
    if acc:
        partInfo[changeIdx] += changeValue
    else:
        partInfo[changeIdx] = changeValue

def genSmallCluster(trainids,nodeTable,partNUM):
    torch.cuda.empty_cache()
    gc.collect()
    # Extract the training set label
    trainLabel = nodeTable[trainids.to(torch.int64)]

    # Clustering by label (quantity varies)
    labelCluster = torch.zeros(torch.max(trainLabel).item()+1,dtype=torch.int32,device="cuda")
    dgl.bincount(trainLabel,labelCluster)
    
    startTime = time.time()
    sortLabelsNUM,labelIdx = torch.sort(labelCluster,descending=True)

    bound = len(torch.nonzero(sortLabelsNUM > 0).reshape(-1))
    sortLabelsNUM = sortLabelsNUM[:bound]
    labelIdx = labelIdx[:bound]
    
    # The small clusters are merged to produce a specific number of clusters
    partInfo = torch.zeros(partNUM,dtype=torch.int64,device="cuda")
    label2part = torch.zeros(torch.max(labelIdx).item()+1,dtype=torch.int64)

    # Scramble labelIdx directly, then partNUM is divided equally
    shuffle_idx = torch.randperm(labelIdx.shape[0])
    labelIdx = labelIdx[shuffle_idx]
    left = 0
    bound = right = labelIdx.shape[0] // partNUM + 1
    for i in range(partNUM):
        label2part[labelIdx[left:right].to(torch.int64)] = i
        left = right
        right = min(labelIdx.shape[0], right + bound)

    print(f"mergeTime time :{time.time()-startTime:.2f}s")
    startTime = time.time()
    # At this time, all the small clusters are put into the large cluster, id: 0 - (partNUM-1)
    trainIdsInPart = label2part[trainLabel.to(torch.int64)] # Finally get the partition in which each training point is located
    for i in range(partNUM):
        print(f"{i}:{torch.nonzero(trainIdsInPart == i).shape[0]}", end = "|")
    return trainIdsInPart
    # Finally, nodeInfo[trainids] = trainIdsInPart was used during training

def countLabelNUM(nodeTable,label):
    # Calculate how many values the corresponding Label has and synchronize subGNUM
    nodeIndex = (nodeTable & (1 << label)) != 0
    NUM = nodeIndex.sum()
    subGNUM[label] = NUM
    return NUM

def findLabelNodes(nodeTable,label):
    # Queries the vertex of the corresponding Label
    nodeIndex = (nodeTable & (1 << label)) != 0
    NodeIds = torch.nonzero(nodeIndex).reshape(-1)
    return NodeIds.to(torch.int32).cuda()


def mergeLabelCost(nodeTable,label_1,label_2):
    # Calculate how much it costs to merge two tags
    Label1Nodes = findLabelNodes(nodeTable,label_1)
    Label2Nodes = findLabelNodes(nodeTable,label_2)
    indexTable1 = torch.zeros_like(Label1Nodes,dtype=torch.int32,device="cuda")
    indexTable2 = torch.zeros_like(Label2Nodes,dtype=torch.int32,device="cuda")
    dgl.findSameNode(Label1Nodes,Label2Nodes,indexTable1,indexTable2)   # return 0/1
    sameNum = torch.sum(indexTable1).item()
    diffNum = (Label1Nodes.shape[0] + Label2Nodes.shape[0] - sameNum)
    # Two points need to be considered here: 1. The average output of each point; 2. The magnitude of each point's features
    # Need to ensure that the graph structure does not exceed the user specified memory +
    # At the same time, it should be ensured that all partitions in general can be stuffed into video memory
    return (diffNum * averDegree + diffNum * featLen) / 1024 / 1024 * 4


def strategy_single(nodeTable, maxBound):
    # Merge two partitions with the highest and lowest cost of the same level, when either partition exceeds maxBound or the level exceeds mergeBound
    for i in range(int(math.log2(mergeBound))):
        while True:
            validIdx = torch.nonzero(subGWeight == i + 1).reshape(-1)
            if (validIdx.shape[0] < 2):
                break
            costs = subGCost[validIdx]
            costs,idx = torch.sort(costs)
            idx = torch.tensor([idx[0],idx[-1]])
            if (idx[0] == idx[-1]):
                continue
            validIdx = validIdx[idx]# Take the largest and smallest two
            validIdx,_ = torch.sort(validIdx) # Sort by index, because later merge is fixed to merge into small tags
            curCost = mergeLabelCost(nodeTable,validIdx[0],validIdx[1])
            
            if (curCost > maxBound):
                subGWeight[validIdx] += 1# Move these two nodes to the next level, i.e. skip them
                break
            
            # Perform merge operations at this time to maintain public data

            mergeLabel(nodeTable, validIdx[0],validIdx[1])
            subGWeight[validIdx[0]] *= 2# Comes from the merger of two levels of the same level
            subGWeight[validIdx[1]] = -1
            subGCost[validIdx[0]] = curCost
            subGCost[validIdx[1]] = -1


def mergeMain(nodeTable, maxBound, strategyName):
    
    # Find two labels that can be merged
    #1. Random merge, randomly select two partitions of the same magnitude to merge, if the cost is unbearable, it will be randomized again. (The random order should be established in advance to prevent repeated random numbers)
    #2. Greedy merge, choose two partitions of the same magnitude to merge, requiring the cost of the merger to be locally optimal
    #3. Exhaustive merge, exhaustive all methods, to find the global optimal way

    # Also note that the process is 32-16-8 because the partitions of the same magnitude are preferentially selected for merging
    # or choose the partition that can be merged first, the process may become 32-31-30-28 -... . However, this approach may result in uneven partition sizes in the end

    # Here is how to implement: simple greedy merge and preferentially select the same magnitude partition
    strategy_func = globals()["strategy_" + strategyName]
    result = strategy_func(nodeTable, maxBound)
    return result

def mergeLabel(nodeTable,label_1,label_2):
    value_1 = ((nodeTable >> label_1) & 1) != 0 # Whether to be in the label_1 cluster
    value_2 = ((nodeTable >> label_2) & 1) != 0 # Whether to be in the label_2 cluster

    mergeNodes = value_1 | value_2

    max_position = max(label_1, label_2)
    min_position = min(label_1, label_2)

    # Set the position corresponding to the large value to 0
    newTable = nodeTable & ~(1 << max_position) # Big label returns to 0
    newTable = newTable | (1 << min_position)   # Small label returns to 1

    nodeTable[mergeNodes] = newTable[mergeNodes]    # The operation is performed only when at least one position is 1

    subGNUM[label_2] = 0
    subGNUM[label_1] = countLabelNUM(nodeTable, label_1)
    subGTrack[label_1] += subGTrack[label_2]

"""
tensor = torch.tensor([1, 5, 10, 15, 4], dtype=torch.int32)
result = custom_operation(tensor, 0, 2)
"""

#maxBound should be in MB
#mergeBound: The merge threshold, i.e. the maximum number of raw partition merges supported. 
# For example, the initial 32 partition, mergeBount = 4, means that a maximum of 4 initial partitions are supported, 
# and the final result is generally 32/4 = 8 subgraph partitions
def startCluster(nodeTable,cluserNUM,maxBound,globalData):

    # nodeTable initially set the number of tags to 32, and then merge to a value, 
    # according to my understanding, it is best to 32->16->8 and generally not to 4, because it will explode
    # But UK is fine, so that's it
    
    # trans.py's calling function
    global subGNUM,subGWeight,subGMap,subGCost,subGTrack
    global mergeBound,averDegree,featLen
    subGCost = []
    subGMap = []
    subGTrack = []

    mergeBound,averDegree,featLen = globalData
    subGNUM = torch.tensor([ 0 for _ in range(cluserNUM)],dtype=torch.int32)
    subGWeight = torch.tensor([ 1 for _ in range(cluserNUM)],dtype=torch.int32)
    # Initialize the subgraph cost. cost = (nodeNUM * averDegree * 4 + nodeNUM * 4 * featLen) / 104/1024 (unit MB)
    for i in range(cluserNUM):
        nodeNUM = findLabelNodes(nodeTable, i).shape[0]
        cost = (nodeNUM * averDegree + nodeNUM * featLen) / 1024 / 1024 * 4
        subGCost.append(cost)
        subGTrack.append([i])

    subGCost = torch.tensor(subGCost, dtype = torch.float32).reshape(-1)
    subGMap = torch.arange(cluserNUM, dtype = torch.int32).reshape(-1)

    # Calculates the initial size of each partition
    for labelId in range(cluserNUM):
        countLabelNUM(nodeTable,labelId)
    
    mergeMain(nodeTable, maxBound, "single")
    subGMapRes = torch.nonzero(subGNUM != 0).reshape(-1)
    print(f"Final partition result: {subGNUM[torch.nonzero(subGNUM != 0).reshape(-1)]}\n cost of each partition: {subGCost[torch.nonzero(subGNUM != 0).reshape(-1)]}")
    # Returns the result of the last merge
    # Returns the actual partition location
    # Return to trainBatch
    return nodeTable,subGMapRes,subGTrack


def transPartId2Bit(trainIdsInPart,trainIds,nodeNUM,TableNUM,labelTableLen):
    MultNodeLabels = []
    for i in range(labelTableLen):
        MultNodeLabels.append(torch.zeros(nodeNUM,dtype=torch.int32,device="cuda"))

    #labelTableLen treat as different dimensions
    #0 - 29, 30 - 59, 60 - 99...
    for i in range(labelTableLen):
        curMask = (trainIdsInPart > (i * TableNUM - 1)) & (trainIdsInPart < ((i + 1) * TableNUM))
        curTrainIdsInPartIdx = torch.nonzero(curMask).reshape(-1).to(torch.int64)
        MultNodeLabels[i][trainIds[curTrainIdsInPartIdx]] = (1 << (trainIdsInPart[curTrainIdsInPartIdx] % TableNUM)).to(torch.int32).cuda()
    nodeInfo = torch.stack(tuple(MultNodeLabels),dim = 1).reshape(-1).cpu()
    return nodeInfo