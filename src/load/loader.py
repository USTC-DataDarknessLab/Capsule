import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue
import numpy as np
import json
import time
import dgl
import torch
from dgl.heterograph import DGLBlock
import copy
import sys
import logging
import os
import gc
import psutil
from tools import *
#torch.set_printoptions(threshold=10000)
curFilePath = os.path.abspath(__file__)
curDir = os.path.dirname(curFilePath)

# Disable operation
# logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.INFO,filename=curDir+'/loader.log',filemode='w',
                    format='%(message)s',datefmt='%H:%M:%S')
                    #format='%(message)s')
logger = logging.getLogger(__name__)

"""
Data loading logic :#@profile
1. Generate training random sequences
2. Pre-load training nodes (all training nodes are loaded in)
2. Preload the graph collection
3. Constantly generate images
4. Release the current subgraph and load the next graph after the graph sampling is complete
"""
class CustomDataset(Dataset):
    def __init__(self,confPath):
        self.cacheData = []  # Subgraph store part
        self.indptr = [] # dst / bound
        self.indices = [] # src / edgelist
        self.graphPipe = Queue()  # Sampling storage pipeline
        self.lossG = False

        self.preFetchExecutor = concurrent.futures.ThreadPoolExecutor(1)  # Thread pool
        self.preFetchFlagQueue = Queue()
        self.preFetchDataCache = Queue()
        self.prefeat = 0

        #### system
        gpu = torch.cuda.get_device_properties(0) # use cuda:0
        self.gpumem = int(gpu.total_memory)

        #### config json ####
        self.dataPath = ''
        self.batchsize,self.cacheNUM,self.partNUM = 0,0,0
        self.maxEpoch,self.classes,self.epochInterval = 0,0,0
        self.featlen = 0
        self.fanout = []
        self.mem = 0
        self.edgecut, self.nodecut,self.featDevice = 0,0,""
        self.train_name,self.framework,self.mode,self.dataset = "","","",""
        self.readTrainConfig(confPath)  # load training args
        # ================
        self.maxPartNodeNUM = 0
        self.datasetInfo = self.readDatasetInfo()


        #### train log ####
        self.trainSubGTrack = self.randomTrainList()    # training trace
        self.subGptr = -1  # The subgraph training pointer, which records the current training position, changes when the graph is loaded
        
        #### Node type loading ####
        self.NodeLen = 0        # Used to record the number of nodes in the data set. The default is the number of train nodes
        self.trainNUM = 0       # Total number of training sets
        self.trainNodeDict,self.valNodeDict,self.testNodeDict = {},{},{}
        self.trainNodeNumbers,self.valNodeNumbers,self.testNodeNumbers = 0,0,0
        self.loadModeData(self.mode)

        #### Graph Structure Info ####
        self.graphNodeNUM = 0  # Number of graph nodes
        self.graphEdgeNUM = 0          # number of graph edges
        self.GID = 0            # ID of the current training subgraph
        self.subGtrainNodesNUM = 0      # Number of current training subgraph nodes
        self.trainNodes = []            # Training subgraph training node records   
        self.nodeLabels = []            # Subgraph tag
        self.trainptr = 0               # Current training set read location
        self.trainLoop = 0              # The number of times the current subgraph can be read
        #### mmap Feature ####
        #self.feats = torch.zeros([self.maxPartNodeNUM, self.featlen], dtype=torch.float32).to(self.featDevice)  ## CUDA oom
        self.map = []
        self.trainfeat = []
        self.memfeat = []
        #### prefetch data ####
        self.template_cache_graph , self.ramapNodeTable = self.initCacheData() # CPU , GPU
        self.initNextGraphData()
        self.uniTable = torch.zeros(len(self.ramapNodeTable),dtype=torch.int32).cuda()


    def __len__(self):
        return self.NodeLen
    
    def __getitem__(self, index):
        if index % self.batchsize == 0:
            self.preGraphBatch()
            cacheData = self.graphPipe.get()
            return tuple(cacheData[:4])

########################## init training data ##########################
    def readTrainConfig(self,confPath):
        with open(confPath, 'r') as f:
            config = json.load(f)
        self.train_name = config['train_name']
        self.dataPath = config['datasetpath']+"/"+config['dataset']
        self.dataset = config['dataset']
        self.batchsize = config['batchsize']
        self.cacheNUM = config['cacheNUM']
        self.partNUM = config['partNUM']
        self.maxEpoch = config['maxEpoch']
        self.featlen = config['featlen']
        self.fanout = config['fanout']
        self.framework = config['framework']
        self.mode = config['mode']
        self.classes = config['classes']
        self.epochInterval = config['epochInterval']
        self.mem = config['memUse']
        self.edgecut = config['edgecut']
        self.nodecut = config['nodecut']
        self.featDevice = config['featDevice']
        self.inCpu = self.featDevice == 'cpu'

    def readDatasetInfo(self):
        confPath = self.dataPath + f"/{self.dataset}.json"
        with open(confPath, 'r') as f:
            config = json.load(f)
        for partid in range(self.partNUM):
            self.maxPartNodeNUM = max(self.maxPartNodeNUM,config[f'part{partid}']["nodeNUM"])
        return config

    def randomTrainList(self): 
        epochList = []
        for _ in range(self.maxEpoch + 1): # Add an extra line
            tarinArray = np.array(self.datasetInfo["path"])
            epochList.append(tarinArray)
        return epochList

########################## Load/release graph structure data ##########################
    #@profile
    def initNextGraphData(self):
        # First get the contents of this load, and then send the prefetch command
        logger.info("----------initNextGraphData----------")
        start = time.time()
        self.subGptr += 1
        self.GID = self.trainSubGTrack[self.subGptr // self.partNUM][self.subGptr % self.partNUM]
        print(f"loading G :{self.GID}..")
        if self.subGptr == 0:
            self.loadingGraphData(self.GID) # The first one needs to be loaded from scratch
        else:
            taskFlag = self.preFetchFlagQueue.get()
            taskFlag.result()
            preCacheData = self.preFetchDataCache.get()
            self.loadingGraphData(self.GID,predata=preCacheData)
        emptyCache()
        self.trainNodes = self.trainNodeDict[self.GID]
        self.trainNodes = self.trainNodes[torch.randperm(self.trainNodes.size(0))]  # reshuffle
        self.subGtrainNodesNUM = self.trainNodeNumbers[self.GID]   
        self.trainLoop = ((self.subGtrainNodesNUM - 1) // self.batchsize) + 1
        self.preFetchFlagQueue.put(self.preFetchExecutor.submit(self.preloadingGraphData))  # Send the next prefetch command
        logger.info(f"loading next graph with {time.time() - start :.4f}s")

    def loadingTrainID(self):
        # Load all training sets of subgraph
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]  
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)   
            trainIDs = torch.as_tensor(np.fromfile(filePath+"/trainIds.bin",dtype=np.int64))
            numberList[index] = len(trainIDs)
            idDict[index] = trainIDs
            self.trainNUM += idDict[index].shape[0]
        return idDict,numberList

    def preloadingGraphData(self):
        # Convert only to numpy format for now
        start = time.time()
        ptr = self.subGptr + 1
        rank = self.trainSubGTrack[ptr // self.partNUM][ptr % self.partNUM]
        filePath = self.dataPath + "/part" + str(rank)
        indices = np.fromfile(filePath + "/indices.bin", dtype=np.int32)
        indptr = np.fromfile(filePath + "/indptr.bin", dtype=np.int32)
        nodeLabels = np.fromfile(filePath+"/labels.bin", dtype=np.int64)

        ###
        # Incremental feature loading
        map = self.map
        sameNodeInfoPath = filePath + '/sameNodeInfo.bin'
        diffNodeInfoPath = filePath + '/diffNodeInfo.bin'
        sameNode = torch.as_tensor(np.fromfile(sameNodeInfoPath, dtype = np.int32))
        diffNode = torch.as_tensor(np.fromfile(diffNodeInfoPath, dtype = np.int32))
        res1_one, res2_one = torch.split(sameNode, (sameNode.shape[0] // 2))
        
        newMap = torch.clone(map)   # newMap.device == map.device
        newMap[res2_one.to(torch.int64)] = map[res1_one.to(torch.int64)]
        if diffNode.shape[0] != 0:
            res1_zero, res2_zero = torch.split(diffNode, (diffNode.shape[0] // 2))
        else:
            res1_zero,res2_zero = torch.Tensor([]),torch.Tensor([])
        addFeat = torch.as_tensor(np.fromfile(filePath + "/addfeat.bin", dtype=np.float32).reshape(-1, self.featlen))
        replace_idx = map[res1_zero[:addFeat.shape[0]].to(torch.int64)].to(torch.int64).to(self.featDevice)
        newMap[res2_zero.to(torch.int64)] = map[res1_zero.to(torch.int64)]
        addFeatInfo = {"addFeat": addFeat, "replace_idx": replace_idx, "map": newMap} 
        self.preFetchDataCache.put([indices,indptr,addFeatInfo,nodeLabels])
        logger.info(f"pre data time :{time.time() - start:.4f}s...")
        return 0

    #@profile
    def loadingGraphData(self,subGID,predata=None):
        filePath = self.dataPath + "/part" + str(subGID)
        if predata == None:
            # First initialization full load
            self.indices = torch.as_tensor(np.fromfile(filePath + "/indices.bin", dtype=np.int32))
            self.indptr = torch.as_tensor(np.fromfile(filePath + "/indptr.bin", dtype=np.int32))
            self.nodeLabels = torch.as_tensor(np.fromfile(filePath + "/labels.bin", dtype=np.int64))
            addFeat = torch.as_tensor(np.fromfile(filePath + "/feat.bin", dtype=np.float32).reshape(-1, self.featlen))
            self.map = torch.arange(self.maxPartNodeNUM, dtype=torch.int32,device="cuda")
        else:
            # After the preload is complete, data is processed, and the prefetched data is kept in the CPU
            self.indices,self.indptr,self.map = None,None,None
            emptyCache()
            self.indices = torch.as_tensor(predata[0])
            self.indptr = torch.as_tensor(predata[1])
            self.nodeLabels = torch.as_tensor(predata[3])
            addFeatInfo = predata[2]
            addFeat = addFeatInfo['addFeat']
            self.map = addFeatInfo['map']  
            replace_idx = addFeatInfo['replace_idx']
        
        # Determine whether to crop, and then put in the GPU
        graphNodeNUM,graphEdgeNUM = int(len(self.indptr) - 1 ),len(self.indices)
        # if True:
        if not self.inCpu and countMemToLoss(graphEdgeNUM,graphNodeNUM,self.featlen,self.mem):
            self.lossG = True   # need trimming
            sortNode = torch.as_tensor(np.fromfile(filePath + "/sortIds.bin", dtype=np.int32))
            saveRatio = 0.6
            randomLoss = 0.8
            cutNode,saveNode = sortNode[int(graphNodeNUM*saveRatio):],sortNode[:int(graphNodeNUM*saveRatio)]
            start = time.time()
            self.indptr,self.indices,nodeMask = \
                streamLossGraph(self.indptr,self.indices,cutNode,sliceNUM=10,randomLoss=randomLoss,degreeCut=60,CutRatio=0.5)
            # print(f"loss_csr time :{time.time() - start:.4f}s...")
            start = time.time()
            # addFeat -> self.feat device
            sliceFeatNUM = 8
            if predata == None:
                self.maxMemNum = int(self.maxPartNodeNUM * (1 - saveRatio) * randomLoss) + 10
                self.maxCudaNum = self.maxPartNodeNUM - self.maxMemNum + 100

                self.memfeat = torch.zeros((self.maxMemNum, self.featlen), dtype=torch.float32)
                self.trainfeat = torch.zeros((self.maxCudaNum, self.featlen), dtype=torch.float32, device='cuda')

                init_cac(nodeMask, addFeat, self.memfeat, self.trainfeat, self.map)

            else:
                featAdd(replace_idx, addFeat, self.memfeat, self.trainfeat)
                loss_feat_cac(nodeMask, self.memfeat, self.trainfeat, self.map)
                print(f"loading feat time :{time.time() - start:.4f}s...")
        else:
            # No need for tailoring,csr,feat,label directly stored in cuda
            self.lossG = False 
            self.indptr,self.indices = self.indptr.cuda(),self.indices.cuda()
            if predata == None: 
                # Indicates first load, direct migration
                self.trainfeat = torch.zeros((self.maxPartNodeNUM, self.featlen), dtype=torch.float32, device=self.featDevice)
                idx = torch.arange(addFeat.shape[0],dtype=torch.int64,device=self.featDevice)
                addFeat = torch.as_tensor(addFeat)
                streamAssign(self.trainfeat,idx,addFeat,sliceNUM=4)
            else:
                # Stream processing
                streamAssign(self.trainfeat,replace_idx,addFeat,sliceNUM=4)



########################## Sample graph structure ##########################
    def sampleNeigGPU_NC(self,sampleIDs,cacheGraph,batchlen):     
        logger.info("----------[sampleNeigGPU_NC]----------")
        sampleIDs = sampleIDs.to(torch.int32).to('cuda:0')
        sampleStart = time.time()
        ptr,seedPtr,NUM = 0, 0, 0
        mapping_ptr = [ptr]
        for l, fan_num in enumerate(self.fanout):
            if l == 0:
                seed_num = batchlen
            else:
                seed_num = len(sampleIDs)
            self.ramapNodeTable[seedPtr:seedPtr+seed_num] = sampleIDs
            seedPtr += seed_num
            out_src = cacheGraph[0][ptr:ptr+seed_num*fan_num]
            out_dst = cacheGraph[1][ptr:ptr+seed_num*fan_num]
            
            if self.lossG == False:
                NUM = dgl.sampling.sample_with_edge(self.indptr,self.indices,
                    sampleIDs,seed_num,fan_num,out_src,out_dst)
            elif self.lossG == True:
                NUM = dgl.sampling.sample_with_edge_and_map(self.indptr,self.indices,
                    sampleIDs,seed_num,fan_num,out_src,out_dst,self.map)
            sampleIDs = cacheGraph[0][ptr:ptr+NUM.item()]
            ptr=ptr+NUM.item()
            mapping_ptr.append(ptr)
        self.ramapNodeTable[seedPtr:seedPtr+NUM] = sampleIDs
        seedPtr += NUM 
        logger.info("Sample Neighbor Time {:.5f}s".format(time.time()-sampleStart))
        mappingTime = time.time()        
        cacheGraph[0] = cacheGraph[0][:mapping_ptr[-1]]
        cacheGraph[1] = cacheGraph[1][:mapping_ptr[-1]]
        unique = self.uniTable.clone()
        logger.info("construct remapping data Time {:.5f}s".format(time.time()-mappingTime))
        
        t = time.time()  
        #cacheGraph[0],cacheGraph[1],unique = dgl.remappingNode(cacheGraph[0],cacheGraph[1],unique)
        
        cacheGraph[0],cacheGraph[1],unique = dgl.mapByNodeSet(self.ramapNodeTable[:seedPtr],unique,cacheGraph[0],cacheGraph[1])
        # print("min idx",torch.min(self.lossMap[unique.to(torch.int64)]))
        # print("max map idx",torch.max(self.lossMap))
        # print("self.feats:",self.feats.shape)
        # print("unique:",unique)
        # print("unique.shape:",unique.shape)
        logger.info("cuda remapping func Time {:.5f}s".format(time.time()-t))
        transTime = time.time()
        if self.framework == "dgl":
            layerNUM = len(mapping_ptr) - 1
            blocks = []
            dstNUM, srcNUM = 0, 0
            for layer in range(1,layerNUM+1):
                src = cacheGraph[0][:mapping_ptr[layer]]
                dst = cacheGraph[1][:mapping_ptr[layer]]
                data = (src,dst)
                if layer == 1:
                    dstNUM,_ = torch.max(dst,dim=0)
                    srcNUM,_ = torch.max(src,dim=0)
                    dstNUM += 1
                    srcNUM += 1      
                elif layer == layerNUM:
                    dstNUM = srcNUM
                    srcNUM = len(unique)
                else:
                    dstNUM = srcNUM
                    srcNUM,_ = torch.max(src,dim=0)
                    srcNUM += 1
                block = self.create_dgl_block(data,srcNUM,dstNUM)
                blocks.insert(0,block)
        elif self.framework == "pyg":
            src = cacheGraph[0][:mapping_ptr[-1]].to(torch.int64)
            dst = cacheGraph[1][:mapping_ptr[-1]].to(torch.int64)
            blocks = torch.stack((src, dst), dim=0)
        logger.info("trans Time {:.5f}s".format(time.time()-transTime))
        logger.info("==>sampleNeigGPU_NC() func time {:.5f}s".format(time.time()-sampleStart))
        logger.info("-"*30)
        return blocks,unique
    
    def initCacheData(self):
        if self.train_name == "NC":
            number = self.batchsize
        else:
            number = self.batchsize * 3
        tmp = number
        cacheGraph = [[],[]]
        remapTable = []
        for _, fan in enumerate(self.fanout):
            dst = torch.full((tmp * fan,), -1, dtype=torch.int32).cuda()  # Using the PyTorch tensor, specify dtype
            src = torch.full((tmp * fan,), -1, dtype=torch.int32).cuda()  # Using the PyTorch tensor, specify dtype
            cacheGraph[0].append(src)
            cacheGraph[1].append(dst)
            tmp = tmp * (fan + 1)
        remapTable = copy.deepcopy(cacheGraph[0])
        remapTable.append(cacheGraph[1][-1])
        remapTable = torch.cat(remapTable,dim=0).to(torch.int32).cuda()
        cacheGraph[0] = torch.cat(cacheGraph[0],dim=0)
        cacheGraph[1] = torch.cat(cacheGraph[1],dim=0)
        return cacheGraph ,remapTable

    def preGraphBatch(self):
        preBatchTime = time.time()
        if self.graphPipe.qsize() >= self.cacheNUM:
            return 0
        if self.trainptr == self.trainLoop:
            logger.debug("trigger of cache reload ,ptr:{}".format(self.trainptr))
            self.trainptr = 0           
            self.initNextGraphData()
        cacheTime = time.time()
        cacheGraph = copy.deepcopy(self.template_cache_graph)
        sampleIDs = -1 * torch.ones(self.batchsize,dtype=torch.int64)
        logger.info("construct copy graph and label cost {:.5f}s".format(time.time()-cacheTime))
        
        createDataTime = time.time()
        batchlen = 0
        if self.trainptr < self.trainLoop - 1:
            # full batch
            sampleIDs = self.trainNodes[self.trainptr*self.batchsize:(self.trainptr+1)*self.batchsize]
            batchlen = self.batchsize
            cacheLabel = self.nodeLabels[sampleIDs]
        else:
            offset = self.trainptr*self.batchsize
            sampleIDs = self.trainNodes[offset:self.subGtrainNodesNUM]
            batchlen = self.subGtrainNodesNUM - offset
            cacheLabel = self.nodeLabels[sampleIDs]
        logger.info("prepare sample data Time cost {:.5f}s".format(time.time()-createDataTime))    

        ##
        sampleTime = time.time()
        blocks,uniqueList = self.sampleNeigGPU_NC(sampleIDs,cacheGraph,batchlen)
        logger.info("sample subG all cost {:.5f}s".format(time.time()-sampleTime))
        ##
     
        featTime = time.time()
        cacheFeat = self.featMerge(uniqueList)
        logger.info("feat merge cost {:.5f}s".format(time.time()-featTime))
        
        cacheData = [blocks,cacheFeat,cacheLabel,batchlen]
        self.graphPipe.put(cacheData)

        self.trainptr += 1
        logger.info("-"*30)
        logger.info("preGraphBatch() cost {:.5f}s".format(time.time()-preBatchTime))
        logger.info("="*30)
        logger.info("\t")
        return 0

########################## Extract Feature ##########################
    def featMerge(self,uniqueList):
        featTime = time.time() 
        if self.lossG == False:
            featIdx = self.map[uniqueList.to(torch.int64).to(self.trainfeat.device)]
            test = self.trainfeat[featIdx.to(torch.int64)]
        elif self.lossG == True:
            mapIdx = self.map[uniqueList.to(self.map.device).to(torch.int64)].to(torch.int64)     
            test = self.trainfeat[mapIdx.to(self.trainfeat.device)]    
        logger.info("subG feat merge cost {:.5f}s".format(time.time()-featTime))
        return test
    
########################## Data adjustment ##########################    
    def loadModeData(self,mode):
        logger.info("loading mode:'{}' data".format(mode))
        self.trainNodeDict,self.trainNodeNumbers = self.loadingTrainID() # Training node dictionary, number of training nodes
        self.NodeLen = self.trainNUM
 
    def create_dgl_block(self, data, num_src_nodes, num_dst_nodes):
        row, col = data
        gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, row, col, 'coo')
        g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        return g

def collate_fn(data):
    return data[0]



if __name__ == "__main__":
    dataset = CustomDataset(curDir+"/../../config/RD_pyg.json")
    with open(curDir+"/../../config/RD_pyg.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['maxEpoch']
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize,collate_fn=collate_fn)
    count = 0
    for index in range(2):
        start = time.time()
        loopTime = time.time()
        for graph,feat,label,number in train_loader:
            # print(graph)
            # print(feat.shape)
            # print(label)
            # print(number)
            # print('-'*20)
            count = count + 1
            if count % 20 == 0:
                print("loop time:{:.5f}".format(time.time()-loopTime))
        print("="*20)
        print("all loop time:{:.5f}".format(time.time()-loopTime))
        print("="*20)