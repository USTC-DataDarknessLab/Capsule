import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue
import numpy as np
import json


"""
数据加载的逻辑:
    1.生成训练随机序列
    2.预加载训练节点(所有的训练节点都被加载进入)
    2.预加载图集合(从初始开始就存入2个)
    3.不断生成采样子图
    4.当图采样完成后释放当前子图,加载下一个图
"""
class CustomDataset(Dataset):
    def __init__(self,confPath):
        # 获得训练基本信息
        self.readConfig(confPath)
        self.graphTrack = self.randomTrainList() # 训练轨迹
        self.trainList = self.loadingTrainID() # 训练节点
        self.executor = concurrent.futures.ThreadPoolExecutor(1) # 线程池
        self.trainTrack = self.randomTrainList() # 获得随机序列
        
        self.sample_flag = None
        
        self.cacheData = []
        self.pipe = Queue()
        self.trained = 0
        self.read = 0
        self.loop = ((len(self.trainIDs)-1) // self.batchsize) + 1
        self.read_called = 0

        #self.initGraphData()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if index % self.batchsize == 0:
            # 调用预取函数
            if self.read_called < self.loop:
                if self.sample_flag is None:
                    future = self.executor.submit(self.preGraphBatch)
                    self.sample_flag = future
                else:
                    data = self.sample_flag.result()
                    self.sample_flag = self.executor.submit(self.preGraphBatch) 
            # 调用实际数据
            if self.pipe.qsize() > 0:
                self.cacheData = self.pipe.get()
            else: #需要等待
                data = self.read_data.result()
                self.cacheData = self.pipe.get()
        return self.cacheData[index % self.batchsize]

    def initGraphData(self):
        self.loadingGraph(self.trainList[0][0])
        self.loadingGraph(self.trainList[0][1])
        
    def readConfig(self,confPath):
        with open(confPath, 'r') as f:
            config = json.load(f)
        self.dataPath = config['datasetpath']+"/"+config['dataset']
        self.batchsize = config['batchsize']
        self.cacheNUM = config['cacheNUM']
        self.partNUM = config['partNUM']
        self.epoch = config['epoch']
        formatted_data = json.dumps(config, indent=4)
        print(formatted_data)

    def loadingTrainID(self):
        idDict = {}
        for index in range(self.partNUM):
            idDict[index] = [i for i in range(10)]
        return idDict

    def loadingGraph(self,subGID):
        # 读取int数组的二进制存储
        # 需要将边界填充到前面的预存图中
        filePath = self.dataPath + "/part" + str(subGID)
        srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
        rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
        # 转换为tensor : tensor_data = torch.from_numpy(data)
        self.cacheData.append(srcdata)
        self.cacheData.append(rangedata)
    
    def loadingHalo(self,mainG,nextG):
        # 加载数据并直接插入到cache的对应位置中
        pass
    
    def mergeGraph(self):
        pass

    def removeGraph(self):
        # 释放图内存
        pass

    def readNeig(self,nodeID):
        return self.src[self.bound[nodeID*2]:self.bound[nodeID*2+1]]

    def prefeat(self):
        pass
    
    def randomTrainList(self):
        epochList = []
        for i in range(self.epoch):
            random_array = np.random.choice(np.arange(0, self.partNUM), size=self.partNUM, replace=False)
            if len(epochList) == 0:
                epochList.append(random_array)
            else:
                # 已经存在列
                lastid = epochList[-1][-1]
                while(lastid == random_array[0]):
                    random_array = np.random.choice(np.arange(0, self.partNUM), size=self.partNUM, replace=False)
                epochList.append(random_array)
            
        return epochList
    
    def preGraphBatch(self):
        # 预取一个batch的采样图，如果当前子图的采样节点(训练节点)已经采样完成，则触发释放和重取图数据机制
        if self.read_called > self.loop:
            return 0
        self.read_called += 1
        print("预取数据部分:{}:{}...".format(self.read,self.read+self.batchsize))
        cacheData = []
        for i in range(self.batchsize):
            sampleID = self.trainIDs[self.read+i]
            sampleG = self.src[self.bound[sampleID]:self.bound[sampleID+1]]
            cacheData.append(sampleG)
        self.pipe.put(cacheData)
        self.read += self.batchsize
        return 0
    
def collate_fn(data):
    return data


if __name__ == "__main__":
    data = [i for i in range(20)]
    dataset = CustomDataset("./processed/part0")


    # train_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn,pin_memory=True)
    # for i in train_loader:
    #     print(i)