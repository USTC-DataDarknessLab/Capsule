import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue
import numpy as np
import json

#变量控制原则 : 谁用谁负责
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
        self.cacheData = []     # 子图存储部分
        self.pipe = Queue()     # 采样存储管道
        self.sampledSubG = []   # 采样子图存储位置
        self.trainNUM = 0       # 训练集数目
        
        # config json 部分
        self.dataPath = ''
        self.batchsize = 0
        self.cacheNUM = 0
        self.partNUM = 0
        self.epoch = 0
        # ================
        
        self.trained = 0
        self.trainptr = 0   # 当前训练集读取位置
        self.loop = 0
        self.subGptr = -1   # 子图训练指针，记录当前已经预取的位置
        self.batch_called = 0   # 批预取函数调用次数
        self.trainingGID = 0 # 当前训练子图的ID
        self.nextGID = 0     # 下一个训练子图
        self.trainNodes = []            # 训练节点
        self.graphNodeNUM = 0 # 当前训练子图

        self.readConfig(confPath)
        self.trainNodeDict = self.loadingTrainID() # 训练节点
        self.executor = concurrent.futures.ThreadPoolExecutor(1) # 线程池
        self.trainSubGTrack = self.randomTrainList() # 训练轨迹
        
        #self.sample_flag = self.executor.submit(self.preGraphBatch) #发送采样命令
        self.loadingGraph()
        self.initNextGraphData()
        print(self.cacheData)

    def __len__(self):  
        return self.trainNUM
    
    def __getitem__(self, index):
        # 批数据预取 缓存1个
        if index % self.batchsize == 0:
            # 调用预取函数
            data = self.sample_flag.result()
            self.sample_flag = self.executor.submit(self.preGraphBatch) 
        
        # 获取采样数据
        if index % self.batchsize == 0:
            # 调用实际数据
            if self.pipe.qsize() > 0:
                self.sampledSubG = self.pipe.get()
            else: #需要等待
                data = self.read_data.result()
                self.sampledSubG = self.pipe.get()
        
        return self.sampledSubG[index % self.batchsize]

    def initNextGraphData(self):
        # 查看是否需要释放
        if len(self.cacheData) > 2:
            self.moveGraph()
        # 对于将要计算的子图(已经加载)，修改相关信息
        self.trainingGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        self.trainNodes = self.trainNodeDict[self.trainingGID]
        self.loop = ((len(self.trainNodes) - 1) // self.batchsize) + 1
        self.graphNodeNUM = len(self.cacheData[1]) / 2 # 获取当前节点数目
        # 对于辅助计算的子图，进行加载，以及加载融合边
        self.loadingGraph()
        self.nextGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        self.loadingHalo()
        

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
        # 加载子图所有训练集
        idDict = {}     
        for index in range(self.partNUM):
            idDict[index] = [i for i in range(10)]
            self.trainNUM += len(idDict[index])
        return idDict

    def loadingGraph(self):
        # 读取int数组的二进制存储， 需要将边界填充到前面的预存图中
        # 由self.subGptr变量驱动
        self.subGptr += 1
        subGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        filePath = self.dataPath + "/part" + str(subGID)
        srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
        rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
        # 转换为tensor : tensor_data = torch.from_numpy(data)
        print("subG {} read success".format(subGID))
        self.cacheData.append(srcdata)
        self.cacheData.append(rangedata)

    def moveGraph(self):
        self.cacheData[0] = self.cacheData[2]
        self.cacheData[1] = self.cacheData[3]
        # del self.cacheData[3]
        # del self.cacheData[2]
        self.cacheData = self.cacheData[0:2]
       
    def readNeig(self,nodeID):
        return self.src[self.bound[nodeID*2]:self.bound[nodeID*2+1]]

    def prefeat(self):
        pass
    
    def loadingHalo(self):
        # 要先加载下一个子图，然后再加载halo( 当前<->下一个 )
        # TODO 读取halo
        filePath = self.dataPath + "/part" + str(self.trainingGID)
        # edges 
        edges = np.fromfile(filePath+"/halo"+str(self.nextGID)+".bin", dtype=np.int32)
        print(edges)
        

    def randomTrainList(self):
        epochList = []
        for i in range(self.epoch + 1): # 额外多增加一行
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
        if self.batch_called > self.loop:
            self.batch_called = 0 #重置
            self.trainptr = 0
            self.initNextGraphData() # 当前epoch采样已经完成，则要预取下轮子图数据
            
        # 常规采样
        self.batch_called += 1
        bound = max(self.trainptr+self.batchsize,self.trainNUM)
        print("预取数据部分:{}:{}...".format(self.trainptr,bound))
        cacheData = []
        for i in range(bound-self.trainptr):
            sampleID = self.trainIDs[self.trainptr+i]
            sampleG = self.cacheData[0][self.cacheData[1][sampleID*2]:self.cacheData[1][sampleID*2+1]]
            cacheData.append(sampleG)
        self.pipe.put(cacheData)
        self.trainptr = bound%self.trainNUM # 循环读取
        return 0
    
def collate_fn(data):
    return data


if __name__ == "__main__":
    data = [i for i in range(20)]
    dataset = CustomDataset("./config.json")
    # train_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn,pin_memory=True)
    # for i in train_loader:
    #     print(i)