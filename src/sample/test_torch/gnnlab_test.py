import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import signn
import time
import struct
import os
def val_test(graphEdge,boundList,nodeID,valArray):
    edges = graphEdge[boundList[nodeID]:boundList[nodeID+1]]
    print(valArray)
    print(edges)

if __name__ == "__main__":
    graphEdge = []
    boundList = []
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    file_path = './../../data/products_4/part0/srcList.bin'
    graphEdge = np.fromfile(file_path, dtype=np.int32)
    graphEdge = torch.tensor(graphEdge).to('cuda:0')
    file_path = "./../../data/products_4/part0/range.bin"
    boundList = np.fromfile(file_path, dtype=np.int32)
    boundList = torch.tensor(boundList).to('cuda:0')

    seed_num = 16
    seed = [i for i in range(seed_num)]
    graphEdge = torch.Tensor(graphEdge).to(torch.int).to('cuda:0')
    boundList = torch.Tensor(boundList).to(torch.int).to('cuda:0')
    seed = torch.Tensor(seed).to(torch.int).to('cuda:0')
    
    fanout = 4
    out_src = [-1 for i in range(seed_num*fanout)]
    out_src = torch.Tensor(out_src).to(torch.int).to('cuda:0')
    out_dst = [-1 for i in range(seed_num*fanout)]
    out_dst = torch.Tensor(out_dst).to(torch.int).to('cuda:0')
    print(out_src)
    print(out_dst)

    start = time.time()
    signn.torch_sample_2hop(boundList,graphEdge,seed,seed_num,fanout,out_src,out_dst)
    print("comput time:",time.time()-start)
    time.sleep(2)
    out_src.to('cpu')
    out_dst.to('cpu')
    print(out_src)
    print(out_dst)
