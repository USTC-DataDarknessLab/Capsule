#include <cuda_runtime.h>
#include "sample.h"
#include <cassert>
#include <iostream>
template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void sample_2hop_kernel(
    int* bound,int* graphEdge,int* seed,
    int seed_num,int fanout,
    int* out_src,int* out_dst,curandState *random_states) {
    
    assert(BLOCK_SIZE == blockDim.x);

    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   
    curandState local_state = random_states[idx];
    for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
        if (index < seed_num) {
            int rid = seed[index];
            int off = bound[rid];
            int len = bound[rid + 1] - bound[rid];
            if (len <= fanout) {
                size_t j = 0;
                for (; j < len; ++j) {
                    out_src[index * fanout + j] = rid;
                    out_dst[index * fanout + j] = graphEdge[off + j];
                }

                // for (; j < fanout; ++j) {
                //     out_src[index * fanout + j] = -1;
                //     out_dst[index * fanout + j] = -1;
                // }
            } else {
                for (int j = 0; j < fanout; ++j) {
                    //int selected_j = curand(&local_state) % (len - j);
                    int selected_j = curand(&local_state) % (len);
                    // int selected_j = j;
                    int selected_node_id = graphEdge[off + selected_j];
                    out_src[index * fanout + j] = rid;
                    out_dst[index * fanout + j] = selected_node_id;
                    //indices[off + selected_j] = indices[off+len-j-1];
                    //indices[off+len-j-1] = selected_node_id;
                }
            } 
        }
    }
    random_states[idx] = local_state;
}   


inline int RoundUpDiv(int target, int unit) {
  return (target + unit - 1) / unit;
}

void sample_2hop(
    int* bound,int* graphEdge,int* seed,
    int seed_num,int fanout,int* out_src,
    int* out_dst)
{   

    const int slice = 1024;
    const int blockSize = 256;
    int batchsize = 1024;
    int steps = RoundUpDiv(seed_num,slice);
    GPURandomStates randomStates(steps*blockSize);
    
    dim3 grid(steps);
    dim3 block(blockSize);
    for (int i = 0 ; i < 10 ; i++)
        std::cout << "out_src:" << out_src[i] << " out_dst:" << out_dst[i] << std::endl;
    // std::cout << "============hello world=================" << std::endl;
    sample_2hop_kernel<blockSize, slice>
    <<<grid,block>>>(bound,graphEdge,seed,
    seed_num,fanout,out_src,out_dst,randomStates.GetStates());
    // for (int i = 0 ; i < 10 ; i++)
    //     std::cout << "out_src:" << out_src[i] << " out_dst:" << out_dst[i] << std::endl;
}