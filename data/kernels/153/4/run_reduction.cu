#include "includes.h"
__global__ void run_reduction(int *con, int *blockCon,int* ActiveList, int nActiveBlock, int* blockSizes)
{
int list_idx = blockIdx.x;
int tx = threadIdx.x;
int block_idx = ActiveList[list_idx];
int start = block_idx*blockDim.x * 2;
int blocksize = blockSizes[block_idx];
__shared__ int s_block_conv;
s_block_conv = 1;
__syncthreads();

if (tx < blocksize)
{
if (!con[start + tx])
s_block_conv = 0;
}
__syncthreads();

if(tx == 0)
{
blockCon[block_idx] = s_block_conv; // active list is negation of tile convergence (active = not converged)
}
}