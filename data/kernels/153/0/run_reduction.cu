#include "includes.h"
__global__ void run_reduction(bool *con, bool *blockCon,int* ActiveList, int nActiveBlock, int* blockSizes)
{
int list_idx = blockIdx.y*gridDim.x + blockIdx.x;
int maxblocksize = blockDim.x;
int tx = threadIdx.x;
int block_idx = ActiveList[list_idx];

int blocksize = blockSizes[block_idx];

__shared__ bool s_block_conv;


s_block_conv = true;
__syncthreads();

if(tx < blocksize)
{
if(!con[maxblocksize*block_idx+tx])
s_block_conv= false;
}
__syncthreads();

if(tx == 0)
{
blockCon[block_idx] = s_block_conv; // active list is negation of tile convergence (active = not converged)
}
}