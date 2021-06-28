#include "includes.h"
__global__ void run_reduction(int *con, int *blockCon,int* ActiveList, int nActiveBlock, int* blockSizes)
{
int list_idx = blockIdx.y*gridDim.x + blockIdx.x;


if(list_idx < nActiveBlock)
{
int block_idx = ActiveList[list_idx];

__shared__ int s_conv[REDUCTIONSHARESIZE];


uint base_addr = block_idx*blockDim.x*2;   // *2 because there are only half block size number of thread
uint tx = threadIdx.x;


s_conv[tx] = con[base_addr + tx];
s_conv[tx + blockDim.x] = con[base_addr + tx + blockDim.x];

__syncthreads();

for(uint i=blockDim.x; i>0; i/=2)
{
if(tx < i)
{
bool b1, b2;
b1 = s_conv[tx];
b2 = s_conv[tx+i];
s_conv[tx] = (b1 && b2) ? 1 : 0 ;
}
__syncthreads();
}

if(tx == 0)
{
blockCon[block_idx] = s_conv[0]; // active list is negation of tile convergence (active = not converged)
}
}
}