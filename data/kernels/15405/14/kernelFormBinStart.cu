#include "includes.h"
__global__ void kernelFormBinStart (	int* devOutputBinStart, unsigned int* devInputBinCirPairBin, unsigned int  bcPairLen)
{

__shared__ int cache[257]; //256 bcpair + the last bc pair in the previous block

int bcPairIdx = blockDim.x * blockIdx.x + threadIdx.x;

if (bcPairIdx >= bcPairLen)
{
return;
}

cache[1 + threadIdx.x] = devInputBinCirPairBin[bcPairIdx];

if ( threadIdx.x == 0 )
{
if ( bcPairIdx != 0 )
{
cache[0] = devInputBinCirPairBin[bcPairIdx - 1];
}
else
{
cache[0] = -1;
}
}

__syncthreads();

if (cache[1 + threadIdx.x] != cache[threadIdx.x])
{
//printf("b: %d, s: %d\n", cache[1 + threadIdx.x], bcPairIdx);
devOutputBinStart[cache[1 + threadIdx.x]] = bcPairIdx;
}
}