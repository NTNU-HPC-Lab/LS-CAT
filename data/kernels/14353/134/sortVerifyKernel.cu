#include "includes.h"
__global__ void sortVerifyKernel(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *errNum)
{
uint idx = blockIdx.x * blockDim.x + threadIdx.x;
uint iterator;

#pragma unroll
for (iterator = 0; iterator < THREAD_SIZE; iterator++)
if ((d_SrcKey[d_DstVal[idx*THREAD_SIZE + iterator]] != d_DstKey[idx*THREAD_SIZE + iterator]))
atomicAdd(errNum, 1);
}