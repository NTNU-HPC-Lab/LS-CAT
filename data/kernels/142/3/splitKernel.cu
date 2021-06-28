#include "includes.h"
__global__ void splitKernel( const unsigned int *d_In, unsigned int *d_Out, size_t size, unsigned int bitPos, unsigned int *lastValue)
{
int threadsPerBlock = blockDim.x * blockDim.y;

int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

int myId = (blockId * threadsPerBlock) + threadId;

if (myId < size)
{
unsigned int _tmp =
(d_In[myId] >> bitPos) & 0x1;
_tmp =
_tmp ? 0 : 1;

d_Out[myId] =
_tmp;

if ((myId == (size - 1)) &&
(lastValue))
{
*lastValue =
_tmp;
}
}
}