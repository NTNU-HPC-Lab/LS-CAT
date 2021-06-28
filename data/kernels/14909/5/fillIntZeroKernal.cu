#include "includes.h"
__global__ void fillIntZeroKernal(int *_bufferPtr,int size)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if(idx<size)
{
_bufferPtr[idx]=0;
}
}