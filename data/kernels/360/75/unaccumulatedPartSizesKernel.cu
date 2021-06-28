#include "includes.h"
__global__ void unaccumulatedPartSizesKernel(int size, int *accumulatedSize, int *sizes)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx == 0)
sizes[idx] = accumulatedSize[0];

else if(idx < size)
{
sizes[idx] = accumulatedSize[idx] - accumulatedSize[idx - 1];
}
}