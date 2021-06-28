#include "includes.h"
__global__ void ResetHeap_kernel(int *mplHeap, int *mplHeapPtr, int numBlock)
{
int index = threadIdx.x + blockDim.x * blockIdx.x;
if (index >= numBlock)
return;

if (index == 0)
mplHeapPtr[0] = numBlock - 1;

mplHeap[index] = numBlock - index - 1;
}