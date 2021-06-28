#include "includes.h"
__global__ void decrement_dynamic_kernel(int* pInts, size_t numInts)
{
size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx > numInts)
return;

pInts[idx] -= 1;
}