#include "includes.h"
__global__ void findAdjacencySizesKernel(int size, int *adjIndexes, int *output) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
output[idx] = adjIndexes[idx + 1] - adjIndexes[idx];
}
}