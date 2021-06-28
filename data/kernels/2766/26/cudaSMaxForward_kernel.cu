#include "includes.h"
__global__ void cudaSMaxForward_kernel(unsigned int size, float* input, float* maxVal, const unsigned int idx, unsigned int* argMax)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
if (input[i] > maxVal[i]) {
maxVal[i] = input[i];
argMax[i] = idx;
}
}
}