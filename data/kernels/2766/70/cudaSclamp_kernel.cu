#include "includes.h"
__global__ void cudaSclamp_kernel(float* x, unsigned int size, float minVal, float maxVal)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
x[i] = (x[i] < minVal) ? minVal :
(x[i] > maxVal) ? maxVal :
x[i];
}
}