#include "includes.h"
__global__ void cudaSSaturation_backPropagate_kernel(float* x, float* dx, unsigned int size, float threshold)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
if (threshold != 0.0f) {
dx[i] *= (x[i] > -threshold && x[i] < threshold)
? 1.0f : 0.0f;
}
}
}