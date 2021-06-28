#include "includes.h"
__global__ void cudaSSaturation_propagate_kernel(float* x, float* y, unsigned int size, float threshold)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
float value = x[i];

if (threshold != 0.0f) {
y[i] = (value < -threshold) ? -threshold
: (value > threshold) ? threshold
: value;
}
}
}