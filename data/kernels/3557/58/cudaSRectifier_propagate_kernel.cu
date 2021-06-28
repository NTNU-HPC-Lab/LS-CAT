#include "includes.h"
__global__ void cudaSRectifier_propagate_kernel(float* x, float* y, unsigned int size, float leakSlope, float clipping)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
float value = x[i];

if (clipping > 0.0f)
y[i] = (value > 0.0f) ? min(value, clipping) : leakSlope * value;
else
y[i] = (value > 0.0f) ? value : leakSlope * value;
}
}