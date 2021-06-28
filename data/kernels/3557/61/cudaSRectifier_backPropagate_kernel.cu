#include "includes.h"
__global__ void cudaSRectifier_backPropagate_kernel(float* x, float* dx, unsigned int size, float leakSlope, float clipping)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
if (clipping > 0.0) {
dx[i] *= (x[i] > clipping) ? 0.0f : (x[i] > 0.0f)
? 1.0f
: leakSlope;
}
else
dx[i] *= (x[i] > 0.0f) ? 1.0f : leakSlope;
}
}