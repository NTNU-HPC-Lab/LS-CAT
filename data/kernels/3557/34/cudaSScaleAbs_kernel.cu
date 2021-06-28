#include "includes.h"
__global__ void cudaSScaleAbs_kernel(unsigned int size, float* input, const float scale, const float beta, float* result)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

if (beta != 0.0f) {
for (unsigned int i = index; i < size; i += stride)
result[i] = fabs(input[i]) * scale + beta * result[i];
}
else {
for (unsigned int i = index; i < size; i += stride)
result[i] = fabs(input[i]) * scale;
}
}