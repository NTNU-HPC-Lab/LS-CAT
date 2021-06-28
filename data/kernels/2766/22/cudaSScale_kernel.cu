#include "includes.h"
__global__ void cudaSScale_kernel(unsigned int size, float* input, const float scale, const float shift, const float beta, float* result)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

if (beta != 0.0f) {
for (unsigned int i = index; i < size; i += stride)
result[i] = input[i] * scale + shift + beta * result[i];
}
else {
for (unsigned int i = index; i < size; i += stride)
result[i] = input[i] * scale  + shift;
}
}