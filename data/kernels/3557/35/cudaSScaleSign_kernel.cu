#include "includes.h"
__global__ void cudaSScaleSign_kernel(unsigned int size, float* input, float* sign, const float scale, const float beta, float* result)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

if (beta != 0.0f) {
for (unsigned int i = index; i < size; i += stride) {
const float sgn = (sign[i] >= 0) ? 1.0f : -1.0f;
result[i] = input[i] * sgn * scale + beta * result[i];
}
}
else {
for (unsigned int i = index; i < size; i += stride) {
const float sgn = (sign[i] >= 0) ? 1.0f : -1.0f;
result[i] = input[i] * sgn * scale;
}
}
}