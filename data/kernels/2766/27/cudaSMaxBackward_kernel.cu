#include "includes.h"
__global__ void cudaSMaxBackward_kernel(unsigned int size, float* diffInput, const unsigned int idx, unsigned int* argMax, const float beta, float* result)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

if (beta != 0.0f) {
for (unsigned int i = index; i < size; i += stride) {
result[i] = (argMax[i] == idx) ? (diffInput[i] + beta * result[i])
: beta * result[i];
}
}
else {
for (unsigned int i = index; i < size; i += stride) {
result[i] = (argMax[i] == idx) ? diffInput[i]
: 0.0f;
}
}
}