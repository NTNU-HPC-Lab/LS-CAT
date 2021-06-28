#include "includes.h"
__global__ void cudaSEuclideanSumBackward_kernel(unsigned int size, float* diffInput, float* input, float* output, const float scale, const float beta, float* result)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

if (beta != 0.0f) {
for (unsigned int i = index; i < size; i += stride) {
result[i] = (output[i] != 0.0f)
? diffInput[i] * scale * (input[i] / output[i]) + beta * result[i]
: beta * result[i];
}
}
else {
for (unsigned int i = index; i < size; i += stride) {
result[i] = (output[i] != 0.0f)
? diffInput[i] * scale * (input[i] / output[i])
: 0.0f;
}
}
}