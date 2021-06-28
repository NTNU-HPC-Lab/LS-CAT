#include "includes.h"
__global__ void kSigmoid_d(const int nThreads, float const *input, float *output) {

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
i < nThreads;
i += blockDim.x * gridDim.x)
{
output[i] = input[i] * (1 - input[i]);
}
}