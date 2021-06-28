#include "includes.h"
__global__ void kTanh(const int nThreads, float const *input, float *output) {

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
i < nThreads;
i += blockDim.x * gridDim.x)
{
output[i] = tanh(input[i]);
}
}