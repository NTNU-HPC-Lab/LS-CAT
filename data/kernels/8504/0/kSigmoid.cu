#include "includes.h"
__global__ void kSigmoid(const int nThreads, float const *input, float *output) {

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
i < nThreads;
i += blockDim.x * gridDim.x)
{
output[i] = 1.0 / (1.0 + std::exp(-input[i]));
}
}