#include "includes.h"
__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int size = blockDim.x * gridDim.x;

for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
}
}