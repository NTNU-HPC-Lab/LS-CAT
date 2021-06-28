#include "includes.h"
__global__ void calcLoss(float *err, float *output, unsigned int Y, const int N)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int totalPos = blockDim.x * gridDim.x;

for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) {
err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
}
}