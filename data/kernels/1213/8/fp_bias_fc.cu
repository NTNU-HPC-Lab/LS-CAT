#include "includes.h"
__global__ void fp_bias_fc(float *preact, float *bias, const int n_channel)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int totalPos = blockDim.x * gridDim.x;

const int N = n_channel;

for (int idx = N * pos / totalPos; idx < N * (pos+1) / totalPos; ++idx) {
preact[idx] += bias[idx];
}
}