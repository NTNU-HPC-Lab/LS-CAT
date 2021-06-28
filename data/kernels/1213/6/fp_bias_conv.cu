#include "includes.h"
__global__ void fp_bias_conv(float* preact, float* bias, const int size, const int n_channel)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int totalPos = blockDim.x * gridDim.x;

const int N = n_channel * size * size;

for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) {
int idx = n;
const int i_channel = ((idx /= 1	) % n_channel);
const int i_row = ((idx /= n_channel	) % size);
const int i_col = ((idx /= size	) % size);

preact[(i_channel * size + i_col) * size + i_row] += bias[i_channel];
}
}