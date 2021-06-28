#include "includes.h"
__global__ void bp_output_fc(float *d_output, float *d_preact, float *weight, const int size, const int in_channel, const int out_channel)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int totalPos = blockDim.x * gridDim.x;

const int N = out_channel * in_channel * size * size;
const int weight_channel = out_channel * in_channel;

for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) {
int idx = n;
const int i_channel = ((idx /= 1	) % weight_channel);
const int i_row = ((idx /= weight_channel	) % size);
const int i_col = ((idx /= size	) % size);

atomicAdd(&d_output[((i_channel % in_channel) * size + i_col) * size + i_row], d_preact[i_channel % out_channel] * weight[(i_channel * size + i_col) * size + i_row]);
}
}