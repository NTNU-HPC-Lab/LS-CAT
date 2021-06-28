#include "includes.h"
__global__ void fp_preact_fc(float* input, float* preact, float* weight, const int size, const int in_channel, const int out_channel)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int totalPos = blockDim.x * gridDim.x;

const int weight_channel = in_channel * out_channel;
const int N = out_channel * in_channel * size * size;  // number of elements of weight matrix

for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) {
int idx = n;
const int i_channel = ((idx /= 1	) % weight_channel);
const int i_row = ((idx /= weight_channel	) % size);
const int i_col = ((idx /= size	) % size);

atomicAdd(&preact[i_channel % out_channel], weight[(i_channel * size + i_col) * size + i_row] * input[((i_channel % in_channel) * size + i_col) * size + i_row]);
}
}