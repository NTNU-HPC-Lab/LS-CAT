#include "includes.h"
__global__ void _bcnn_dropout_layer_kernel(float *input, int size, float *rand, float prob, float scale)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (id < size) {
input[id] = (rand[id] < prob) ? 0 : input[id] * scale;
}
}