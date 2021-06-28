#include "includes.h"
__global__ void backward_maxpool_depth_layer_kernel(int n, int w, int h, int c, int batch, float *delta, float *prev_delta, int *indexes)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (id >= n) return;

int index = indexes[id];
prev_delta[index] += delta[id];
}