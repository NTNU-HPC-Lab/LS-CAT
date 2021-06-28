#include "includes.h"
__global__ void naive_backward_cross_entropy(float *in, int *one_hot_classes, float batches, int size, float *out)
{
int bid = blockIdx.x * blockDim.x + threadIdx.x;
if (!(bid < size)) return;
out[bid] = (in[bid] - one_hot_classes[bid]) / batches;
}