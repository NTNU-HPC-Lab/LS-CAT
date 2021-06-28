#include "includes.h"
__global__ void calc_avg_activation_kernel(float *src, float *dst, int size, int channels, int batches)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int xy = i % size;
int b = i / size;

if (i < size*batches) {
dst[i] = 0;
for (int c = 0; c < channels; ++c) {
dst[i] += src[xy + size*(c + channels*b)];
}
dst[i] = dst[i] / channels;
}
}