#include "includes.h"
__global__ void mean_array_kernel(float *src, int size, float alpha, float *avg)
{
const int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i >= size) return;

avg[i] = avg[i] * (1 - alpha) + src[i] * alpha;
src[i] = avg[i];
}