#include "includes.h"
__global__ void MemsetKernel(const float value, int w, int h, float *image)
{
int i = threadIdx.y + blockDim.y * blockIdx.y;
int j = threadIdx.x + blockDim.x * blockIdx.x;

if (i >= h || j >= w) return;

const int pos = i * w + j;

image[pos] = value;
}