#include "includes.h"
__global__ void addGridThreads(int n, float *x, float *y)
{
// Let the kernel calculate which part of the input signal to play with, but
// now also include the grid information
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = index; i < n; i += stride)
y[i] = x[i] + y[i];
}