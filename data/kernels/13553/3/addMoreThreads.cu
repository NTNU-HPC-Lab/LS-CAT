#include "includes.h"
__global__ void addMoreThreads(int n, float *x, float *y)
{
// Let the kernel calculate which part of the input signal to play with
int index = threadIdx.x;
int stride = blockDim.x;

// Just did this to keep the syntax similar to the previous example
for (int i = index; i < n; i += stride)
y[i] = x[i] + y[i];
}