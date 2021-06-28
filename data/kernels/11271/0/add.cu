#include "includes.h"

// CUDA Kernel function to add the elements of two arrays on the GPU

__global__ void add(int n, float *x, float *y)
{

int index = threadIdx.x;
int stride = blockDim.x;
for (int i = index; i < n; i+= stride)
y[i] = x[i] + y[i];
}