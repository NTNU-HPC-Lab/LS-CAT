#include "includes.h"


// CUDA kernel to add elements

__global__    void add(int N, float *x)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i<N)
x[i] = x[i] *2;
}