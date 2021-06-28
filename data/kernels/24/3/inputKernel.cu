#include "includes.h"
__global__ void inputKernel(float *x, int N)
{
int ix   = blockIdx.x * blockDim.x + threadIdx.x;
int iy   = blockIdx.y * blockDim.y + threadIdx.y;
int idx = iy * NUM_OF_X_THREADS + ix;

if (idx < N)
x[idx]  = x[idx] + (float)idx;
}