#include "includes.h"
__global__ void addVectors( int size, float *d_a, float *d_b, float *d_c)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;

if (tid < size)
{
d_c[tid] = d_a[tid] + d_b[tid];
}
}