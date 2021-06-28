#include "includes.h"
__global__ void divMat(float *a, int N)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;

if((idx*N) < (N*N))
a[idx *N] /= N;
}