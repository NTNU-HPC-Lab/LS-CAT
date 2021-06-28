#include "includes.h"
__global__ void dotProdKernel(float *a, float *b, float *ab, int N)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;

if( (idx*N) < (N*N) ) {
ab[idx * N] = a[idx *N] * b[idx * N];
}
}