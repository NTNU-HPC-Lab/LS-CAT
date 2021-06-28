#include "includes.h"
__global__ void subMat(float *a, float *b, float *sub, int N)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;

if((idx*N) < (N*N))
sub[idx * N] = a[idx * N] - b[idx * N];
}