#include "includes.h"
__global__ void addMat(float *a, float *b, float *add, int N)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;

if((idx*N) < (N*N))
add[idx * N] = a[idx *N] + b[idx * N];
}