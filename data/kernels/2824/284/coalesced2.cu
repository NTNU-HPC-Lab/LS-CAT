#include "includes.h"
__global__ void coalesced2(float *A, float *C, const int N)
{
int i = (blockIdx.x * blockDim.x + threadIdx.x)*2;

if (i+1 < N) { C[i] = A[i]; C[i+1] = A[i+1];}
}