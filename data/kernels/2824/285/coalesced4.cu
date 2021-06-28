#include "includes.h"
__global__ void coalesced4(float *A, float *C, const int N)
{
int i = (blockIdx.x * blockDim.x + threadIdx.x)*4;

if (i+3 < N) { C[i] = A[i]; C[i+1] = A[i+1];
C[i+2] = A[i+2]; C[i+3] = A[i+3];}
}