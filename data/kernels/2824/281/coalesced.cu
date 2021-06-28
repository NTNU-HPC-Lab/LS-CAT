#include "includes.h"
__global__ void coalesced(float *A, float *C, const int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = A[i];
}