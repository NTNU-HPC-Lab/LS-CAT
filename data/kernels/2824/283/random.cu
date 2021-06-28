#include "includes.h"
__global__ void random(float *A, float *B, float *C, const int N)
{
int i = (blockIdx.x * blockDim.x + threadIdx.x);
i = B[i];

if (i < N) C[i] = A[i];
}