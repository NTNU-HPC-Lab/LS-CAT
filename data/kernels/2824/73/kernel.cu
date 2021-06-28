#include "includes.h"
__global__ void kernel(float *A, float *C, const int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x*16;

if (i < N) C[i] = A[i];
}