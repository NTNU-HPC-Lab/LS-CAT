#include "includes.h"
__global__ void skip_128b(float *A, float *C, const int N)
{
int i = (blockIdx.x * blockDim.x + threadIdx.x)+32*(threadIdx.x%32);

if (i < N) C[i] = A[i];
}