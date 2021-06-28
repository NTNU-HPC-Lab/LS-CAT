#include "includes.h"
__global__ void triad(float* A, float* B, float* C, float s)
{
int gid = threadIdx.x + (blockIdx.x * blockDim.x);
C[gid] = A[gid] + s*B[gid];
}