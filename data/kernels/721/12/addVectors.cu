#include "includes.h"
__global__ void addVectors( float *d_A, float *d_B, float *d_C, int size)
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i < size)
{
d_C[i] = d_A[i] + d_B[i];
}
}