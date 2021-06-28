#include "includes.h"
__global__ void VecAdd(float* A, float *B, float *C)
{
int idx = threadIdx.x;
C[idx] = A[idx] + B[idx];
}