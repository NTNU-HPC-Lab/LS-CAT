#include "includes.h"
__global__ void  Mask_Intersect_Kernel( int* A, int* B, int* devOut)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;
devOut[idx] = A[idx] * B[idx];
}