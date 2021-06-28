#include "includes.h"
__global__ void  Mask_Union_Kernel( int* A, int* B, int* devOut)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;

if( A[idx] + B[idx] > 0)
devOut[idx] = 1;
else
devOut[idx] = 0;
}