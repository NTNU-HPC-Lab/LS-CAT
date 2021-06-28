#include "includes.h"
__global__ void  Mask_Subtract_Kernel( int* A, int* B, int* devOut)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;
if( B[idx] == 0)
devOut[idx] = A[idx];
else
devOut[idx] = 0;
}