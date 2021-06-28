#include "includes.h"
__global__ void  Mask_Invert_Kernel( int* A, int* devOut)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;
devOut[idx] = 1 - A[idx];
}