#include "includes.h"
__global__ void  Mask_Difference_Kernel( int* A, int* B, int* devOut)
{
const int idx = blockDim.x*blockIdx.x + threadIdx.x;

if(A[idx] == B[idx])
devOut[idx] = 0;
else
devOut[idx] = 1;

// Should test if the extra algebra ops are worth removing the if-statement
// Convert to {-1, +1}
//int aval = A[idx]*2 - 1;
//int bval = B[idx]*2 - 1;
//devOut[idx] = (aval*bval+1)/2;
}