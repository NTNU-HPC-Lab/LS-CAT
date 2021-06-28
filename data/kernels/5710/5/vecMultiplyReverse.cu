#include "includes.h"
__global__ void vecMultiplyReverse(int *A, int *B, int *C)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i%2 == 0)
{
C[i] = A[i] + B[i];
}
else if(i%2 != 0)
{
C[i] = A[i] - B[i];
}
}