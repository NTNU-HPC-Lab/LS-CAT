#include "includes.h"
__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
if(i<n) //boundary check for threads.
{
C[i] = A[i] + B[i];
}
}