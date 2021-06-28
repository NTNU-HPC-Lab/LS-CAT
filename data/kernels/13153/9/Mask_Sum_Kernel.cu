#include "includes.h"
__global__ void  Mask_Sum_Kernel( int* A, int valCount, int* scalarOut)
{

const int localIdx    = threadIdx.x;
const int globalIdx   = blockDim.x*blockIdx.x + threadIdx.x;
const int blockIdxOut = blockIdx.x / blockDim.x;

while(valCount > 1)
{
int localCount = blockDim.x;
while(localCount > 1)
{
localCount = localCount / 2;
if(localIdx < localCount)
A[globalIdx] += A[globalIdx + localCount];
}

if(localIdx == 0)
A[blockIdxOut] = A[globalIdx];

valCount = valCount / blockDim.x;
}

if(globalIdx==0)
scalarOut[0] = A[0];
}