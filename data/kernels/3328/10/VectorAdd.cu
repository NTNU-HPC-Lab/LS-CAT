#include "includes.h"
__global__ void VectorAdd(float *VecA, float *VecB, float *VecC, int size)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < size)
VecC[i] = VecA[i] + VecB[i];
}