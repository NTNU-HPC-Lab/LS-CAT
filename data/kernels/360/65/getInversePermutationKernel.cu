#include "includes.h"
__global__ void getInversePermutationKernel(int size, int *original, int *inverse)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < size)
{
inverse[original[idx]] = idx;
}
}