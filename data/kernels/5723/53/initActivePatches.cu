#include "includes.h"
__global__ void initActivePatches(int* buffer, int num)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i < num)
buffer[i] = i;
}