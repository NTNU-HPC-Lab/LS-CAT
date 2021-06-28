#include "includes.h"
__global__ void initActiveSlices(int* buffer, int num)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i < num)
buffer[i] = i;
}